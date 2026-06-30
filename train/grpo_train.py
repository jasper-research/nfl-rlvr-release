"""Phase 4 — GRPO training (RunPod A100 80GB).

TRL GRPOTrainer + vLLM rollouts, QLoRA 4-bit on Qwen/Qwen3.6-35B-A3B. Reward = per-sample Brier
on the realized outcome (reward/grpo_reward.py). Option (c): generous completion budget, pure
Brier, KL on. Inference/deployment is done later locally in MLX.

Run on the pod (single A100 80GB):
    accelerate launch -m train.grpo_train --config train/config.yaml
    # or: python -m train.grpo_train --config train/config.yaml

Validate the loop locally first (tiny model, CPU/MPS, no 4-bit/vLLM):
    python -m train.grpo_train --config train/config.yaml --smoke

CAVEATS (read before the run):
  * Qwen3.6-35B-A3B is an "Image-Text-to-Text" (multimodal) MoE. We use the text path only.
    If AutoModelForCausalLM can't load it, switch to the repo's documented text-LM class.
  * TRL's GRPO API moves fast — a few GRPOConfig kwargs below are set defensively (try/except).
  * LoRA target_modules may need adjusting for the MoE expert/router naming — verify on the pod.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# repo root on path so `reward` imports when launched as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reward.extract import extract_probability          # noqa: E402
from reward.grpo_reward import make_reward_fn            # noqa: E402
from reward.metrics import (                             # noqa: E402
    brier_score,
    expected_calibration_error,
    murphy_decomposition,
)


# --------------------------------------------------------------------------- config

def load_cfg(path: str, smoke: bool) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if smoke:
        s = cfg.get("smoke", {})
        cfg["model"]["name"] = s.get("model", "Qwen/Qwen3-0.6B")
        cfg["model"]["load_in_4bit"] = False
        cfg["grpo"]["use_vllm"] = False
        cfg["grpo"]["max_steps"] = s.get("max_steps", 3)
        cfg["grpo"]["num_generations"] = s.get("num_generations", 4)
        cfg["grpo"]["max_completion_length"] = s.get("max_completion_length", 256)
        cfg["grpo"]["per_device_train_batch_size"] = 2
        cfg["grpo"]["gradient_accumulation_steps"] = 1
        cfg["grpo"]["save_steps"] = 9999
    return cfg


# --------------------------------------------------------------------------- data

# v10 (Path A): strip the chain-of-thought so RL can't corrupt the reasoning — the model emits the
# probability directly. Done at load time (no data regen). Applied to BOTH train and eval prompts.
_COT_INSTRUCTION = "Reason step by step, then give a final probability as a percentage."
_DIRECT_INSTRUCTION = "Respond with ONLY the win probability as a percentage and nothing else (for example: 63%)."
# WS2: CoT kept, but a fixed answer marker so the masked trainer can find the answer span.
from .masked_grpo import MASK_MARKER   # noqa: E402  ("Probability:")
_MASKED_INSTRUCTION = ("Reason step by step. Then, on a new line, give your final answer in exactly "
                       f"this format: {MASK_MARKER} NN%   (NN = the win probability).")


def to_direct(prompt: str) -> str:
    return prompt.replace(_COT_INSTRUCTION, _DIRECT_INSTRUCTION)


def to_masked(prompt: str) -> str:
    return prompt.replace(_COT_INSTRUCTION, _MASKED_INSTRUCTION)


def _transform(prompt: str, direct: bool, masked: bool) -> str:
    return to_masked(prompt) if masked else (to_direct(prompt) if direct else prompt)


def build_dataset(path: str, direct: bool = False, masked: bool = False):
    """Each row -> conversational prompt (so the chat/thinking template is applied) plus the
    columns the reward needs. TRL passes non-'prompt' columns to the reward fn by name."""
    from datasets import load_dataset

    ds = load_dataset("json", data_files=path, split="train")

    def to_chat(row):
        content = _transform(row["prompt"], direct, masked)
        return {"prompt": [{"role": "user", "content": content}],
                "actual_outcome": int(row["actual_outcome"]),
                "target": row.get("target")}        # soft p̂. vegas_wp DELIBERATELY EXCLUDED from the
                                                    # training dataset — it is eval-only (the baseline).

    keep = {"prompt", "actual_outcome", "target"}   # no vegas_wp reaches the model or reward
    return ds.map(to_chat, remove_columns=[c for c in ds.column_names if c not in keep])


# --------------------------------------------------------------------------- model

def load_model_and_tokenizer(mcfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mcfg["name"], trust_remote_code=mcfg.get("trust_remote_code", True))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  # CPU/MPS smoke -> fp32
    kwargs = dict(
        trust_remote_code=mcfg.get("trust_remote_code", True),
        dtype=dtype,                                    # `torch_dtype` is deprecated in newer transformers
        attn_implementation=mcfg.get("attn_implementation", "sdpa"),
    )
    if mcfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=mcfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, mcfg.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(mcfg["name"], **kwargs)
    return model, tok


def lora_config(lcfg: dict):
    from peft import LoraConfig
    return LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    )


# --------------------------------------------------------------------------- eval callback

class CalibrationEval:
    """Held-out calibration check every save_steps: greedily generate on a fixed eval subset,
    parse probabilities, log ECE/Brier/Murphy vs realized outcomes (and the Vegas baseline) to
    W&B so the held-out curve is visible *during* training (the training-batch metrics aren't
    trustworthy — see v1). Batched HF .generate, fully guarded so a failure never kills training.
    Greedy + small subset keeps it fast (~2-3 min) and low-VRAM (won't OOM next to vLLM colocate).
    The authoritative, full n>=1000 eval still lives in eval/eval_checkpoints.py post-run."""

    def __init__(self, eval_path, tokenizer, subset, max_new_tokens, batch_size,
                 direct=False, masked=False):
        import json, random
        all_rows = [json.loads(l) for l in Path(eval_path).open()]
        # FIXED RANDOM sample (not first-N): representative + identical across runs/steps, so the
        # step-0 baseline and every checkpoint are scored on the SAME games (apples-to-apples).
        self.rows = random.Random(1234).sample(all_rows, min(subset, len(all_rows)))
        for r in self.rows:                         # match the training prompt (direct / masked-CoT)
            r["prompt"] = _transform(r["prompt"], direct, masked)
        self.tok = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def __call__(self, model, step):
        import time
        import torch, wandb
        try:
            tok = self.tok
            prev_side = tok.padding_side
            tok.padding_side = "left"          # left-pad so generated tokens align across the batch
            model.eval()
            texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                             add_generation_prompt=True, tokenize=False)
                     for r in self.rows]
            device = next(model.parameters()).device
            n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            t0 = time.time()
            print(f"[eval @ {step}] generating {len(texts)} held-out prompts "
                  f"({n_batches} batches of {self.batch_size}, greedy)...", flush=True)
            gen_texts = []
            with torch.no_grad():
                for bi, i in enumerate(range(0, len(texts), self.batch_size), 1):
                    chunk = texts[i:i + self.batch_size]
                    enc = tok(chunk, return_tensors="pt", padding=True,
                              add_special_tokens=False).to(device)
                    out = model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                         do_sample=False, pad_token_id=tok.pad_token_id)
                    gen = out[:, enc["input_ids"].shape[1]:]   # strip the (left-padded) prompt
                    gen_texts.extend(tok.batch_decode(gen, skip_special_tokens=True))
                    print(f"[eval @ {step}]   batch {bi}/{n_batches}  "
                          f"({len(gen_texts)}/{len(texts)} prompts, {time.time()-t0:.0f}s)", flush=True)
            model.train()
            tok.padding_side = prev_side
            torch.cuda.empty_cache()
            print(f"[eval @ {step}] generation done in {time.time()-t0:.0f}s; scoring...", flush=True)

            probs, outs, vegas = [], [], []
            for text, r in zip(gen_texts, self.rows):
                p = extract_probability(text)
                if p is None:
                    continue
                probs.append(p); outs.append(int(r["actual_outcome"])); vegas.append(r["vegas_wp"])
            if len(probs) < 10:
                print(f"[eval @ {step}] only {len(probs)} parsed; skipping log")
                return
            payload = {
                "eval/format_success": len(probs) / len(self.rows),
                "eval/brier": brier_score(probs, outs),
                "eval/ece": expected_calibration_error(probs, outs),
                "eval/vegas_brier": brier_score(vegas, outs),
                "eval/mean_predicted_prob": sum(probs) / len(probs),
                "eval/global_step": step,          # no explicit wandb step (clashes -> dropped)
            }
            payload.update({f"eval/{k}": v for k, v in murphy_decomposition(probs, outs).items()})
            if wandb.run is not None:
                wandb.log(payload)
            print(f"[eval @ {step}] ", {k: round(v, 4) for k, v in payload.items()
                                        if k != "eval/global_step"})
        except Exception as e:  # never let eval kill training; surface the real cause once
            import traceback
            model.train()
            print(f"[eval @ {step}] skipped: {e}\n{traceback.format_exc()}")


def make_trainer_callback(cal_eval, save_steps):
    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        # No step-0 eval: the base model is identical every run; baseline already measured
        # (Brier 0.2525 / ECE 0.1917 on the fixed 128). Eval only at each checkpoint save.
        def on_save(self, args, state, control, model=None, **kw):
            if cal_eval and model is not None:
                cal_eval(model, state.global_step)

    return _CB()


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="train/config.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny model, few steps, no 4bit/vLLM")
    ap.add_argument("--max-steps", type=int, default=None, help="override grpo.max_steps (e.g. 5)")
    ap.add_argument("--no-vllm", action="store_true", help="use HF generation instead of vLLM")
    ap.add_argument("--no-eval", action="store_true", help="skip in-training held-out eval (fast diagnostics)")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in output_dir")
    ap.add_argument("--no-4bit", action="store_true", help="load bf16 + LoRA instead of QLoRA 4-bit")
    # --- WS1 sweep overrides (so we don't proliferate config files) ---
    ap.add_argument("--lr", type=float, default=None, help="override grpo.learning_rate")
    ap.add_argument("--beta", type=float, default=None, help="override grpo.beta (KL)")
    ap.add_argument("--num-generations", type=int, default=None, help="override grpo.num_generations")
    ap.add_argument("--temperature", type=float, default=None, help="override grpo.temperature")
    ap.add_argument("--no-soft-target", action="store_true", help="reward vs raw 0/1 outcome, not p̂")
    ap.add_argument("--blend-lambda", type=float, default=None, help="reward = λ·p̂ + (1-λ)·y")
    ap.add_argument("--grad-accum", type=int, default=None, help="override gradient_accumulation_steps")
    ap.add_argument("--no-scale-rewards", action="store_true", help="set scale_rewards=false (Dr-GRPO)")
    ap.add_argument("--seed", type=int, default=None, help="override seed (for variance runs)")
    ap.add_argument("--tag", default=None, help="output_dir -> checkpoints/grpo-<tag>")
    ap.add_argument("--mask-reasoning", action="store_true",
                    help="WS2: keep CoT but RL only the answer (Probability:) tokens via masked GRPO")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.smoke)
    g = cfg["grpo"]
    if args.max_steps is not None:
        g["max_steps"] = args.max_steps
    if args.no_vllm:
        g["use_vllm"] = False
    if args.no_4bit:
        cfg["model"]["load_in_4bit"] = False
    if args.lr is not None:
        g["learning_rate"] = args.lr
    if args.beta is not None:
        g["beta"] = args.beta
    if args.num_generations is not None:
        g["num_generations"] = args.num_generations
    if args.temperature is not None:
        g["temperature"] = args.temperature
    if args.no_soft_target:
        cfg.setdefault("reward", {})["use_soft_target"] = False
    if args.blend_lambda is not None:
        cfg.setdefault("reward", {})["blend_lambda"] = args.blend_lambda
    if args.grad_accum is not None:
        g["gradient_accumulation_steps"] = args.grad_accum
    if args.no_scale_rewards:
        g["scale_rewards"] = False
    if args.seed is not None:
        g["seed"] = args.seed
    if args.tag:
        cfg["output_dir"] = f"checkpoints/grpo-{args.tag}"
    if args.mask_reasoning:                          # WS2: CoT back, RL only the answer tokens
        g["direct_answer"] = False                   # keep the chain-of-thought
        g["max_completion_length"] = max(int(g.get("max_completion_length", 0)), 640)
        cfg.setdefault("eval", {})["max_new_tokens"] = max(
            int(cfg.get("eval", {}).get("max_new_tokens", 0)), 640)

    import os
    import torch
    _cuda = torch.cuda.is_available()
    os.environ.setdefault("WANDB_PROJECT", cfg.get("wandb_project", "nfl-rlvr-calibration"))

    from trl import GRPOConfig, GRPOTrainer
    if args.mask_reasoning:
        from .masked_grpo import make_masked_trainer_cls
        TrainerCls = make_masked_trainer_cls()
    else:
        TrainerCls = GRPOTrainer

    direct = g.get("direct_answer", False)
    train_ds = build_dataset(cfg["data"]["train"], direct=direct, masked=args.mask_reasoning)
    model, tok = load_model_and_tokenizer(cfg["model"])
    peft_cfg = lora_config(cfg["lora"])

    # Map our config -> GRPOConfig, setting version-sensitive kwargs defensively.
    grpo_kwargs = dict(
        output_dir=cfg["output_dir"],
        num_generations=g["num_generations"],
        max_prompt_length=g["max_prompt_length"],
        max_completion_length=g["max_completion_length"],
        temperature=g["temperature"],
        top_p=g["top_p"],
        beta=g["beta"],
        learning_rate=float(g["learning_rate"]),
        per_device_train_batch_size=g["per_device_train_batch_size"],
        gradient_accumulation_steps=g["gradient_accumulation_steps"],
        num_iterations=g.get("num_iterations", 1),
        max_steps=g["max_steps"],
        warmup_steps=g.get("warmup_steps", 10),
        logging_steps=g.get("logging_steps", 10),
        save_steps=g.get("save_steps", 50),
        bf16=_cuda,                                       # bf16 only on CUDA; fp32 on CPU/MPS smoke
        report_to=("none" if args.smoke else "wandb"),    # smoke: no wandb dependency
        use_vllm=g.get("use_vllm", False),
    )
    for opt_key in ("loss_type", "scale_rewards", "vllm_gpu_memory_utilization",
                    "importance_sampling_level", "vllm_importance_sampling_correction", "seed",
                    "save_only_model"):
        if opt_key in g:
            grpo_kwargs[opt_key] = g[opt_key]
    # Robust to TRL API drift across versions: keep only kwargs this GRPOConfig accepts.
    import dataclasses
    import trl
    accepted = {f.name for f in dataclasses.fields(GRPOConfig)}
    # max_prompt_length isn't a GRPOConfig field in some TRL versions (e.g. 1.6.0). Harmless here:
    # prompts are ~300 tokens, far below any limit — nothing to truncate. Don't warn on known drops.
    _benign_drops = {"max_prompt_length"}
    dropped = sorted(k for k in grpo_kwargs if k not in accepted and k not in _benign_drops)
    if dropped:
        print(f"[grpo_train] TRL {trl.__version__} GRPOConfig ignores: {dropped}")
    grpo_args = GRPOConfig(**{k: v for k, v in grpo_kwargs.items() if k in accepted})

    reward_fn = make_reward_fn(
        use_soft_target=cfg["reward"].get("use_soft_target", True),
        blend=cfg["reward"].get("blend_lambda", 1.0),
        sharpness_weight=cfg["reward"]["sharpness_weight"],
        format_penalty=cfg["reward"]["format_penalty"],
        clip=tuple(cfg["reward"]["clip"]),
    )

    cal = None
    if not args.smoke and not args.no_eval and cfg.get("eval", {}).get("subset"):
        ecfg = cfg["eval"]
        cal = CalibrationEval(cfg["data"]["eval"], tok, ecfg["subset"],
                              ecfg.get("max_new_tokens", 640), ecfg.get("batch_size", 4),
                              direct=direct, masked=args.mask_reasoning)

    from transformers import TrainerCallback

    class _MetricsCallback(TrainerCallback):
        """Flush our reward diagnostics at the trainer's global_step (shared x-axis with TRL)."""
        def on_log(self, args, state, control, **kw):
            try:
                import wandb
                if wandb.run is None:
                    return
                metrics = reward_fn.pop_metrics()
                if metrics:
                    # no explicit step: wandb auto-advances it, matching TRL's own logs.
                    # Forcing step=global_step clashes with per-step profiling logs ("step N
                    # < current") and gets dropped.
                    metrics["train/global_step"] = state.global_step
                    wandb.log(metrics)
            except Exception as e:
                print(f"[metrics] skipped: {e}")

    callbacks = [_MetricsCallback()]
    if cal:
        callbacks.append(make_trainer_callback(cal, g.get("save_steps", 50)))

    trainer = TrainerCls(            # MaskedGRPOTrainer if --mask-reasoning else GRPOTrainer
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_args,
        train_dataset=train_ds,
        peft_config=peft_cfg,
        processing_class=tok,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(cfg["output_dir"])
    print(f"done -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
