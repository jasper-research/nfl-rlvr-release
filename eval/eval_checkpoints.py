"""Phase 5 — evaluate the GRPO checkpoint ladder on a held-out split (on the pod, GPU).

Loads the base model ONCE in vLLM, then hot-swaps each LoRA checkpoint as a separate
generation pass — so zero-shot (no adapter), checkpoint-50, checkpoint-100, ... all run off a
single weight load. Reuses the real metric/parse code (reward.metrics, reward.extract); only the
generation layer differs from eval/evaluate.py (offline vLLM instead of an HTTP backend).

This is the authoritative test the training curves only *suggest*: does the adapter actually
lower held-out ECE/Brier vs the zero-shot base and approach Vegas?

Run on the pod (stop training first to free the GPU):
    python -m eval.eval_checkpoints --split eval --n 1000

    # explicit adapters / knobs:
    python -m eval.eval_checkpoints --split eval --n 1000 \
        --base Qwen/Qwen2.5-7B-Instruct \
        --output-dir checkpoints/grpo-qwen2.5-7b \
        --max-tokens 1024 --temperature 0.0 --gpu-mem-util 0.85

Writes eval/results/checkpoints_<split>.json (every variant + Vegas, full metric blocks) and
prints a side-by-side table. Greedy (temperature 0) by default for a deterministic comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reward.extract import extract_probability          # noqa: E402
from eval.evaluate import metrics_block                 # noqa: E402  (reuse the exact metric block)
from train.grpo_train import _transform                 # noqa: E402  (canonical direct/masked xform)

RESULTS = Path(__file__).resolve().parent / "results"

# v10: match the no-CoT direct-answer training prompt (train/grpo_train.py::to_direct).
_COT = "Reason step by step, then give a final probability as a percentage."
_DIRECT = "Respond with ONLY the win probability as a percentage and nothing else (for example: 63%)."


def load_split(split: str, n: int):
    rows = [json.loads(l) for l in open(f"data/grpo/{split}.jsonl")]
    return rows if n <= 0 else rows[:n]


def discover_adapters(output_dir: str, explicit: list[str] | None):
    """Return [(label, path)] for each LoRA adapter dir, sorted by training step.

    A dir is an adapter iff it contains adapter_config.json. Includes checkpoint-* and the
    final output_dir itself (the last save_model)."""
    if explicit:
        cands = [Path(p) for p in explicit]
    else:
        out = Path(output_dir)
        cands = sorted(out.glob("checkpoint-*"),
                       key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1)))
        if (out / "adapter_config.json").exists():
            cands.append(out)  # final adapter
    found = []
    for p in cands:
        if (p / "adapter_config.json").exists():
            label = p.name if p.name.startswith("checkpoint-") else "final"
            found.append((label, str(p)))
        else:
            print(f"  (skip {p}: no adapter_config.json)")
    return found


def generate(llm, sampling, messages, lora_request):
    from vllm import SamplingParams  # noqa: F401  (type only)
    outs = llm.chat(messages, sampling, lora_request=lora_request, use_tqdm=True)
    return [o.outputs[0] for o in outs]


def score(gen_outputs, rows):
    """gen_outputs: list of vLLM CompletionOutput. Return per-row {prob,outcome,vegas,trunc,parsed}."""
    preds = []
    for o, r in zip(gen_outputs, rows):
        text = o.text
        truncated = (o.finish_reason == "length")
        p = extract_probability(text)
        preds.append({"game_id": r.get("game_id"), "prob": p, "outcome": int(r["actual_outcome"]),
                      "vegas": r["vegas_wp"], "truncated": truncated, "parsed": p is not None,
                      "text": text})
    return preds


def variant_result(label, preds):
    clean = [d for d in preds if d["parsed"] and not d["truncated"]]
    probs = [d["prob"] for d in clean]
    outs = [d["outcome"] for d in clean]
    vegas = [d["vegas"] for d in clean]
    m = metrics_block(probs, outs)
    m_vegas = metrics_block(vegas, outs)  # Vegas on the SAME clean items (apples-to-apples)
    return {
        "name": label,
        "n_total": len(preds),
        "coverage": len(clean) / max(1, len(preds)),
        "parse_rate": sum(d["parsed"] for d in preds) / max(1, len(preds)),
        "truncation": sum(d["truncated"] for d in preds) / max(1, len(preds)),
        "mean_completion_chars": sum(len(d["text"]) for d in preds) / max(1, len(preds)),
        "metrics": m,
        "vegas_on_same_items": m_vegas,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--output-dir", default="checkpoints/grpo-qwen2.5-7b")
    ap.add_argument("--adapters", nargs="*", default=None,
                    help="explicit adapter dirs; default = auto-discover checkpoint-* + final")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=1000, help="0 = all")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--direct", action="store_true", help="no-CoT direct-answer prompt (match v10/WS1)")
    ap.add_argument("--masked", action="store_true", help="WS2 masked-CoT prompt (Probability: NN%) — "
                    "REQUIRED for ws2 checkpoints so they see their trained instruction")
    ap.add_argument("--tag", default="", help="suffix for the output json (avoid clobbering, e.g. ws2)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (deterministic)")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-preds", action="store_true",
                    help="also dump per-game predictions (game_id/prob/outcome/vegas) per variant to "
                         "preds_<split>_<tag>.json — needed for the paired bootstrap")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = load_split(args.split, args.n)
    messages = [[{"role": "user", "content": _transform(r["prompt"], args.direct, args.masked)}]
                for r in rows]
    adapters = discover_adapters(args.output_dir, args.adapters)
    print(f"[eval_checkpoints] split={args.split} n={len(rows)}  variants: "
          f"zeroshot + {[a[0] for a in adapters]}")

    llm = LLM(model=args.base, enable_lora=True, max_lora_rank=args.max_lora_rank,
              dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, trust_remote_code=True)
    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_tokens, seed=args.seed)

    results = {}
    all_preds = {}   # label -> per-game preds (for --save-preds / paired bootstrap)

    # 1) zero-shot base (no adapter)
    print("\n=== zeroshot (base, no adapter) ===")
    preds = score(generate(llm, sampling, messages, None), rows)
    results["zeroshot"] = variant_result("zeroshot", preds)
    all_preds["zeroshot"] = preds

    # 2) each checkpoint adapter
    for i, (label, path) in enumerate(adapters, start=1):
        print(f"\n=== {label}  ({path}) ===")
        req = LoRARequest(label, i, path)
        preds = score(generate(llm, sampling, messages, req), rows)
        results[label] = variant_result(label, preds)
        all_preds[label] = preds

    # 3) Vegas on the full eval set (ceiling reference)
    vprobs = [r["vegas_wp"] for r in rows]
    vouts = [int(r["actual_outcome"]) for r in rows]
    results["vegas"] = {"name": "vegas", "n_total": len(rows), "coverage": 1.0,
                        "metrics": metrics_block(vprobs, vouts)}

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"checkpoints_{args.split}{('_' + args.tag) if args.tag else ''}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))

    if args.save_preds:
        # per-game preds per variant (+ vegas/outcome carried on every row) for the paired bootstrap
        slim = {lbl: [{"game_id": d["game_id"], "prob": d["prob"], "outcome": d["outcome"],
                       "vegas": d["vegas"], "parsed": d["parsed"], "truncated": d["truncated"]}
                      for d in preds_list]
                for lbl, preds_list in all_preds.items()}
        preds_path = RESULTS / f"preds_{args.split}{('_' + args.tag) if args.tag else ''}.json"
        preds_path.write_text(json.dumps(slim))
        print(f"-> per-game preds: {preds_path}")

    # --- comparison table ---
    order = ["zeroshot"] + [a[0] for a in adapters] + ["vegas"]
    print("\n" + "#" * 78)
    print(f"{'variant':<16}{'n':>6}{'cov':>7}{'Brier':>9}{'ECE':>8}{'MCE':>8}{'acc':>7}{'chars':>8}")
    print("-" * 78)
    for k in order:
        r = results[k]
        m = r["metrics"]
        cov = f"{r.get('coverage', 1.0):.0%}"
        chars = f"{r.get('mean_completion_chars', 0):.0f}" if "mean_completion_chars" in r else "-"
        print(f"{k:<16}{m['n']:>6}{cov:>7}{m['brier']:>9.4f}{m['ece']:>8.4f}"
              f"{m['mce']:>8.4f}{m['accuracy']:>7.3f}{chars:>8}")
    print("#" * 78)
    print("Lower Brier/ECE = better. Vegas ECE≈0.02 is the ceiling; zero-shot is the 'before'.")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
