# RunPod runbook — GRPO training for Qwen2.5-7B-Instruct

End-to-end: pick a pod → get code + data on it → **don't** copy model weights (download from HF) →
set up W&B → train → pull the (small) adapter back → convert for local MLX inference.

The GRPO loop is already validated locally (`--smoke` passed on TRL 1.6.0). This run is the real one.

---

## 0. Pod & GPU choice

Qwen2.5-7B + QLoRA GRPO is **small** — it does NOT need an A100 80GB.

| GPU | VRAM | Fits 7B QLoRA GRPO? | ~$/hr | Notes |
|---|---|---|---|---|
| RTX 4090 / A40 | 24 / 48 GB | ✅ (4090 tight, A40 comfortable) | ~$0.4–0.8 | cheapest viable |
| **A100 80GB** | 80 GB | ✅ lots of headroom (could full-FT) | ~$1.5–2 | **recommended** — fast, simple |
| H100 | 80 GB | ✅ | ~$2.5–3 | fastest; cheaper per-run if time-bound |

- Use **Secure Cloud** (or Community for ablations — ~half price).
- Attach a **persistent volume** (~50 GB, ~$0.07/GB/mo) mounted at `/workspace` so the HF model
  cache + checkpoints survive pod restarts and you never re-download weights.
- Lifecycle: **spin up → train → pull adapter → terminate.** No idle cost.

Cost estimate: 7B, ~300–500 steps, concise rollouts → **~$5–20** for the hero run (far below the
old 35B estimate).

---

## 1. Get the CODE onto the pod

The repo is a local git repo with **no remote**, and data artifacts are `.gitignore`d. Pick one:

**Option A — GitHub (recommended, reproducible):**
```bash
# on your Mac, once: create a private repo and push
cd paper2_fanhuddle
gh repo create nfl-rlvr --private --source=. --push   # or: git remote add origin <url> && git push -u origin main
# on the pod:
cd /workspace && git clone https://github.com/<you>/nfl-rlvr.git && cd nfl-rlvr
```

**Option B — `runpodctl` (no GitHub):**
```bash
# Mac: zip the code (exclude venvs/data), send
cd paper2_fanhuddle
git archive --format=tar.gz -o /tmp/code.tgz HEAD
runpodctl send /tmp/code.tgz          # prints a one-time code
# pod:
runpodctl receive <code> && tar xzf code.tgz -C /workspace/nfl-rlvr
```

(`scp`/`rsync` over the pod's SSH also works.)

## 2. Get the DATA onto the pod — regenerate, don't copy

`data/grpo/` is gitignored and trivially reproducible from free public data. On the pod:
```bash
cd /workspace/nfl-rlvr
python -m pip install -r requirements.txt          # nfl_data_py + pandas (CPU)
python -m data.collect_pbp --years 2015-2024 --out data/raw
python -m data.build_qa_pairs --raw data/raw/pbp_2015_2024.parquet --out data/qa --stride 8 --split-seasons
python -m data.build_grpo_dataset                  # -> data/grpo/{train,eval,test}.jsonl
```
(If you'd rather copy: the three `data/grpo/*.jsonl` are a few MB — `runpodctl send`/`scp` them.)

## 3. MODEL WEIGHTS — do NOT transfer from the Mac

The Mac has an **MLX-4bit** build (`Qwen2.5-7B-Instruct-MLX-4bit`); training needs the **HF**
weights, and the trainer downloads them directly — fast on a datacenter link. Just point the HF
cache at the persistent volume so it's cached once:
```bash
export HF_HOME=/workspace/hf-cache        # persists across restarts
# first run auto-downloads Qwen/Qwen2.5-7B-Instruct (~15 GB bf16) into HF_HOME
```

## 4. Install the training stack + set up W&B monitoring

```bash
cd /workspace/nfl-rlvr
python -m pip install -r requirements-train.txt    # torch, transformers, trl, peft, accelerate, vllm, wandb
# W&B: get your key from https://wandb.ai/authorize
export WANDB_API_KEY=...          # or: wandb login
# project name is read from config (wandb_project: nfl-rlvr-calibration)
```
The training script sets `report_to="wandb"` and logs, every step:
`reward` / `reward_std`, `kl`, `entropy`, `completions/mean_length` (TRL native) **plus** our
`train/mean_predicted_prob`, `train/mean_confidence`, `train/pred_prob_std` (collapse detector),
`train/batch_ece`, `train/format_success_rate`. Watch these live at wandb.ai.

**Live health rules** (from the plan): `mean_confidence`↑ & reward↑ = working; `mean_confidence`↑ &
Brier↓ = overconfident → raise `beta` (KL); `pred_prob_std`→0 or flat `mean_confidence` =
collapse → enable the sharpness bonus; `entropy` collapse → stop.

## 5. Train

```bash
export HF_HOME=/workspace/hf-cache WANDB_API_KEY=...
accelerate launch -m train.grpo_train --config train/config.yaml
#   or, single-GPU:  python -m train.grpo_train --config train/config.yaml
```
Checkpoints (LoRA adapters) save every `save_steps` to `checkpoints/grpo-qwen2.5-7b/` on the
persistent volume. A held-out calibration eval (ECE/Brier/Murphy vs Vegas) runs each save via the
built-in callback and logs to W&B under `eval/*`.

## 6. Pull the result back — only the adapter (small)

LoRA adapters are tens of MB (not the 15 GB base). From the pod:
```bash
cd /workspace/nfl-rlvr/checkpoints && tar czf adapter.tgz grpo-qwen2.5-7b
runpodctl send adapter.tgz        # receive on the Mac with the printed code
```

## 7. Back on the Mac — eval + local inference

- **Large-n eval on the GPU before terminating** (recommended): serve the trained model with vLLM
  on the pod (OpenAI-compatible) and point `eval/evaluate.py --backend vllm --vllm-url ... --n 1000`
  at it — fast, gives the publishable n≥1000 numbers for trained + zero-shot in one sitting.
- **Local MLX inference:** merge the LoRA into the base (`peft` `merge_and_unload`) → convert to MLX
  (`mlx_lm.convert`) → serve in oMLX, exactly like the baselines. Then
  `eval/evaluate.py --backend omlx --model <merged-mlx>` for the trained "after" row.

---

## Gotchas (already handled / to watch)
- TRL API drifts: `grpo_train.py` filters `GRPOConfig` kwargs to whatever the installed TRL accepts
  (TRL 1.6 dropped `max_prompt_length`; prompts are short, so it's a no-op).
- `generation_batch_size` (per_device_bs × grad_accum × world_size) **must be divisible by
  `num_generations`**. Real config: 2×8 = 16, divisible by 8 ✓. If you change batch/accum, keep this.
- `flash-attn` install is slow; `attn_implementation: sdpa` (default) is fine.
- bf16 is enabled only on CUDA (the script forces fp32 off-GPU); on the A100 you get bf16.
