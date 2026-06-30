# Verifiable Rewards for Calibrated Probabilistic Forecasting

Code and paper for training a 7B language model, with reinforcement learning and a verifiable
label-free reward, to produce calibrated NFL in-game win-probability forecasts at the level of the
betting market. No human labels and no supervised fine-tuning.

- **Paper:** [`paper/main.pdf`](paper/main.pdf) (LaTeX source under `paper/`).
- **Data and trained adapters:** archived on Zenodo, DOI `10.5281/zenodo.21082572` (see
  [Data and models](#data-and-models)).

> Authorship withheld during anonymous review.

## Method

The forecaster reads a public game state (score margin, time, down and distance, field position,
possession, pregame spread) and returns a single probability that the team in possession wins. It is
post-trained with group-relative reinforcement learning (GRPO) against two design choices:

1. **A conditional-rate reward.** The reward scores each forecast against a state-conditioned
   empirical win rate `p̂(x)` estimated from past game outcomes, not against the single realized
   outcome:

   ```
   r = 1 − (p − p̂(x))²
   ```

   `p̂(x)` is built by bucketing training-season plays on score margin × time remaining × pregame
   spread and taking each bucket's win fraction, with hierarchical empirical-Bayes shrinkage
   (pseudocount 25) toward coarser buckets. It uses only realized outcomes and the public pregame
   line; the live market probability is held out for evaluation and never appears in a prompt.

2. **Gradient decoupling.** Applying the reward to a full chain of thought decalibrates the model,
   because the gradient rewrites the reasoning into pseudo-quantitative arguments for extreme
   numbers. We keep the gradient off the reasoning, either by **direct prediction** (drop the chain
   of thought) or by a **gradient mask** confined to the answer span (which requires turning off the
   KL penalty).

## Results (held-out 2024 season, n = 5185)

| Model | Brier | ECE |
|---|---|---|
| Qwen2.5-7B-Instruct, zero-shot (direct) | 0.2057 | 0.0569 |
| Qwen2.5-7B-Instruct, zero-shot (CoT) | 0.1681 | 0.0687 |
| **Masked-CoT RLVR (ours)** | 0.1522 | 0.0293 |
| **Direct RLVR (ours)** | 0.1443 | 0.0292 |
| DeepSeek-V4, zero-shot | 0.1438 | 0.0430 |
| Empirical rate `p̂` (teacher) | 0.1432 | 0.0437 |
| Betting market | 0.1355 | 0.0273 |

The direct model matches the market's calibration (ECE 0.029 vs 0.027) and reaches the same Brier as
a zero-shot frontier model and the tabular rate, the information ceiling of the public game state. The
masked model keeps a faithful chain of thought at a small cost in sharpness.

## Repository layout

```
data/        collect_pbp.py, build_qa_pairs.py, build_winrate_buckets.py, build_grpo_dataset.py
reward/      extract.py (answer parser), grpo_reward.py (conditional-rate reward), metrics.py, tests/
train/       grpo_train.py (TRL GRPOTrainer + vLLM), masked_grpo.py (answer-span mask), config.yaml
eval/        evaluate.py, backends.py, paired_bootstrap.py, teacher_ceiling.py, judge_reasoning.py
preflight/   probe_qwen.py (zero-shot format/calibration probe)
paper/       LaTeX source, figures, and build_assets.py (recomputes every number from preds)
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt        # CPU: data, reward, eval
# training only (GPU): uv pip install -r requirements-train.txt
```

## Reproduce

**Option A — from the archived data (fastest).** Download the Zenodo archive and place `grpo/`,
`qa/`, `winrate_buckets.json`, `preds/`, and `adapters/` as described in its `README`. Then:

```bash
.venv/bin/python -m pytest reward/tests/ -q          # reward + metrics tests
.venv/bin/python eval/paired_bootstrap.py            # paired bootstraps from per-play preds
.venv/bin/python paper/build_assets.py               # rebuild every figure and table from preds
```

**Option B — rebuild the data from scratch** (free, from public nflverse data):

```bash
.venv/bin/python -m data.collect_pbp --years 2015-2024 --out data/raw
.venv/bin/python -m data.build_qa_pairs --raw data/raw/pbp_2015_2024.parquet --out data/qa --stride 8 --split-seasons
.venv/bin/python -m data.build_winrate_buckets       # -> winrate_buckets.json (the p̂ teacher)
.venv/bin/python -m data.build_grpo_dataset          # -> data/grpo/{train,eval,test}.jsonl
```

Training (single NVIDIA L40S, 48 GB; bf16 + LoRA + colocated vLLM) is driven by `train/grpo_train.py`
and `train/config.yaml`; see `train/RUNPOD.md`. Splits are by season: train 2015–2022, select 2023,
test 2024.

## Data and models

The prepared dataset and the trained LoRA adapters are archived on Zenodo (DOI
`10.5281/zenodo.21082572`). The archive contains the GRPO prompts with outcomes and the `p̂` target,
the win-rate bucket table, the held-out game-state features, the per-play predictions for every
model, and the direct and masked LoRA adapters for Qwen2.5-7B-Instruct. See `DATA.md` for the schema
and provenance. Game data is derived from the public [nflverse](https://github.com/nflverse) play-by-play
release.

## License

Code is released under the MIT License (`LICENSE`). The dataset (Zenodo) is released under CC-BY-4.0;
the underlying nflverse play-by-play data is CC-BY-4.0. The LoRA adapters are derived from
Qwen2.5-7B-Instruct and inherit the Apache-2.0 license of that base model.

## Citation

See `CITATION.cff`. (Author fields are completed at de-anonymization.)
