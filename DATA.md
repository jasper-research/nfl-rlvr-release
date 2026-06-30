# Data and models

The prepared dataset and trained adapters are archived on Zenodo, DOI `10.5281/zenodo.XXXXXXX`. This
file summarizes the contents; the archive's own `README` carries the full schema.

## What is where

| Path (in the Zenodo archive) | Contents |
|---|---|
| `grpo/{train,eval,test}.jsonl` | GRPO prompts with `actual_outcome`, the `target` (`p̂`), and `vegas_wp` (eval-only) |
| `winrate_buckets.json` | the empirical-rate teacher `p̂(x)` (score × time × spread, EB-shrunk) |
| `qa/{eval,test}.jsonl` | held-out game states with ground-truth `features` |
| `preds/preds_test_*.json` | per-play predictions (direct, masked, DeepSeek) — reproduce all tables/figures |
| `preds/reasoning_judged_eval.jsonl` | blinded-judge labels for the faithfulness analysis |
| `adapters/{direct,masked}` | LoRA adapters for Qwen2.5-7B-Instruct |

## Reproduce vs. rebuild

- **Reproduce** the paper's numbers without a GPU: download the archive, then run
  `eval/paired_bootstrap.py` and `paper/build_assets.py` against `preds/`.
- **Rebuild** the data from scratch (free, from public nflverse data): run the `data/` scripts in the
  order given in the repository `README` (Option B).

## Provenance and license

Game data derives from the public [nflverse](https://github.com/nflverse) play-by-play release
(CC-BY-4.0). The prepared data files are CC-BY-4.0. The LoRA adapters derive from Qwen2.5-7B-Instruct
and inherit its Apache-2.0 license. No personal data is included.

## Key integrity note

The live market win probability (`vegas_wp`) is used **only at evaluation** and never appears in any
training prompt; the `target` is built from realized outcomes and the public pregame line alone. The
audit is in the paper's appendix and is checked by the data pipeline.
