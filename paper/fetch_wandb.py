"""One-time pull of in-training calibration histories for the decalibrating CoT runs (v6, v9)
from W&B, cached to figdata/wandb_eval.json so the figure build stays offline and deterministic.

Needs WANDB_API_KEY and a WANDB_ENTITY set to your own W&B account. Re-run only to refresh;
build_assets.py reads the cached JSON, not W&B.  Run: python paper/fetch_wandb.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import wandb

OUT = Path(__file__).resolve().parent / "figdata" / "wandb_eval.json"
# Set to "<your-wandb-entity>/nfl-rlvr-calibration" to refresh from your own runs.
PROJECT = os.environ.get("WANDB_ENTITY", "<entity>") + "/nfl-rlvr-calibration"
# version -> run id (confirmed by config: v6 outcome reward, v9 soft p-hat + scale_rewards; both
# full-gradient CoT, i.e. the decalibrating runs). Trailing run-name number == version for v3-v9.
RUNS = {"v6_cot_outcome": "hck70zak", "v9_cot_phat": "9ivo06mi"}
KEYS = ["eval/global_step", "eval/ece", "eval/brier", "eval/resolution", "eval/reliability",
        "eval/vegas_brier"]


def main():
    api = wandb.Api()
    out = {"_source": f"W&B {PROJECT}; full-gradient CoT runs (decalibrating). step 0 = base model.",
           "runs": {}}
    for tag, rid in RUNS.items():
        r = api.run(f"{PROJECT}/{rid}")
        traj = {}
        for d in r.history(keys=KEYS, pandas=False):
            if "eval/ece" not in d:
                continue
            gs = d.get("eval/global_step", d.get("_step"))
            traj[int(gs)] = {k.replace("eval/", ""): round(v, 4)
                             for k, v in d.items()
                             if k.startswith("eval/") and k != "eval/global_step"
                             and isinstance(v, (int, float))}
        out["runs"][tag] = {"wandb_id": rid, "name": r.name, "traj": dict(sorted(traj.items()))}
        print(f"{tag:16s} {r.name:20s} steps {list(out['runs'][tag]['traj'])}")
    OUT.parent.mkdir(exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
