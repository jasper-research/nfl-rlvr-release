"""Parse in-training [eval @ N] calibration lines from GRPO train logs into figdata for F2.

The CalibrationEval callback prints one dict per checkpoint:
    [eval @ 150]  {'eval/brier': ..., 'eval/ece': ..., 'eval/reliability': ..., ...}
We scan each named log, pull those dicts, and emit a per-run step-indexed trajectory.

Decalibrating CoT runs (v6/v9) were not captured to a local log; their points come from the
ledger (or W&B) and are added separately. Run from repo root: python paper/parse_intraining.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BK = ROOT.parent / "nfl-rlvr-backup" / "staging_logs" / "ws2_logs"
OUT = Path(__file__).resolve().parent / "figdata"
OUT.mkdir(exist_ok=True)

# label -> log file (calibrating runs recoverable from local logs)
LOGS = {
    "direct_lead":  BK / "train-ws1-lr2e5-ga16.log",   # WS1 direct, lr2e-5 + ga16 (calibration lead)
    "masked_lead":  BK / "train-ws2-lr3e5.log",        # WS2 masked-CoT, lr3e-5 (masked lead)
    "direct_lr1e5": BK / "train-ws1-lr1e5.log",
    "masked_lr4e5": BK / "train-ws2-lr4e5.log",
}

EVAL_RE = re.compile(r"\[eval @ (\d+)\]\s+(\{.*?\})")


def parse_log(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    traj = {}
    for m in EVAL_RE.finditer(text):
        step = int(m.group(1))
        try:
            d = ast.literal_eval(m.group(2))
        except Exception:
            continue
        traj[step] = {k.replace("eval/", ""): v for k, v in d.items()}
    return dict(sorted(traj.items())) or None


def main():
    out = {"_note": "per-step in-training calibration eval (fixed n=128, greedy), parsed from "
                     "GRPO train-log [eval @ N] lines. vegas_brier=0.1451 on the same 128.",
           "runs": {}}
    for label, path in LOGS.items():
        traj = parse_log(path)
        if traj:
            out["runs"][label] = traj
            steps = list(traj)
            print(f"{label:14s} steps {steps}  "
                  f"ece {[round(traj[s]['ece'],4) for s in steps]}  "
                  f"brier {[round(traj[s]['brier'],4) for s in steps]}")
        else:
            print(f"{label:14s} -- LOG NOT FOUND ({path.name})")

    # Decalibrating CoT runs + v10 direct: sourced from the ledger (not in local logs / need W&B).
    # ECE rises for CoT-RLVR; falls for direct. Marked source='ledger' (sparse).
    out["runs_from_ledger"] = {
        "_source": "notes/experiments.md (v6-v9 not captured to local logs; W&B has full history)",
        "v9_cot_phat":   {"50": {"ece": 0.27}, "100": {"ece": 0.31, "resolution": 0.02}},
        "v6_cot_outcome": {"50": {"ece": 0.19}, "150": {"ece": 0.30}},
        "v10_direct_phat": {"50": {"brier": 0.1847, "ece": 0.0607},
                            "100": {"brier": 0.1635, "ece": 0.0516},
                            "150": {"brier": 0.1719, "ece": 0.0872},
                            "200": {"brier": 0.1647, "ece": 0.0904},
                            "250": {"brier": 0.1597, "ece": 0.0848}},
    }
    json.dump(out, open(OUT / "intraining_eval.json", "w"), indent=2)
    print(f"\nwrote {OUT/'intraining_eval.json'}")


if __name__ == "__main__":
    main()
