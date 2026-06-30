"""Build state-conditioned empirical win-rate targets `p̂(state)` from TRAIN outcomes, and attach
them to every GRPO row. Writes TWO targets per row:

  * `target`       — coarse: (score_diff × time × spread)                 [the v8–WS1 target]
  * `target_fine`  — fine:   (score_diff × time × spread × field × down)  [WS1 granularity study]

Why a soft target at all (v8): per-sample Brier on a single 0/1 outcome gives *contradictory*
targets for near-identical states → decalibration. The empirical win-frequency of the state bucket
is a smooth, calibrated, label-free target (built only from realized outcomes; no Vegas).

Why finer buckets (WS1): the coarse target ignores field position, so a "down 6 in the red zone"
state and "down 6 at own 20" share a target — exactly the case the model misread. Adding field
position (and down) gives a per-situation target that can teach the model to use that signal, and
raises the teacher's ceiling toward Vegas. Sparse fine buckets shrink hierarchically toward their
coarser parents (Empirical-Bayes, pseudocount M). Eval/test targets use the TRAIN table only.

Run:  python -m data.build_winrate_buckets
Prints coarse-vs-fine teacher Brier/ECE on eval so we see if finer helps BEFORE training on it.
"""

from __future__ import annotations

import bisect
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward.metrics import brier_score, expected_calibration_error   # noqa: E402

GRPO = Path("data/grpo")
SPLITS = ["train", "eval", "test"]
M = 25.0   # Empirical-Bayes shrinkage pseudocount

# --- bins (posteam perspective); bisect_right(edges, x) -> int bin ---
SCORE_EDGES = [-21, -14, -10, -7, -4, -1, 0, 1, 4, 7, 10, 14, 21]
TIME_EDGES = [120, 300, 600, 900, 1800, 2700]          # secs remaining (small = late)
SPREAD_EDGES = [-10, -7, -3, -0.5, 0.5, 3, 7, 10]      # posteam spread (neg = underdog)
FIELD_EDGES = [10, 20, 40, 60, 80]                     # yards to opponent goal (small = scoring range)

_SPREAD_RE = re.compile(r"Pregame line:\s+\w+\s+(favored|underdogs)\s+by\s+([\d.]+)")
_DOWN_RE = re.compile(r"(\d)(?:st|nd|rd|th) and \d+")
_OWN_RE = re.compile(r"ball at their own (\d+)")
_OPP_RE = re.compile(r"ball at the \w+ (\d+) yard line \(opponent territory\)")


def parse_spread(p: str) -> float:
    m = _SPREAD_RE.search(p)
    if not m:
        return 0.0
    v = float(m.group(2))
    return v if m.group(1) == "favored" else -v


def parse_yardline_100(p: str) -> int:
    """Yards to the opponent's goal (0–100, smaller = closer to scoring)."""
    m = _OWN_RE.search(p)
    if m:
        return 100 - int(m.group(1))          # own X => 100-X to opp goal
    m = _OPP_RE.search(p)
    if m:
        return int(m.group(1))                # opp X yard line => X to goal
    if "midfield" in p:
        return 50
    return 50                                  # default to midfield if unparseable


def parse_down(p: str) -> int:
    m = _DOWN_RE.search(p)
    return int(m.group(1)) if m else 0


def sbin(r): return bisect.bisect_right(SCORE_EDGES, int(r["meta"]["score_differential"]))
def tbin(r): return bisect.bisect_right(TIME_EDGES, int(r["meta"]["game_seconds_remaining"]))
def pbin(r): return bisect.bisect_right(SPREAD_EDGES, parse_spread(r["prompt"]))
def fbin(r): return bisect.bisect_right(FIELD_EDGES, parse_yardline_100(r["prompt"]))
def dbin(r): return parse_down(r["prompt"])

COARSE = [sbin, tbin, pbin]
FINE = [sbin, tbin, pbin, fbin, dbin]   # importance order; backoff drops the last feature first


def fit(rows, feats):
    """Hierarchical EB shrinkage. Returns target_for(row): the finest available prefix estimate,
    each level shrunk toward its coarser parent (and the coarsest toward the global rate)."""
    K = len(feats)
    def fullkey(r): return tuple(f(r) for f in feats)
    counts = [defaultdict(lambda: [0, 0]) for _ in range(K + 1)]
    for r in rows:
        y = int(r["actual_outcome"]); fk = fullkey(r)
        for L in range(K + 1):
            c = counts[L][fk[:L]]; c[0] += y; c[1] += 1
    g = counts[0][()][0] / counts[0][()][1]
    hat = [dict() for _ in range(K + 1)]
    hat[0][()] = g
    for L in range(1, K + 1):
        for k, (w, n) in counts[L].items():
            hat[L][k] = (w + M * hat[L - 1].get(k[:-1], g)) / (n + M)

    def target_for(r):
        fk = fullkey(r)
        for L in range(K, -1, -1):
            if fk[:L] in hat[L]:
                return round(float(hat[L][fk[:L]]), 5)
        return round(float(g), 5)
    return target_for, g


def main():
    train = [json.loads(l) for l in (GRPO / "train.jsonl").open()]
    coarse, g = fit(train, COARSE)
    fine, _ = fit(train, FINE)

    for split in SPLITS:
        path = GRPO / f"{split}.jsonl"
        rows = [json.loads(l) for l in path.open()]
        for r in rows:
            r["target"] = coarse(r)
            r["target_fine"] = fine(r)
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(rows)} rows tagged (target + target_fine)")

    # --- ceiling comparison on eval: does fine beat coarse? (decide before GPU) ---
    ev = [json.loads(l) for l in (GRPO / "eval.jsonl").open()]
    y = [int(r["actual_outcome"]) for r in ev]
    tc = [r["target"] for r in ev]
    tf = [r["target_fine"] for r in ev]
    vg = [r["vegas_wp"] for r in ev]
    print(f"\nglobal posteam win rate = {g:.3f}")
    print("teacher ceiling on eval 2023 (n=%d):" % len(ev))
    print(f"  coarse target      Brier {brier_score(tc, y):.4f}  ECE {expected_calibration_error(tc, y):.4f}")
    print(f"  fine   target      Brier {brier_score(tf, y):.4f}  ECE {expected_calibration_error(tf, y):.4f}")
    print(f"  vegas (reference)  Brier {brier_score(vg, y):.4f}  ECE {expected_calibration_error(vg, y):.4f}")


if __name__ == "__main__":
    main()
