"""Information-ceiling diagnostic (no GPU): is there extractable game-outcome signal beyond the
coarse score×time×spread buckets that the LLM's reward (`p̂`) is built from?

Compares held-out Brier of several STATIC-feature teachers against the coarse `p̂` and Vegas:
  - coarse p̂ (our reward; score×time×spread, EB-shrunk)   [data/grpo/*.jsonl `target`]
  - nflverse `wp` (a tuned full-feature WP model)          [data/qa/*.jsonl features.wp]
  - a GBM trained here on ALL standard features
  - Vegas (`vegas_wp`)                                      [the market ceiling]

RESULT (2026-06-19): every static model — including two with MORE features than us — is WORSE than
the coarse `p̂` (test: nflverse 0.1562, GBM 0.1587 vs coarse 0.1432). So there is no extractable
game-level signal beyond score×time×spread; the LLM (WS1 0.1442 ≈ coarse 0.1432) is at the
static-information ceiling, and Vegas's 0.008 edge (0.1355) is live in-game / market information not
present in the prompt. => the "beat the lookup table" (outcome-reward) probe and WS3 are both futile;
the Vegas gap is irreducible for any static-input model.

Run: python -m eval.teacher_ceiling   (needs scikit-learn for the GBM row; skips it if absent)
"""

from __future__ import annotations

import json
import numpy as np

FEATS = ["score_differential", "game_seconds_remaining", "spread_line", "down", "ydstogo",
         "yardline_100", "qtr"]


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(p, y, nb=10):
    p, y = np.asarray(p), np.asarray(y)
    b = np.minimum((p * nb).astype(int), nb - 1)
    e = 0.0
    for k in range(nb):
        m = b == k
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def load(split):
    g = [json.loads(l) for l in open(f"data/grpo/{split}.jsonl")]
    q = [json.loads(l) for l in open(f"data/qa/qa_{split}.jsonl")]
    assert len(g) == len(q)
    y = np.array([r["actual_outcome"] for r in g], float)
    assert (y == np.array([r["actual_outcome"] for r in q], float)).all(), "grpo/qa misaligned"
    return {
        "y": y,
        "coarse_phat": np.array([r["target"] for r in g], float),
        "vegas": np.array([r["vegas_wp"] for r in g], float),
        "nflverse_wp": np.array([r["features"]["wp"] for r in q], float),
        "X": np.array([[r["features"].get(f, np.nan) for f in FEATS] for r in q], float),
    }


def main():
    gbm = None
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        tr = load("train")
        gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                              l2_regularization=1.0).fit(tr["X"], tr["y"].astype(int))
    except Exception as e:
        print(f"(GBM row skipped — {e})")

    for split in ["eval", "test"]:
        d = load(split)
        rows = [("coarse p̂ (reward)", d["coarse_phat"]),
                ("nflverse wp", d["nflverse_wp"])]
        if gbm is not None:
            rows.append(("GBM (all feats)", gbm.predict_proba(d["X"])[:, 1]))
        rows.append(("vegas", d["vegas"]))
        print(f"\n=== {split} (n={len(d['y'])}) ===   Brier / ECE")
        for name, p in rows:
            print(f"  {name:<22} {brier(p, d['y']):.4f} / {ece(p, d['y']):.4f}")


if __name__ == "__main__":
    main()
