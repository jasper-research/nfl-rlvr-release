"""Paired bootstrap on per-game Brier (and accuracy) — the powerful test for the WS1-vs-WS2 and
vs-Vegas gaps. Unlike comparing two independent CIs, this resamples the SAME games and looks at the
per-game difference, so it detects gaps the wide marginal CIs miss.

Needs per-game predictions, produced by `eval_checkpoints --save-preds` (writes
eval/results/preds_<split>_<tag>.json = {variant: [{game_id, prob, outcome, vegas, parsed}, ...]}).

Workflow (after the ladders pick the lead checkpoints):
  # 1) per-game preds for the two leads on TEST (cheap — single checkpoint each via --adapters)
  python -m eval.eval_checkpoints --output-dir checkpoints/grpo-ws1-lr2e5-ga16 --direct --max-tokens 48 \
      --split test --n 0 --save-preds --tag ws1-pb --adapters checkpoints/grpo-ws1-lr2e5-ga16/checkpoint-200
  python -m eval.eval_checkpoints --output-dir checkpoints/grpo-ws2-lr3e5 --masked --max-tokens 1024 \
      --split test --n 0 --save-preds --tag ws2-pb --adapters checkpoints/grpo-ws2-lr3e5/checkpoint-XXX
  # 2) the test (fill the exact variant labels printed by the runs)
  python -m eval.paired_bootstrap \
      --a eval/results/preds_test_ws1-pb.json:checkpoint-200:WS1 \
      --b eval/results/preds_test_ws2-lr3e5.json:checkpoint-XXX:WS2

Each --a/--b/--c is "predsfile:variant:label". Vegas is read from the rows (same games) automatically.
Reports mean per-game Brier (and accuracy) difference + 95% bootstrap CI for every pair incl. Vegas;
a CI that excludes 0 = a statistically significant gap.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

REPS = 10000
SEED = 0


def load_spec(spec):
    """'predsfile:variant:label' -> (label, variant, list[ row|None ]) IN ROW ORDER.
    Each row = (prob, outcome, vegas) if parsed else None. We pair the two files by POSITION (both
    are generated over the same load_split rows in the same order — game_id is per-GAME, not per-play,
    so it is NOT a valid join key: ~18 plays share a game_id)."""
    parts = spec.split(":")
    if len(parts) < 3:
        raise SystemExit(f"--a/--b must be 'predsfile:variant:label'; got {spec!r}")
    label = parts[-1]
    variant = parts[-2]
    path = ":".join(parts[:-2])
    data = json.load(open(path))
    if variant not in data:
        raise SystemExit(f"variant {variant!r} not in {path} (have: {list(data)})")
    out = []
    for d in data[variant]:
        if d.get("parsed") and d.get("prob") is not None:
            out.append((float(d["prob"]), int(d["outcome"]), float(d["vegas"])))
        else:
            out.append(None)
    return label, variant, out


def boot_diff(per_game_diff, reps=REPS, seed=SEED):
    a = np.asarray(per_game_diff, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    means = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(name, diff):
    """diff = per-game (X - Y); negative => X better (lower Brier). CI excluding 0 = significant."""
    mean, lo, hi = boot_diff(diff)
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not sig (CI spans 0)"
    print(f"  {name:<22} Δ={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   {sig}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="predsfile:variant:label (model A)")
    ap.add_argument("--b", required=True, help="predsfile:variant:label (model B)")
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()

    la, va, ra = load_spec(args.a)
    lb, vb, rb = load_spec(args.b)
    if len(ra) != len(rb):
        raise SystemExit(f"A and B have different row counts ({len(ra)} vs {len(rb)}) — not the "
                         "same split/order; cannot pair by position.")
    # pair by POSITION; keep rows where BOTH parsed
    idx = [i for i in range(len(ra)) if ra[i] is not None and rb[i] is not None]
    if not idx:
        raise SystemExit("no rows where both A and B parsed")
    print(f"[paired_bootstrap] {la}({va}) vs {lb}({vb}) vs Vegas | n={len(idx)}/{len(ra)} paired "
          f"(both parsed) | reps={args.reps}")

    y = np.array([ra[i][1] for i in idx], float)
    pa = np.array([ra[i][0] for i in idx], float)
    pb = np.array([rb[i][0] for i in idx], float)
    pv = np.array([ra[i][2] for i in idx], float)   # vegas (same rows)

    bA, bB, bV = (pa - y) ** 2, (pb - y) ** 2, (pv - y) ** 2
    accA = ((pa > 0.5).astype(int) == y).astype(float)
    accB = ((pb > 0.5).astype(int) == y).astype(float)
    accV = ((pv > 0.5).astype(int) == y).astype(float)

    print(f"\n  Brier: {la} {bA.mean():.4f} | {lb} {bB.mean():.4f} | Vegas {bV.mean():.4f}")
    print("\nPaired Brier differences (negative => first model better):")
    report(f"{la} - {lb}", bA - bB)
    report(f"{la} - Vegas", bA - bV)
    report(f"{lb} - Vegas", bB - bV)
    print("\nPaired accuracy differences (positive => first model better):")
    report(f"{la} - {lb}", accA - accB)
    report(f"{la} - Vegas", accA - accV)
    report(f"{lb} - Vegas", accB - accV)


if __name__ == "__main__":
    main()
