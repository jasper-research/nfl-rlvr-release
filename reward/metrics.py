"""Calibration metrics for probabilistic NFL win-probability predictions.

Self-contained (numpy only). These are the primitives the RLVR reward function
is built on, and the same functions are reused at evaluation time so that the
training signal and the reported metrics are computed identically.

Conventions
-----------
- ``probs``    : predicted P(possession team wins), each in [0, 1]
- ``outcomes`` : realized result, 1 if the predicted/possession team won else 0

Lower is better for Brier / ECE / log-loss; higher is better for accuracy.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _as_arrays(probs, outcomes):
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    if p.shape != y.shape:
        raise ValueError(f"probs/outcomes shape mismatch: {p.shape} vs {y.shape}")
    if p.size == 0:
        raise ValueError("empty input")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("outcomes must be binary 0/1")
    return p, y


def brier_score(probs, outcomes) -> float:
    """Mean squared error of probabilistic predictions. In [0, 1], lower better."""
    p, y = _as_arrays(probs, outcomes)
    return float(np.mean((p - y) ** 2))


def log_loss(probs, outcomes) -> float:
    """Binary cross-entropy. Lower better. Clipped to avoid log(0)."""
    p, y = _as_arrays(probs, outcomes)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def binary_accuracy(probs, outcomes, threshold: float = 0.5) -> float:
    """Fraction of correct hard predictions (p > threshold => predict win)."""
    p, y = _as_arrays(probs, outcomes)
    return float(np.mean((p > threshold) == (y == 1)))


def reliability_curve(probs, outcomes, n_bins: int = 10):
    """Per-bin confidence, accuracy, and count for a reliability diagram.

    Bins are equal-width over [0, 1]. The final bin is closed on the right so
    that a prediction of exactly 1.0 lands in the last bin rather than dropping.

    Returns
    -------
    bin_confidence : np.ndarray, shape (n_bins,)  -- mean predicted prob in bin
    bin_accuracy   : np.ndarray, shape (n_bins,)  -- empirical win rate in bin
    bin_counts     : np.ndarray, shape (n_bins,)  -- number of samples in bin
    Empty bins report NaN confidence/accuracy and a count of 0.
    """
    p, y = _as_arrays(probs, outcomes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=False puts p in bin i where edges[i] <= p < edges[i+1].
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        cnt[b] = c
        if c:
            conf[b] = p[mask].mean()
            acc[b] = y[mask].mean()
    return conf, acc, cnt


def expected_calibration_error(probs, outcomes, n_bins: int = 10) -> float:
    """Expected Calibration Error: count-weighted |confidence - accuracy|.

    In [0, 1], lower is better (0 == perfectly calibrated).
    """
    conf, acc, cnt = reliability_curve(probs, outcomes, n_bins=n_bins)
    total = cnt.sum()
    if total == 0:
        return 0.0
    valid = cnt > 0
    gaps = np.abs(conf[valid] - acc[valid])
    weights = cnt[valid] / total
    return float(np.sum(weights * gaps))


def maximum_calibration_error(probs, outcomes, n_bins: int = 10) -> float:
    """Worst-case per-bin calibration gap. In [0, 1], lower is better."""
    conf, acc, cnt = reliability_curve(probs, outcomes, n_bins=n_bins)
    valid = cnt > 0
    if not np.any(valid):
        return 0.0
    return float(np.max(np.abs(conf[valid] - acc[valid])))


def murphy_decomposition(probs, outcomes, n_bins: int = 10) -> dict:
    """Murphy decomposition of the (binned) Brier score:

        Brier = Reliability − Resolution + Uncertainty

    - reliability: calibration error (squared, count-weighted bin gaps) — lower is better.
    - resolution:  sharpness/discrimination (bins move away from base rate) — higher is better.
    - uncertainty: irreducible base-rate variance ō(1−ō) — fixed by the data.

    `brier_check` should match the binned mean Brier (sanity tie-out). Eval/analysis only.
    """
    p, y = _as_arrays(probs, outcomes)
    base = float(y.mean())
    uncertainty = base * (1.0 - base)

    conf, acc, cnt = reliability_curve(p, y, n_bins=n_bins)
    total = cnt.sum()
    reliability = resolution = 0.0
    if total:
        valid = cnt > 0
        w = cnt[valid] / total
        reliability = float(np.sum(w * (conf[valid] - acc[valid]) ** 2))
        resolution = float(np.sum(w * (acc[valid] - base) ** 2))

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_check": reliability - resolution + uncertainty,
    }
