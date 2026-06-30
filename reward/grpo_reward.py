"""The GRPO reward — the core contribution of the paper.

Per-rollout reward = **Brier against the conditional-rate target**:  r = 1 − (p − p̂(x))², where
p̂(x) is the state-conditioned empirical win rate (the `target` column; see
data/build_winrate_buckets.py). Scoring against the single realized outcome y ∈ {0,1} is the special
case `target=None` / `blend=0`, and it injects the per-play coin-flip variance that decalibrates
training; the conditional rate removes that variance while staying verifiable (it is built from
realized outcomes alone). The final configuration uses the pure rate target (`blend=1.0`).

Why this yields calibration (proper scoring rule): Brier's expectation over functions of the game
state is minimized uniquely by η(x) = P(win | x), the calibrated forecaster, and reliability is an
additive term of the Murphy decomposition (Brier = Reliability − Resolution + Uncertainty). No
overconfidence penalty (it double-counts Brier and breaks properness). ECE at eval (see metrics.py)
is the falsification test that training found η(x) rather than memorizing outcomes.

The target and the label are built from `actual_outcome` (the game was played — verifiable), NEVER
from `vegas_wp`, which is held out for evaluation and never appears in a prompt.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .extract import extract_probability
from .metrics import expected_calibration_error


def compute_reward(
    response: str,
    actual_outcome: int,
    *,
    target: Optional[float] = None,  # v8: soft target p̂(state); falls back to actual_outcome
    blend: float = 1.0,              # WS1: τ = blend·p̂ + (1-blend)·y. 1.0=pure p̂, 0.0=pure outcome
    sharpness_weight: float = 0.0,   # ablation only; OFF by default
    format_penalty: float = 0.0,     # reward when no probability can be parsed
    clip: tuple = (0.01, 0.99),
) -> tuple[float, dict]:
    """Scalar reward for one rollout, plus a diagnostics dict (not used by GRPO).

    - parse fails  -> format_penalty (default 0.0).
    - else         -> 1 − (clip(p) − τ)², optionally + a sharpness bonus,
      where τ = `target` (state-conditioned empirical win-rate p̂, v8) if given, else the realized
      `actual_outcome` ∈ {0,1}. The soft target removes the single-outcome coin-flip noise that
      decalibrates per-sample Brier; both still have the calibrated probability as their minimizer.

    The parsed probability is *clipped* (not rejected) so a legitimately near-certain blowout is
    still scored; clipping keeps Brier well-behaved without discarding the rollout's gradient.
    """
    if actual_outcome not in (0, 1):
        raise ValueError("actual_outcome must be 0 or 1")

    p = extract_probability(response)
    if p is None:
        return format_penalty, {"prob": None, "format_ok": False, "reward": format_penalty}

    if target is not None:
        tgt = blend * float(target) + (1.0 - blend) * float(actual_outcome)
    else:
        tgt = float(actual_outcome)
    p = float(min(max(p, clip[0]), clip[1]))
    brier = (p - tgt) ** 2
    reward = 1.0 - brier
    sharpness = abs(p - 0.5)
    if sharpness_weight:
        reward = reward + reward * sharpness * sharpness_weight

    return float(reward), {
        "prob": p,
        "brier": brier,
        "brier_reward": 1.0 - brier,
        "sharpness": sharpness,
        "format_ok": True,
        "reward": float(reward),
    }


def has_sufficient_variance(rewards, min_std: float = 0.05) -> bool:
    """DAPO dynamic-sampling filter: a group with near-zero reward spread produces a near-zero
    advantage and wastes the update. Keep only groups with genuine variance."""
    r = np.asarray(rewards, dtype=np.float64)
    return bool(r.size > 1 and np.std(r) >= min_std)


def make_reward_fn(
    *,
    use_soft_target: bool = True,    # v8: score against p̂(state) `target` column when present
    blend: float = 1.0,              # WS1: τ = blend·p̂ + (1-blend)·y (only when use_soft_target)
    sharpness_weight: float = 0.0,
    format_penalty: float = 0.0,
    clip: tuple = (0.01, 0.99),
):
    """Build a TRL-GRPOTrainer-compatible reward function with step-aligned logging.

    TRL calls `reward_fn(prompts, completions, **columns)` with dataset columns as aligned lists.
    We read `actual_outcome`, return one scalar per completion, and buffer per-completion
    diagnostics. The returned function carries `.pop_metrics()`: a TrainerCallback calls it in
    on_log and logs the result at the trainer's global_step, so our metrics share the same x-axis
    as TRL's native reward/entropy/length curves (no direct wandb.log here)."""
    buffer = []

    def reward_fn(prompts=None, completions=None, actual_outcome=None, target=None, **kwargs):
        if completions is None:
            raise ValueError("reward_fn requires completions")
        if actual_outcome is None:
            raise ValueError("dataset must provide an 'actual_outcome' column")

        n = len(completions)
        targets = target if (use_soft_target and target is not None) else [None] * n
        rewards = []
        for comp, y, t in zip(completions, actual_outcome, targets):
            text = comp if isinstance(comp, str) else _completion_text(comp)
            r, d = compute_reward(
                text, int(y),
                target=(float(t) if t is not None else None),
                blend=blend,
                sharpness_weight=sharpness_weight,
                format_penalty=format_penalty,
                clip=clip,
            )
            rewards.append(r)
            d["actual_outcome"] = int(y)
            d["chars"] = len(text)
            buffer.append(d)
        return rewards

    def pop_metrics() -> dict:
        """Aggregate diagnostics buffered since the last call, then clear the buffer."""
        diags, buffer[:] = list(buffer), []
        if not diags:
            return {}
        valid = [d for d in diags if d.get("format_ok")]
        payload = {
            "train/format_success_rate": len(valid) / max(len(diags), 1),
            # response_length (our char metric; TRL also logs completions/mean_length in tokens)
            "train/mean_completion_chars": float(np.mean([d.get("chars", 0) for d in diags])),
        }
        if valid:
            probs = np.array([d["prob"] for d in valid])
            outs = np.array([d["actual_outcome"] for d in valid])
            payload.update({
                "train/mean_brier_reward": float(np.mean([d["brier_reward"] for d in valid])),
                "train/mean_reward": float(np.mean([d["reward"] for d in valid])),
                "train/mean_predicted_prob": float(np.mean(probs)),       # DAPO mean_probability
                "train/mean_confidence": float(np.mean(np.abs(probs - 0.5))),  # key diagnostic
                "train/pred_prob_std": float(np.std(probs)),  # ->0 = collapse (entropy proxy)
            })
            if len(valid) >= 20:
                payload["train/batch_ece"] = expected_calibration_error(probs, outs)
        return payload

    reward_fn.pop_metrics = pop_metrics
    return reward_fn


def _completion_text(comp) -> str:
    """Handle both plain-string and chat-format ([{role, content}, ...]) completions."""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp and isinstance(comp[-1], dict):
        return comp[-1].get("content", "")
    return str(comp)
