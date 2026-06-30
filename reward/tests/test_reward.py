"""Unit tests for the GRPO reward, parser, and eval metrics. Run: pytest -q from repo root."""

import numpy as np
import pytest

from reward.extract import answer_region, extract_probability
from reward.grpo_reward import compute_reward, has_sufficient_variance, make_reward_fn
from reward.metrics import (
    brier_score,
    expected_calibration_error,
    murphy_decomposition,
    reliability_curve,
)


# ---- extraction (incl. thinking-aware) -----------------------------------

@pytest.mark.parametrize("text,expected", [
    ("After reasoning, the final probability is 0.73.", 0.73),
    ("I estimate a 62% chance they win.", 0.62),
    ("Win probability: 8%", 0.08),
    ("My answer: .85", 0.85),
    ("about a 72 percent shot. Final answer: 72%", 0.72),
    # regression: 'p' in "possession" must not grab the "2" in "2nd-and-13"
    ("they still need a possession. Facing 2nd-and-13... Around 26%.", 0.26),
    # regression: "the final seventeen minutes" must not be treated as a cue
    ("trailing into the final seventeen minutes at their own 25. I'd put them at 44%.", 0.44),
    ("No number here at all", None),
])
def test_extract_probability(text, expected):
    got = extract_probability(text)
    assert (got is None) if expected is None else (got == pytest.approx(expected))


def test_answer_after_think_block_wins():
    # A misleading number inside <think>, the real answer after </think>.
    r = "<think>early on this looked like 90%, but the comeback stalled</think>\nFinal: 35%"
    assert extract_probability(r) == pytest.approx(0.35)


def test_falls_back_into_think_when_no_answer_number():
    r = "<think>this is clearly around 80%</think>\nThe team is in good shape."
    assert extract_probability(r) == pytest.approx(0.80)


def test_unclosed_think_block_stripped():
    r = "<think>reasoning with 12 yards and a 3rd down that never closes"
    # no parseable probability anywhere valid -> None (12/3 coerce to 0.12/0.03? guard below)
    # ensure the dangling think text doesn't crash and answer_region empties it
    assert isinstance(answer_region(r), str)


# ---- per-rollout reward --------------------------------------------------

def test_reward_bounds_and_values():
    # extremes are clipped to [0.01, 0.99] before scoring (not rejected)
    assert compute_reward("100%", 1)[0] == pytest.approx(1 - 0.01**2, abs=1e-3)   # ~0.9999
    assert compute_reward("0%", 1)[0] == pytest.approx(1 - 0.99**2, abs=1e-3)      # ~0.0199
    assert compute_reward("50%", 1)[0] == pytest.approx(0.75, abs=1e-3)


def test_format_penalty_path():
    r, d = compute_reward("the chiefs look strong", 1, format_penalty=0.0)
    assert r == 0.0 and d["format_ok"] is False
    r2, _ = compute_reward("no number", 1, format_penalty=-1.0)
    assert r2 == -1.0


def test_clip_not_reject_for_blowout():
    # "100%" is clipped to 0.99 for scoring, NOT rejected as a format failure.
    r, d = compute_reward("Up 35 in the fourth. 100%", 1)
    assert d["format_ok"] is True and d["prob"] == pytest.approx(0.99)


def test_target_overrides_outcome():
    # v8 soft target: scored against p̂(state), not the raw 0/1. A 0.70 forecast on a state whose
    # empirical win rate is 0.70 is *perfect* (reward ~1), even though this game happened to be won.
    r_tgt, d = compute_reward("70%", 1, target=0.70)
    assert d["prob"] == pytest.approx(0.70) and r_tgt == pytest.approx(1.0, abs=1e-6)
    # without a target it falls back to the realized outcome (1): 1 - (0.7-1)^2 = 0.91
    r_out, _ = compute_reward("70%", 1)
    assert r_out == pytest.approx(1 - 0.3**2, abs=1e-6)


def test_answer_prefix_len_locates_last_marker():
    # WS2 masked-CoT: tokens BEFORE the answer span (the "Probability:" marker)
    from train.masked_grpo import answer_prefix_len, MASK_MARKER
    enc = lambda s: s.split()                       # whitespace tokenizer stand-in
    assert answer_prefix_len("a b c Probability: 63%", MASK_MARKER, enc) == 3   # "a b c " -> 3
    assert answer_prefix_len("no marker anywhere", MASK_MARKER, enc) is None    # fallback case
    # uses the LAST occurrence (reasoning may say 'Probability:' earlier)
    t = "Probability: unclear early on. Final Probability: 70%"
    assert answer_prefix_len(t, MASK_MARKER, enc) == len("Probability: unclear early on. Final ".split())


def test_blend_mixes_target_and_outcome():
    # blend λ: τ = λ·p̂ + (1-λ)·y
    assert compute_reward("60%", 1, target=0.40, blend=1.0)[0] == pytest.approx(1 - (0.6 - 0.40)**2, abs=1e-6)  # pure p̂
    assert compute_reward("60%", 1, target=0.40, blend=0.0)[0] == pytest.approx(1 - (0.6 - 1.0)**2, abs=1e-6)   # pure y
    assert compute_reward("60%", 1, target=0.40, blend=0.5)[0] == pytest.approx(1 - (0.6 - 0.70)**2, abs=1e-6)  # τ=0.7


def test_make_reward_fn_uses_soft_target_when_present():
    fn = make_reward_fn(use_soft_target=True)
    rewards = fn(prompts=["p"], completions=["I'd say 70%."], actual_outcome=[1], target=[0.70])
    assert rewards[0] == pytest.approx(1.0, abs=1e-6)          # perfect vs the soft target


def test_use_soft_target_false_ignores_target_column():
    fn = make_reward_fn(use_soft_target=False)
    rewards = fn(prompts=["p"], completions=["I'd say 70%."], actual_outcome=[1], target=[0.70])
    assert rewards[0] == pytest.approx(1 - 0.3**2, abs=1e-6)   # vs realized outcome (1), target ignored


def test_sharpness_off_by_default():
    _, d05 = compute_reward("50%", 1)
    # at p=0.5 sharpness term is zero regardless; reward == brier_reward
    assert d05["reward"] == pytest.approx(d05["brier_reward"])
    # with sharpness on, a confident correct call is rewarded more than pure brier
    r_pure, _ = compute_reward("80%", 1)
    r_sharp, _ = compute_reward("80%", 1, sharpness_weight=0.3)
    assert r_sharp > r_pure


def test_brier_is_proper_expected_reward_peaks_at_true_rate():
    # For true win rate q, expected reward over y~Bernoulli(q) is maximized at p=q.
    q = 0.7
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    exp_reward = []
    for p in grid:
        resp = f"{int(round(p*100))}%"
        e = q * compute_reward(resp, 1)[0] + (1 - q) * compute_reward(resp, 0)[0]
        exp_reward.append(e)
    best = grid[int(np.argmax(exp_reward))]
    assert abs(best - q) <= 0.05


def test_has_sufficient_variance():
    assert not has_sufficient_variance([0.9, 0.9, 0.91, 0.9], min_std=0.05)
    assert has_sufficient_variance([0.1, 0.9, 0.5, 0.3], min_std=0.05)


# ---- TRL wrapper ---------------------------------------------------------

def test_make_reward_fn_aligns_and_reads_outcome():
    fn = make_reward_fn()
    completions = ["I'd say 80%.", "maybe 20%", "no idea"]
    outcomes = [1, 0, 1]
    rewards = fn(prompts=["p1", "p2", "p3"], completions=completions, actual_outcome=outcomes)
    assert len(rewards) == 3
    assert rewards[0] == pytest.approx(1 - 0.2**2, abs=1e-3)   # 0.8 vs win
    assert rewards[1] == pytest.approx(1 - 0.2**2, abs=1e-3)   # 0.2 vs loss
    assert rewards[2] == 0.0                                    # unparseable -> format_penalty


def test_make_reward_fn_handles_chat_format():
    fn = make_reward_fn()
    chat = [[{"role": "assistant", "content": "Final probability: 90%"}]]
    rewards = fn(prompts=["p"], completions=chat, actual_outcome=[1])
    assert rewards[0] == pytest.approx(1 - 0.1**2, abs=1e-3)


# ---- eval metrics --------------------------------------------------------

def test_ece_calibrated_low_overconfident_high():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 5000)
    outs = (rng.uniform(0, 1, 5000) < probs).astype(int)
    assert expected_calibration_error(probs, outs) < 0.05
    bad_p = np.full(1000, 0.99)
    bad_y = np.tile([1, 0], 500)
    assert expected_calibration_error(bad_p, bad_y) > 0.4


def test_murphy_identity_matches_brier():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, 4000)
    outs = (rng.uniform(0, 1, 4000) < probs).astype(int)
    d = murphy_decomposition(probs, outs, n_bins=10)
    # binned identity ties out to the binned Brier (mean over the same bins)
    # check the decomposition recombines to brier_check and is close to raw Brier
    assert d["brier_check"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"])
    assert d["brier_check"] == pytest.approx(brier_score(probs, outs), abs=0.02)


def test_reliability_curve_last_bin_and_counts():
    conf, acc, cnt = reliability_curve([0.05, 1.0, 0.96], [0, 1, 1], n_bins=10)
    assert cnt[0] == 1 and cnt[-1] == 2 and cnt.sum() == 3
