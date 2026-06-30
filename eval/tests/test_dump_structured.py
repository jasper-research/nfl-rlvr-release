"""Tests for the reasoning-quantification ground-truth derivation + judge JSON parsing.
These guard the facts the whole CoT audit is scored against, so they must be exact. Run: pytest -q."""

import pytest

from eval.dump_structured import (
    build_ground_truth,
    field_position,
    parse_model_spec,
    score_state,
    spread_from_question,
    stratum,
)
from eval.judge_reasoning import _parse_json, cohens_kappa


# ---- field position (yardline_100 = yards to opponent end zone) -----------
@pytest.mark.parametrize("y,expected", [
    (64, "own 36"),         # own 36 -> 100-64
    (75, "own 25"),
    (12, "opponent 12 (red zone)"),
    (20, "opponent 20 (red zone)"),
    (43, "opponent 43"),
    (50, "midfield (50)"),
])
def test_field_position(y, expected):
    assert field_position(y) == expected


def test_field_position_none():
    assert field_position(None) == "unknown"


# ---- score state (score_differential is posteam - defteam) ----------------
def test_score_state():
    assert score_state("WAS", 0) == "tied"
    assert score_state("WAS", 4) == "WAS leading by 4"
    assert score_state("ARI", -7) == "ARI trailing by 7"
    assert score_state("ARI", -0.0) == "tied"


# ---- spread parsed from the templated question ----------------------------
def test_spread_from_question():
    q_fav = "NFL game: WAS (possession) vs ARI.\n- Pregame line: WAS favored by 7.0\nQuestion: ..."
    assert spread_from_question(q_fav, "WAS") == {"side": "favored", "team": "WAS", "points": 7.0}
    q_dog = "...\n- Pregame line: ARI underdogs by 7.0\n..."
    assert spread_from_question(q_dog, "ARI") == {"side": "underdog", "team": "ARI", "points": 7.0}
    q_pick = "...\n- Pregame line: pick'em\n..."
    assert spread_from_question(q_pick, "WAS")["side"] == "pickem"
    assert spread_from_question("no line here", "WAS")["side"] == "unknown"


# ---- full ground-truth row ------------------------------------------------
def _row():
    return {
        "game_id": "2023_01_ARI_WAS", "season": 2023, "week": 1, "actual_outcome": 1,
        "target_prob": 0.76,
        "question": ("NFL game: WAS (possession) vs ARI.\n- 1st quarter, 13:16 left in the quarter\n"
                     "- WAS tied\n- 1st and 10, ball at their own 36 yard line\n"
                     "- Pregame line: WAS favored by 7.0\nQuestion: ..."),
        "features": {"posteam": "WAS", "defteam": "ARI", "qtr": 1, "time": "13:16",
                     "game_seconds_remaining": 3496, "down": 1, "ydstogo": 10, "yardline_100": 64,
                     "score_differential": 0, "vegas_wp": 0.74, "spread_line": 7.0},
    }


def test_build_ground_truth():
    gt = build_ground_truth(_row())
    assert gt["posteam"] == "WAS" and gt["defteam"] == "ARI"
    assert gt["period"] == "Q1" and gt["clock"] == "13:16"
    assert gt["score"] == "tied" and gt["score_differential"] == 0
    assert gt["down"] == 1 and gt["ydstogo"] == 10
    assert gt["field_position"] == "own 36"
    assert gt["spread"] == {"side": "favored", "team": "WAS", "points": 7.0}
    assert gt["actual_outcome"] == 1


def test_overtime_period():
    r = _row()
    r["features"]["qtr"] = 5
    assert build_ground_truth(r)["period"] == "overtime"


def test_stratum_buckets():
    gt = build_ground_truth(_row())
    assert stratum(gt) == ("early", "close", "favored")
    r = _row()
    r["features"]["qtr"] = 3
    r["features"]["score_differential"] = -14
    assert stratum(build_ground_truth(r)) == ("late", "lopsided", "favored")


# ---- model spec parsing ---------------------------------------------------
def test_parse_model_spec():
    assert parse_model_spec("base::cot") == {"label": "base", "adapter": None, "mode": "cot"}
    s = parse_model_spec("ws2:checkpoints/grpo-ws2/checkpoint-150:masked")
    assert s == {"label": "ws2", "adapter": "checkpoints/grpo-ws2/checkpoint-150", "mode": "masked"}
    with pytest.raises(SystemExit):
        parse_model_spec("bad:mode:xyz")          # xyz not a valid mode
    with pytest.raises(SystemExit):
        parse_model_spec("noparts")


# ---- judge JSON parsing (robust to fences / surrounding prose) ------------
def test_parse_json_plain_and_fenced():
    assert _parse_json('{"a": 1}') == {"a": 1}
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json('Here is the verdict:\n{"a": true, "b": "x"}\nDone.') == {"a": True, "b": "x"}
    assert _parse_json("no json at all") is None


def test_cohens_kappa():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    # chance-level agreement -> kappa ~ 0
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0, abs=1e-9)


# ---- paired bootstrap ----------------------------------------------------
def test_boot_diff_sign_and_significance():
    import numpy as np
    from eval.paired_bootstrap import boot_diff
    # a clearly-positive per-game difference -> CI strictly above 0 (significant)
    mean, lo, hi = boot_diff(np.full(300, 0.02) + np.random.default_rng(0).normal(0, 0.005, 300))
    assert mean == pytest.approx(0.02, abs=0.003) and lo > 0
    # a zero-mean symmetric difference -> CI straddles 0 (not significant)
    sym = np.random.default_rng(1).normal(0, 0.02, 400)
    m, lo2, hi2 = boot_diff(sym)
    assert lo2 < 0 < hi2
