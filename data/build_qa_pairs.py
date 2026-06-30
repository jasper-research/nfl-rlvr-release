"""Layer 2 — convert play-by-play rows into QA pairs with verifiable ground truth.

Each record:
    {
      "question":        natural-language game-state prompt,
      "target_prob":     vegas_wp  (P(posteam wins); SFT regression target),
      "actual_outcome":  1 if posteam won the game else 0  (reward ground truth),
      "game_id", "season", "week",
      "difficulty":      uncertainty in [0,1] (1 == coin-flip game state),
      "features":        structured fields a reasoning-trace generator can ground on,
    }

IMPORTANT correctness note
--------------------------
nflverse `result` is the HOME team's final margin (home_score - away_score),
while `vegas_wp` is the POSSESSION team's win probability. The outcome label must
therefore be aligned to the possession team, not to `result > 0`. We do that here.
Ties (result == 0) are dropped — calibration ground truth must be binary.

CLI
---
    python -m data.build_qa_pairs --raw data/raw/pbp_2023.parquet --out data/qa
    python -m data.build_qa_pairs --raw data/raw/pbp_2015_2024.parquet --out data/qa --split-seasons
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Season splits per the plan: train 2015-2022, eval 2023 (held out), test 2024.
TRAIN_SEASONS = set(range(2015, 2023))
EVAL_SEASONS = {2023}
TEST_SEASONS = {2024}


def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(int(n), f"{int(n)}th")


def _field_position(posteam: str, defteam: str, yardline_100) -> str:
    """yardline_100 = yards from the opponent's goal line."""
    if yardline_100 is None or (isinstance(yardline_100, float) and yardline_100 != yardline_100):
        return "field position unknown"
    y = int(round(yardline_100))
    if y == 50:
        return "at midfield (the 50)"
    if y > 50:
        return f"at their own {100 - y} yard line"
    return f"at the {defteam} {y} yard line (opponent territory)"


def _score_clause(score_diff) -> str:
    d = int(round(score_diff)) if score_diff == score_diff else 0
    if d > 0:
        return f"leading by {d}"
    if d < 0:
        return f"trailing by {abs(d)}"
    return "tied"


def _spread_clause(posteam: str, home_team: str, spread_line) -> str:
    """nflverse spread_line: positive == home team favored, in points."""
    if spread_line is None or spread_line != spread_line:
        return ""
    home_fav_by = float(spread_line)
    posteam_fav_by = home_fav_by if posteam == home_team else -home_fav_by
    if abs(posteam_fav_by) < 1e-6:
        return f"\n- Pregame line: pick'em"
    side = "favored by" if posteam_fav_by > 0 else "underdogs by"
    return f"\n- Pregame line: {posteam} {side} {abs(posteam_fav_by):.1f}"


def _player_clause(play) -> str:
    bits = []
    p = play.get("passer_player_name")
    py = play.get("passing_yards")
    if isinstance(p, str) and p:
        yd = f" ({int(py)} pass yds so far)" if py == py and py is not None else ""
        bits.append(f"{p} at QB{yd}")
    r = play.get("rusher_player_name")
    ry = play.get("rushing_yards")
    if isinstance(r, str) and r:
        yd = f" ({int(ry)} rush yds so far)" if ry == ry and ry is not None else ""
        bits.append(f"lead back {r}{yd}")
    return ("\n- Personnel on the play: " + "; ".join(bits)) if bits else ""


def _clock_clause(qtr, clock) -> str:
    """Overtime-aware time description. qtr 5 = OT; never render 'Nth quarter' for it."""
    clock = clock or "??:??"
    if qtr == 5:
        return f"Overtime, {clock} left in overtime"
    if qtr is None or qtr != qtr:
        return f"{clock} left"
    return f"{_ordinal(qtr)} quarter, {clock} left in the quarter"


def build_question(play: dict) -> str:
    posteam, defteam = play["posteam"], play["defteam"]
    when = _clock_clause(play.get("qtr"), play.get("time"))
    pos = _field_position(posteam, defteam, play.get("yardline_100"))
    score = _score_clause(play.get("score_differential"))
    dd = f"{_ordinal(play['down'])} and {int(play['ydstogo'])}" if play.get("down") == play.get("down") else "between plays"
    spread = _spread_clause(posteam, play.get("home_team"), play.get("spread_line"))
    players = _player_clause(play)

    return (
        f"NFL game: {posteam} (possession) vs {defteam}.\n"
        f"- {when}\n"
        f"- {posteam} {score}\n"
        f"- {dd}, ball {pos}"
        f"{spread}"
        f"{players}\n\n"
        f"Question: What is the probability that {posteam} win this game? "
        f"Reason step by step, then give a final probability as a percentage."
    )


def posteam_won(play: dict) -> "int | None":
    """1 if the possession team won, 0 if lost, None for ties (dropped)."""
    result = play.get("result")
    if result is None or result != result:
        return None
    margin = float(result)  # home_score - away_score
    if margin == 0:
        return None
    home_won = margin > 0
    is_home = play.get("posteam") == play.get("home_team")
    return int(home_won == is_home)


def difficulty(vegas_wp: float) -> float:
    """Coin-flip game states are hardest. 1.0 at p=0.5, 0.0 at p in {0,1}."""
    return float(1.0 - abs(2.0 * vegas_wp - 1.0))


def row_to_qa(play: dict) -> "dict | None":
    outcome = posteam_won(play)
    if outcome is None:
        return None
    vwp = play.get("vegas_wp")
    if vwp is None or vwp != vwp:
        return None
    vwp = float(vwp)

    features = {
        "posteam": play["posteam"], "defteam": play["defteam"],
        "qtr": int(play["qtr"]) if play.get("qtr") == play.get("qtr") else None,
        "time": play.get("time"),
        "game_seconds_remaining": (int(play["game_seconds_remaining"])
                                   if play.get("game_seconds_remaining") == play.get("game_seconds_remaining") else None),
        "down": int(play["down"]) if play.get("down") == play.get("down") else None,
        "ydstogo": int(play["ydstogo"]) if play.get("ydstogo") == play.get("ydstogo") else None,
        "yardline_100": (int(play["yardline_100"]) if play.get("yardline_100") == play.get("yardline_100") else None),
        "score_differential": (int(play["score_differential"]) if play.get("score_differential") == play.get("score_differential") else None),
        "vegas_wp": vwp,
        "wp": float(play["wp"]) if play.get("wp") == play.get("wp") else None,
        "epa": float(play["epa"]) if play.get("epa") == play.get("epa") else None,
        "spread_line": float(play["spread_line"]) if play.get("spread_line") == play.get("spread_line") else None,
    }
    return {
        "question": build_question(play),
        "target_prob": vwp,
        "actual_outcome": outcome,
        "game_id": play.get("game_id"),
        "season": int(play["season"]) if play.get("season") == play.get("season") else None,
        "week": int(play["week"]) if play.get("week") == play.get("week") else None,
        "difficulty": difficulty(vwp),
        "features": features,
    }


def build(raw_path: str, stride: int = 1):
    """Yield QA records from a raw parquet file. `stride` subsamples plays
    (stride=8 keeps ~1 in 8) to cut near-duplicate consecutive game states."""
    import pandas as pd

    df = pd.read_parquet(raw_path)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    n_in = n_out = 0
    for play in df.to_dict("records"):
        n_in += 1
        qa = row_to_qa(play)
        if qa is not None:
            n_out += 1
            yield qa
    print(f"{raw_path}: {n_out:,} QA pairs from {n_in:,} plays (stride={stride})")


def _split_for(season: "int | None") -> str:
    if season in EVAL_SEASONS:
        return "eval"
    if season in TEST_SEASONS:
        return "test"
    return "train"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="raw parquet from collect_pbp")
    ap.add_argument("--out", default="data/qa", help="output directory")
    ap.add_argument("--stride", type=int, default=8,
                    help="keep 1 in N plays to reduce redundant states (default 8)")
    ap.add_argument("--split-seasons", action="store_true",
                    help="route records into train/eval/test jsonl by season")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split_seasons:
        files = {s: (out_dir / f"qa_{s}.jsonl").open("w") for s in ("train", "eval", "test")}
        counts = {s: 0 for s in files}
        for qa in build(args.raw, stride=args.stride):
            split = _split_for(qa["season"])
            files[split].write(json.dumps(qa) + "\n")
            counts[split] += 1
        for f in files.values():
            f.close()
        print("wrote:", {k: f"{v:,}" for k, v in counts.items()})
    else:
        path = out_dir / "qa.jsonl"
        n = 0
        with path.open("w") as f:
            for qa in build(args.raw, stride=args.stride):
                f.write(json.dumps(qa) + "\n")
                n += 1
        print(f"wrote {n:,} QA pairs -> {path}")


if __name__ == "__main__":
    main()
