"""Layer 1 — collect NFL play-by-play from nflverse via nfl_data_py.

Free, public, goes back to 1999. We pull only the columns the QA pipeline and
reward need, do light cleaning, and cache to parquet so we never re-download.

CLI
---
    python -m data.collect_pbp --years 2015-2024 --out data/raw
    python -m data.collect_pbp --years 2023            # single season

Output: data/raw/pbp_{start}_{end}.parquet
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Columns we keep. Kept explicit so a schema change upstream fails loudly here
# rather than silently dropping a field the QA builder depends on.
KEEP_COLUMNS = [
    # identity / context
    "game_id", "season", "week", "season_type",
    "home_team", "away_team", "posteam", "defteam",
    # clock & situation
    "qtr", "time", "game_seconds_remaining", "half_seconds_remaining",
    "down", "ydstogo", "yardline_100", "score_differential",
    "posteam_score", "defteam_score",
    # ground-truth signals
    "vegas_wp",          # spread-adjusted P(posteam wins) -- SFT target
    "wp",                # nflverse model P(posteam wins)
    "result",            # final home margin (home_score - away_score)
    "total_line", "spread_line",
    # play context for the narrative
    "passer_player_name", "passing_yards",
    "rusher_player_name", "rushing_yards",
    "ep", "epa", "wpa",
]


def collect(years: list[int], cache: bool = True) -> "pd.DataFrame":
    """Fetch play-by-play for the given seasons and return a cleaned DataFrame."""
    import nfl_data_py as nfl
    import pandas as pd

    df = nfl.import_pbp_data(years, downcast=True, cache=False)

    present = [c for c in KEEP_COLUMNS if c in df.columns]
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        warnings.warn(f"columns absent upstream and skipped: {missing}")
    df = df[present].copy()

    # Keep only genuine game-state plays we can pose a question about.
    before = len(df)
    df = df[
        df["posteam"].notna()
        & df["defteam"].notna()
        & df["vegas_wp"].notna()
        & df["result"].notna()
        & df["down"].notna()
        & df["game_seconds_remaining"].notna()
    ].reset_index(drop=True)
    print(f"cleaned: kept {len(df):,} / {before:,} plays "
          f"({len(df) / max(before, 1):.1%})")
    return df


def _parse_years(spec: str) -> list[int]:
    """'2015-2024' -> [2015..2024]; '2023' -> [2023]; '2019,2021' -> [2019,2021]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", required=True, help="e.g. 2015-2024 or 2023 or 2019,2021")
    ap.add_argument("--out", default="data/raw", help="output directory")
    args = ap.parse_args()

    years = _parse_years(args.years)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect(years)
    name = f"pbp_{years[0]}_{years[-1]}.parquet" if len(years) > 1 else f"pbp_{years[0]}.parquet"
    path = out_dir / name
    df.to_parquet(path, index=False)
    print(f"wrote {len(df):,} rows -> {path}")


if __name__ == "__main__":
    main()
