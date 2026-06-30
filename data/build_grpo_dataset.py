"""Build the GRPO-ready dataset from the QA splits.

GRPO needs only prompts + the verifiable outcome. We strip each QA record down to:
    {
      "prompt":         game-state question  (NO vegas_wp — only the public spread),
      "actual_outcome": 1 if possession team won else 0   (the RLVR reward target),
      "vegas_wp":       eval-only near-ceiling baseline    (NEVER shown to the model),
      "game_id", "season", "difficulty",
      "meta": {qtr, score_differential, game_seconds_remaining, week}  # eval subgroups
    }

The thinking trigger (/think or the chat template's enable_thinking flag) is applied by the
generation backend at run time, NOT baked into `prompt` — so the dataset stays trigger-agnostic
and we don't regenerate it after the preflight settles the exact convention.

    python -m data.build_grpo_dataset            # reads data/qa/, writes data/grpo/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLITS = ("train", "eval", "test")


def to_grpo(rec: dict) -> dict:
    f = rec["features"]
    return {
        "prompt": rec["question"],
        "actual_outcome": rec["actual_outcome"],
        "vegas_wp": f.get("vegas_wp"),          # eval only
        "game_id": rec.get("game_id"),
        "season": rec.get("season"),
        "difficulty": rec.get("difficulty"),
        "meta": {
            "qtr": f.get("qtr"),
            "score_differential": f.get("score_differential"),
            "game_seconds_remaining": f.get("game_seconds_remaining"),
            "week": rec.get("week"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa-dir", default="data/qa")
    ap.add_argument("--out", default="data/grpo")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        src = Path(args.qa_dir) / f"qa_{split}.jsonl"
        n = 0
        with (out / f"{split}.jsonl").open("w") as w:
            for line in src.open():
                rec = to_grpo(json.loads(line))
                # Hard guarantee: the answer signal never leaks into the prompt.
                low = rec["prompt"].lower()
                assert "vegas" not in low, f"vegas leaked into prompt: {rec['game_id']}"
                if rec["vegas_wp"] is not None:
                    assert f"{rec['vegas_wp']:.2f}" not in rec["prompt"], "wp value in prompt"
                w.write(json.dumps(rec) + "\n")
                n += 1
        print(f"{split}: {n:,} -> {out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
