"""WS2 reasoning-quantification judge (Phase 6).

Audits each model's chain-of-thought (from eval/results/reasoning_dump_<split>.jsonl) against the
EXACT ground-truth game state, using a blinded frontier judge (OpenRouter, cached). For every
(play, model) it scores five ground-truth-checkable state-reading errors plus faithfulness, then
aggregates per-model rates with bootstrap 95% CIs.

The judge is blinded: it never sees which model produced the reasoning (one CoT at a time, no
label), so it can't favor base/WS2/v6–v9. Each error is true ONLY on an explicit contradiction of
the true state (omission ≠ error), and the judge must quote the offending text — so every flag is
auditable.

The claim this tests:
    masking calibrates WITHOUT corrupting reasoning → WS2 state-error rates ≈ base, ≪ v6–v9.

Run (anywhere with OPENROUTER_API_KEY; results cached to eval/cache/):
    python -m eval.judge_reasoning --split eval                 # judge + aggregate
    python -m eval.judge_reasoning --split eval --limit 20      # cheap pilot first
    python -m eval.judge_reasoning --split eval --validate labels.jsonl   # vs hand labels (kappa)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.backends import openrouter                    # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

BOOL_CATS = ["possession_misread", "score_inversion", "spread_inversion", "clock_misread",
             "pregame_anchor"]

SYSTEM = ("You are an expert NFL analyst auditing a win-probability forecaster's written reasoning "
          "for FACTUAL errors against the TRUE game state you are given. Be strict and literal. "
          "Flag an error ONLY when the reasoning explicitly states something that contradicts the "
          "true state — never for a mere omission. Output ONLY a single JSON object, no prose.")

_SCHEMA_HINT = """Return JSON exactly:
{
 "possession_misread": {"v": true|false, "q": "<verbatim quote or empty>"},
 "score_inversion":    {"v": true|false, "q": "..."},
 "spread_inversion":   {"v": true|false, "q": "..."},
 "clock_misread":      {"v": true|false, "q": "..."},
 "pregame_anchor":     {"v": true|false, "q": "..."},
 "faithfulness":       {"v": "consistent"|"inconsistent"|"unclear", "q": "<one-line why>"}
}
Definitions:
- possession_misread: says the wrong team has the ball.
- score_inversion: says the wrong team leads/trails (incl. "leading" when actually trailing).
- spread_inversion: treats the pregame favorite as the underdog, or vice versa.
- clock_misread: states the wrong quarter, or a materially wrong time remaining.
- pregame_anchor: bases the win-prob conclusion on the pregame line while ignoring/contradicting the
  CURRENT live score (e.g. calls a team that is currently LEADING unlikely to win because it was the
  pregame underdog). Only flag when the live score is non-tied and the reasoning lets the line override it.
- faithfulness: does the stated final probability follow from the reasoning's own conclusion?"""


def judge_prompt(gt: dict, model_out: dict) -> str:
    sp = gt["spread"]
    spread = (f"{sp['team']} {sp['side']} by {sp['points']}" if sp["side"] in ("favored", "underdog")
              else ("pick'em" if sp["side"] == "pickem" else "unknown"))
    dd = (f"{gt['down']} and {gt['ydstogo']}" if gt.get("down") else "n/a")
    return (
        f"TRUE game state (ground truth):\n"
        f"- Possession: {gt['posteam']} has the ball; opponent is {gt['defteam']}\n"
        f"- Score: {gt['score']}\n"
        f"- Period: {gt['period']}, {gt['clock']} left\n"
        f"- Down & distance: {dd}\n"
        f"- Field position: {gt['field_position']}\n"
        f"- Pregame line: {spread}\n\n"
        f"FORECASTER REASONING:\n\"\"\"\n{model_out['text'].strip()}\n\"\"\"\n\n"
        f"The forecaster's final stated probability that {gt['posteam']} wins: "
        f"{model_out.get('parsed_prob')}\n\n" + _SCHEMA_HINT)


def _parse_json(txt: str):
    s = txt.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if "```" in s[3:] else s.strip("`")
        s = s[4:] if s.lower().startswith("json") else s
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < 0:
        return None
    try:
        return json.loads(s[a:b + 1])
    except Exception:
        return None


def judge_one(backend, gt, model_out):
    prompt = judge_prompt(gt, model_out)
    out = backend.generate(prompt, max_tokens=900, temperature=0.0, top_p=1.0, system=SYSTEM)
    v = _parse_json(out["text"])
    if v is None:  # one repair attempt (different prompt -> not a cache hit)
        out = backend.generate(prompt + "\n\nReturn ONLY the JSON object.", max_tokens=900,
                               temperature=0.0, top_p=1.0, system=SYSTEM)
        v = _parse_json(out["text"])
    return v


def boot_ci(flags, reps=2000, seed=0):
    a = np.asarray(flags, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(reps, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(judged: list, labels: list):
    """judged: rows {ground_truth, models:{label:{verdict|None,...}}}. Print per-model rate+CI."""
    print("\n" + "#" * 78)
    for cat in BOOL_CATS + ["faithfulness(inconsistent)"]:
        print(f"\n[{cat}]  rate [95% CI]   (n parsed)")
        for label in labels:
            flags, n_parsed = [], 0
            for row in judged:
                m = row["models"].get(label, {})
                v = m.get("verdict")
                if not v:
                    continue
                n_parsed += 1
                if cat == "faithfulness(inconsistent)":
                    flags.append(1 if v.get("faithfulness", {}).get("v") == "inconsistent" else 0)
                else:
                    flags.append(1 if v.get(cat, {}).get("v") else 0)
            mean, lo, hi = boot_ci(flags)
            print(f"   {label:<8} {mean:6.1%} [{lo:5.1%},{hi:5.1%}]   (n={n_parsed})")
    print("\n" + "#" * 78)
    print("Lower is better for every row. Claim holds if WS2 ≈ base and both ≪ v6–v9 on the "
          "error rows, and WS2 faithfulness(inconsistent) is low.")


def cohens_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum(((a == c).mean() * (b == c).mean()) for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def validate(judged, labels_path):
    """labels.jsonl rows: {game_id, model, <cat>: bool, ...}. Report judge↔human agreement + kappa."""
    hand = {(r["game_id"], r["model"]): r for r in (json.loads(l) for l in open(labels_path))}
    print(f"\n[validate] {len(hand)} hand-labeled (play,model) pairs")
    for cat in BOOL_CATS:
        H, J = [], []
        for row in judged:
            gid = row["ground_truth"]["game_id"]
            for label, m in row["models"].items():
                key = (gid, label)
                if key in hand and m.get("verdict"):
                    H.append(int(bool(hand[key].get(cat))))
                    J.append(int(bool(m["verdict"].get(cat, {}).get("v"))))
        if H:
            agree = np.mean(np.array(H) == np.array(J))
            print(f"   {cat:<20} agree={agree:5.1%}  kappa={cohens_kappa(H, J):+.2f}  (n={len(H)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N plays (pilot)")
    ap.add_argument("--validate", default=None, help="hand-labels jsonl -> agreement + kappa")
    args = ap.parse_args()

    dump_path = RESULTS / f"reasoning_dump_{args.split}.jsonl"
    rows = [json.loads(l) for l in open(dump_path)]
    if args.limit:
        rows = rows[:args.limit]
    labels = list(rows[0]["models"].keys()) if rows else []
    backend = openrouter(args.judge_model)
    print(f"[judge_reasoning] {len(rows)} plays × {labels}  judge={args.judge_model}")

    n_fail = 0
    for i, row in enumerate(rows):
        for label, m in row["models"].items():
            v = judge_one(backend, row["ground_truth"], m)
            m["verdict"] = v
            n_fail += (v is None)
        if (i + 1) % 25 == 0:
            print(f"  judged {i + 1}/{len(rows)} plays")
    if n_fail:
        print(f"  WARNING: {n_fail} verdicts failed to parse (excluded from rates)")

    out_path = RESULTS / f"reasoning_judged_{args.split}.jsonl"
    with open(out_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"-> {out_path}")

    aggregate(rows, labels)
    if args.validate:
        validate(rows, args.validate)


if __name__ == "__main__":
    main()
