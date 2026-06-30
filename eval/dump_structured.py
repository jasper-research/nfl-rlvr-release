"""WS2 reasoning-quantification dump (Phase 6, on the pod / GPU).

For a stratified sample of held-out plays, generate the FULL chain-of-thought from each model
(base zero-shot · WS2 masked-CoT ckpt · v6–v9 full-CoT-RL ckpt) on the SAME prompts, and write one
JSONL row per play carrying the *ground-truth game state* (from the qa `features` dict, which is
exact-by-construction) alongside every model's completion + parsed probability.

This is the input to `eval/judge_reasoning.py`, which audits each CoT against the ground truth for
state-reading errors (possession/score/spread/clock misreads, pregame-anchoring) — the evidence
behind "masking calibrates WITHOUT corrupting reasoning (≈ base, ≪ v6–v9)".

We dump from data/qa/qa_<split>.jsonl (not data/grpo): the qa `question` text is byte-identical to
the grpo `prompt`, but qa rows also carry `features` (posteam, score_differential, spread_line,
yardline_100, down, qtr, time) = the ground truth we audit against. No fragile per-play join.

Run on the pod:
    python -m eval.dump_structured --split eval --n 250 \
        --model "base::cot" \
        --model "ws2:checkpoints/grpo-ws2-masked-b0/checkpoint-150:masked" \
        --model "v69:checkpoints/grpo-v9-qwen2.5-7b/checkpoint-150:cot"

Each --model is "label:adapter_path:mode"; adapter empty = base (no LoRA); mode ∈ {cot,direct,masked}
selects the prompt transform (cot = the raw reason-step-by-step prompt). Writes
eval/results/reasoning_dump_<split>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward.extract import extract_probability          # noqa: E402
# NOTE: `from train.grpo_train import _transform` is done lazily inside main() — it pulls trl, which
# isn't in the CPU test env; keeping it lazy lets the pure ground-truth helpers below stay testable.

RESULTS = Path(__file__).resolve().parent / "results"

_SPREAD_RE = re.compile(r"Pregame line:\s*(\w+)\s+(favored by|underdogs by)\s+([\d.]+)")
_PICKEM_RE = re.compile(r"Pregame line:\s*pick'?em")


def field_position(yardline_100) -> str:
    """yardline_100 = yards to the OPPONENT end zone (nflverse). 64 -> 'own 36'; 12 -> 'opponent 12'."""
    if yardline_100 is None:
        return "unknown"
    y = int(round(float(yardline_100)))
    if y > 50:
        return f"own {100 - y}"
    if y < 50:
        rz = " (red zone)" if y <= 20 else ""
        return f"opponent {y}{rz}"
    return "midfield (50)"


def score_state(posteam: str, sd) -> str:
    sd = int(round(float(sd)))
    if sd > 0:
        return f"{posteam} leading by {sd}"
    if sd < 0:
        return f"{posteam} trailing by {-sd}"
    return "tied"


def spread_from_question(question: str, posteam: str) -> dict:
    """Ground-truth pregame line, read from the (templated, correct-by-construction) prompt text."""
    if _PICKEM_RE.search(question):
        return {"side": "pickem", "team": posteam, "points": 0.0}
    m = _SPREAD_RE.search(question)
    if not m:
        return {"side": "unknown", "team": posteam, "points": None}
    team, side, pts = m.group(1), m.group(2), float(m.group(3))
    return {"side": "favored" if side == "favored by" else "underdog", "team": team, "points": pts}


def build_ground_truth(row: dict) -> dict:
    """Exact game state from the qa row's `features` + the spread clause. Pure/testable."""
    f = row["features"]
    qtr = int(f["qtr"])
    return {
        "game_id": row["game_id"], "season": row.get("season"), "week": row.get("week"),
        "posteam": f["posteam"], "defteam": f["defteam"],
        "period": "overtime" if qtr >= 5 else f"Q{qtr}",
        "clock": f.get("time"),
        "score": score_state(f["posteam"], f["score_differential"]),
        "score_differential": int(round(float(f["score_differential"]))),
        "down": int(f["down"]) if f.get("down") is not None else None,
        "ydstogo": int(f["ydstogo"]) if f.get("ydstogo") is not None else None,
        "field_position": field_position(f.get("yardline_100")),
        "spread": spread_from_question(row["question"], f["posteam"]),
        "vegas_wp": f.get("vegas_wp"), "target_prob": row.get("target_prob"),
        "actual_outcome": int(row["actual_outcome"]),
    }


def stratum(gt: dict) -> tuple:
    qtr = gt["period"]
    phase = "OT" if qtr == "overtime" else ("early" if qtr in ("Q1", "Q2") else "late")
    leverage = "close" if abs(gt["score_differential"]) <= 8 else "lopsided"
    return (phase, leverage, gt["spread"]["side"])


def stratified_sample(rows: list, n: int, seed: int) -> list:
    """Round-robin across (phase × leverage × spread-side) strata for balanced coverage."""
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(stratum(r["_gt"]), []).append(r)
    for v in by.values():
        rng.shuffle(v)
    keys = sorted(by, key=lambda k: (-len(by[k]), k))   # deterministic order
    out, i = [], 0
    while len(out) < min(n, len(rows)):
        progressed = False
        for k in keys:
            if i < len(by[k]):
                out.append(by[k][i]); progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
        i += 1
    return out


def parse_model_spec(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) < 3:
        raise SystemExit(f"--model must be 'label:adapter:mode' (adapter empty for base); got {spec!r}")
    label, mode = parts[0], parts[-1]
    adapter = ":".join(parts[1:-1])
    if mode not in ("cot", "direct", "masked"):
        raise SystemExit(f"mode must be cot|direct|masked; got {mode!r}")
    return {"label": label, "adapter": adapter or None, "mode": mode}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--model", action="append", dest="models", required=True,
                    help="repeatable: 'label:adapter_path:mode' (adapter empty=base; mode cot|direct|masked)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from train.grpo_train import _transform              # lazy (pulls trl; pod only)

    specs = [parse_model_spec(s) for s in args.models]

    rows = [json.loads(l) for l in open(f"data/qa/qa_{args.split}.jsonl")]
    for r in rows:
        r["_gt"] = build_ground_truth(r)
    sample = stratified_sample(rows, args.n, args.seed)
    print(f"[dump_structured] split={args.split}  sampled {len(sample)}/{len(rows)}  models={[s['label'] for s in specs]}")
    from collections import Counter
    print("  strata:", dict(Counter(stratum(r["_gt"]) for r in sample)))

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    has_lora = any(s["adapter"] for s in specs)
    llm = LLM(model=args.base, enable_lora=has_lora, max_lora_rank=args.max_lora_rank,
              dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=2048, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)

    # generate per model (each with its own prompt transform)
    gens = {}
    for i, s in enumerate(specs, start=1):
        msgs = [[{"role": "user", "content": _transform(r["question"], s["mode"] == "direct",
                                                        s["mode"] == "masked")}] for r in sample]
        lora = LoRARequest(s["label"], i, s["adapter"]) if s["adapter"] else None
        print(f"  generating {s['label']} (mode={s['mode']}, adapter={s['adapter']}) ...")
        outs = llm.chat(msgs, sampling, lora_request=lora, use_tqdm=True)
        gens[s["label"]] = [o.outputs[0].text for o in outs]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"reasoning_dump_{args.split}.jsonl"
    with open(out_path, "w") as fh:
        for j, r in enumerate(sample):
            models = {}
            for s in specs:
                txt = gens[s["label"]][j]
                models[s["label"]] = {"mode": s["mode"], "text": txt,
                                      "parsed_prob": extract_probability(txt)}
            fh.write(json.dumps({"ground_truth": r["_gt"], "models": models}) + "\n")
    print(f"-> {out_path}  ({len(sample)} plays × {len(specs)} models)")


if __name__ == "__main__":
    main()
