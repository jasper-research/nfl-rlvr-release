"""Phase 5 — evaluate a model (or the Vegas baseline) on a held-out split.

Computes the calibration metrics that ARE the paper's results: Brier, ECE, Murphy decomposition
(reliability/resolution/uncertainty), hard accuracy, and a reliability diagram — plus coverage
(parse-success, truncation) and a side-by-side Vegas comparison on the same items.

Examples:
    # Vegas baseline (instant, no model) on all of 2023:
    python -m eval.evaluate --backend vegas --split eval

    # Qwen3.6 zero-shot via the local oMLX server, 150 games:
    OMLX_API_KEY=... python -m eval.evaluate --backend omlx --split eval --n 150 --max-tokens 6144

    # Frontier reference via OpenRouter:
    OPENROUTER_API_KEY=... python -m eval.evaluate --backend openrouter --model anthropic/claude-sonnet-4.6 --split eval --n 150

Writes eval/results/<name>_<split>.json (metrics + per-bin reliability) and, if matplotlib is
present, eval/results/<name>_<split>_reliability.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a repo-root .env into os.environ (no extra deps)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

from reward.extract import extract_probability          # noqa: E402
from reward.metrics import (                             # noqa: E402
    binary_accuracy,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    murphy_decomposition,
    reliability_curve,
)
from eval import backends as B                           # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def load_split(split: str, n: int, sample_seed=None):
    """First-n by default. With sample_seed, take a RANDOM n (the grpo files are game-ordered, so
    first-n = weeks 1-4 only — a biased early-season slice; use sample_seed for a representative sub-set)."""
    rows = [json.loads(l) for l in open(f"data/grpo/{split}.jsonl")]
    if n <= 0 or n >= len(rows):
        return rows
    if sample_seed is not None:
        import random
        idx = sorted(random.Random(sample_seed).sample(range(len(rows)), n))
        return [rows[i] for i in idx]
    return rows[:n]


def _one(backend, r, max_tokens, temperature, top_p):
    g = backend.generate(r["prompt"], max_tokens=max_tokens,
                         temperature=temperature, top_p=top_p)
    text, reasoning = g["text"], g["reasoning"]
    truncated = g.get("finish_reason") == "length"
    full = (f"<think>{reasoning}</think>\n" if reasoning else "") + text
    p = extract_probability(text) or extract_probability(full)
    return {"game_id": r.get("game_id"), "prob": p, "outcome": int(r["actual_outcome"]),
            "vegas": r["vegas_wp"], "truncated": truncated, "parsed": p is not None}


def predict(backend, rows, max_tokens, temperature, top_p, concurrency=1):
    """Return per-row dicts IN ORDER. concurrency>1 runs API calls in a thread pool (each call is
    independently disk-cached, so this is safe + resumable). Use it for big API runs (5K plays)."""
    if concurrency <= 1:
        out = []
        for i, r in enumerate(rows):
            out.append(_one(backend, r, max_tokens, temperature, top_p))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(rows)} done")
        return out
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out = [None] * len(rows)
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_one, backend, r, max_tokens, temperature, top_p): i
                for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(rows)} done")
    return out


def _bootstrap_ci(probs, outcomes, n_boot=2000, seed=0):
    """95% CIs for Brier and ECE by resampling predictions (no API cost)."""
    import numpy as np
    p, y = np.asarray(probs, float), np.asarray(outcomes, float)
    n = len(p)
    rng = np.random.default_rng(seed)
    briers, eces = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        briers.append(brier_score(p[idx], y[idx]))
        eces.append(expected_calibration_error(p[idx], y[idx]))
    pct = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return {"brier95": pct(briers), "ece95": pct(eces)}


def metrics_block(probs, outcomes):
    conf, acc, cnt = reliability_curve(probs, outcomes, n_bins=10)
    return {
        "n": len(probs),
        "brier": brier_score(probs, outcomes),
        "ece": expected_calibration_error(probs, outcomes),
        "mce": maximum_calibration_error(probs, outcomes),
        "accuracy": binary_accuracy(probs, outcomes),
        "ci": _bootstrap_ci(probs, outcomes),
        "murphy": murphy_decomposition(probs, outcomes),
        "reliability_curve": {
            "bin_confidence": [None if c != c else round(float(c), 4) for c in conf],
            "bin_accuracy": [None if a != a else round(float(a), 4) for a in acc],
            "bin_count": [int(c) for c in cnt],
        },
    }


def reliability_png(model_curve, vegas_curve, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping reliability PNG: {e})")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    for curve, lbl, mk in [(model_curve, title, "o-"), (vegas_curve, "Vegas", "s-")]:
        xs = [c for c in curve["bin_confidence"] if c is not None]
        ys = [a for c, a in zip(curve["bin_confidence"], curve["bin_accuracy"]) if c is not None]
        ax.plot(xs, ys, mk, label=lbl)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("empirical win rate")
    ax.set_title(f"Reliability — {title}"); ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True,
                    choices=["vegas", "omlx", "openrouter", "vllm", "deepseek"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--vllm-url", default=None)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="take a RANDOM --n sample (avoids the first-n weeks-1-4 bias); same seed = "
                         "same games, so frontier preds align with a matching eval_checkpoints sample")
    ap.add_argument("--save-preds", action="store_true",
                    help="dump per-game preds to preds_<split>_<name>.json (for the paired bootstrap)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel API calls (use 8-16 for big runs; each call is disk-cached)")
    args = ap.parse_args()

    _load_dotenv()
    rows = load_split(args.split, args.n, args.sample_seed)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Vegas: the prediction IS vegas_wp (the ceiling baseline).
    if args.backend == "vegas":
        probs = [r["vegas_wp"] for r in rows]
        outs = [int(r["actual_outcome"]) for r in rows]
        m = metrics_block(probs, outs)
        result = {"name": "vegas", "split": args.split, "coverage": 1.0,
                  "truncation": 0.0, "metrics": m}
        name = "vegas"
    else:
        if args.backend == "omlx":
            backend = B.omlx(args.model)
        elif args.backend == "openrouter":
            backend = B.openrouter(args.model or os.environ.get("OPENROUTER_MODEL")
                                   or "anthropic/claude-sonnet-4.6")
        elif args.backend == "deepseek":
            backend = B.deepseek(args.model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-pro")
        else:
            backend = B.vllm(args.vllm_url, args.model)
        name = backend.name

        preds = predict(backend, rows, args.max_tokens, args.temperature, args.top_p,
                        concurrency=args.concurrency)
        # clean calibration set: parsed AND not truncated
        clean = [d for d in preds if d["parsed"] and not d["truncated"]]
        probs = [d["prob"] for d in clean]
        outs = [d["outcome"] for d in clean]
        vegas = [d["vegas"] for d in clean]
        m = metrics_block(probs, outs)
        # Vegas on the SAME clean items, for an apples-to-apples gap
        m_vegas = metrics_block(vegas, outs)
        result = {
            "name": name, "split": args.split,
            "coverage": len(clean) / len(preds),
            "parse_rate": sum(d["parsed"] for d in preds) / len(preds),
            "truncation": sum(d["truncated"] for d in preds) / len(preds),
            "metrics": m,
            "vegas_on_same_items": m_vegas,
        }
        reliability_png(m["reliability_curve"], m_vegas["reliability_curve"],
                        RESULTS / f"{name}_{args.split}_reliability.png", name)
        if args.save_preds:
            # same shape as eval_checkpoints --save-preds, under one variant key = the model name,
            # so eval.paired_bootstrap can pair it (by row position) against a WS1/WS2 preds file.
            slim = {name: [{"game_id": d["game_id"], "prob": d["prob"], "outcome": d["outcome"],
                            "vegas": d["vegas"], "parsed": d["parsed"], "truncated": d["truncated"]}
                           for d in preds]}
            (RESULTS / f"preds_{args.split}_{name}.json").write_text(json.dumps(slim))
            print(f"-> per-game preds: {RESULTS / f'preds_{args.split}_{name}.json'}")

    (RESULTS / f"{name}_{args.split}.json").write_text(json.dumps(result, indent=2))
    mm = result["metrics"]
    print("\n" + "#" * 60)
    print(f"{name}  [{args.split}]  n={mm['n']}")
    if args.backend != "vegas":
        print(f"coverage={result['coverage']:.1%}  parse={result['parse_rate']:.1%}  "
              f"truncation={result['truncation']:.1%}")
    print(f"Brier={mm['brier']:.4f} (95% CI {mm['ci']['brier95']})  "
          f"ECE={mm['ece']:.4f} (95% CI {mm['ci']['ece95']})  "
          f"MCE={mm['mce']:.4f}  acc={mm['accuracy']:.3f}")
    print(f"Murphy: reliability={mm['murphy']['reliability']:.4f} "
          f"resolution={mm['murphy']['resolution']:.4f} uncertainty={mm['murphy']['uncertainty']:.4f}")
    if args.backend != "vegas":
        vm = result["vegas_on_same_items"]["metrics"] if "metrics" in result["vegas_on_same_items"] else result["vegas_on_same_items"]
        print(f"Vegas on same items: Brier={vm['brier']:.4f}  ECE={vm['ece']:.4f}")
    print(f"-> {RESULTS / f'{name}_{args.split}.json'}")


if __name__ == "__main__":
    main()
