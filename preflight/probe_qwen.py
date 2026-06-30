"""Phase 3 — zero-shot preflight via an OpenAI-compatible endpoint (local oMLX).

Decision gate before training: does the base thinking model already emit a parseable final
probability? If yes (expected) -> skip SFT, go straight to GRPO with pure Brier, no format
warmup. If shaky -> enable the format-reward warmup.

Talks to any OpenAI-compatible /chat/completions server (oMLX locally; OpenRouter for the
frontier baseline). No model download — oMLX already serves Qwen3.6-35B-A3B-4bit.

    OMLX_API_KEY=omlx-xxxx .venv/bin/python -m preflight.probe_qwen \
        --base-url http://127.0.0.1:1234/v1 --model Qwen3.6-35B-A3B-4bit --n 20

Reports: format-success rate (the gate), <think> presence, predicted-prob distribution,
overconfidence, output length, and a few raw samples.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import urllib.request
from pathlib import Path

from reward.extract import extract_probability


def chat(base_url, api_key, model, content, temperature, max_tokens, timeout=180, system=""):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    choice = resp["choices"][0]
    msg = choice["message"]
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return text, reasoning, choice.get("finish_reason")


def list_models(base_url, api_key):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/models", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:1234/v1"))
    ap.add_argument("--model", default=os.environ.get("OMLX_MODEL", "Qwen3.6-35B-A3B-4bit"))
    ap.add_argument("--api-key", default=os.environ.get("OMLX_API_KEY", ""))
    ap.add_argument("--grpo", default="data/grpo/eval.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--think-suffix", default="", help="e.g. '\\n/think' to force thinking")
    ap.add_argument("--system", default="", help="system prompt")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    try:
        print("server models:", list_models(args.base_url, args.api_key))
    except Exception as e:
        print("(could not list models:", e, ")")

    prompts = [json.loads(l) for l in Path(args.grpo).open()][: args.n]
    suffix = args.think_suffix.replace("\\n", "\n")

    probs, lengths, n_think, n_parse, n_total, n_trunc = [], [], 0, 0, 0, 0
    shown = 0
    for rec in prompts:
        for _ in range(args.samples):
            text, reasoning, finish = chat(args.base_url, args.api_key, args.model,
                                           rec["prompt"] + suffix, args.temp, args.max_tokens,
                                           system=args.system)
            n_total += 1
            resp_truncated = finish == "length"
            if resp_truncated:
                n_trunc += 1
            full = (f"<think>{reasoning}</think>\n" if reasoning else "") + text
            lengths.append(len(text) + len(reasoning))
            if reasoning or "<think>" in text.lower():
                n_think += 1
            p = extract_probability(text) or extract_probability(full)
            if p is not None:
                n_parse += 1
                probs.append(p)
            print(f"[{n_total}] parsed={p} vegas={rec['vegas_wp']:.2f} "
                  f"outcome={rec['actual_outcome']} trunc={resp_truncated} chars={len(text)+len(reasoning)}")
            if shown < args.show:
                shown += 1
                print("\n" + "=" * 70)
                print("PROMPT:", rec["prompt"].split("Question")[0].strip()[:220])
                print(f"-> parsed: {p} | vegas_wp: {rec['vegas_wp']:.3f} | outcome: {rec['actual_outcome']}")
                if reasoning:
                    print("THINK:", reasoning[:200])
                print("ANSWER head:", text[:200])
                print("ANSWER tail:", text[-200:])
                print("truncated?:", resp_truncated)

    print("\n" + "#" * 70)
    print(f"samples: {n_total}")
    print(f"format-success rate : {n_parse / max(n_total,1):.1%}   <- gate (want >= ~90%)")
    print(f"truncated (length)  : {n_trunc / max(n_total,1):.1%}   <- if high, raise max_tokens")
    print(f"<think> present     : {n_think / max(n_total,1):.1%}")
    if probs:
        print(f"pred prob  mean={st.mean(probs):.3f}  std={st.pstdev(probs):.3f}  "
              f"min={min(probs):.2f}  max={max(probs):.2f}")
        print(f"mean confidence |p-0.5| : {st.mean(abs(p - 0.5) for p in probs):.3f}")
        extreme = sum(1 for p in probs if p > 0.9 or p < 0.1) / len(probs)
        print(f"extreme (>0.9 or <0.1)  : {extreme:.1%}  (overconfidence signal)")
    if lengths:
        print(f"output length chars: median={int(st.median(lengths))} max={max(lengths)}")
    print("\nDecision: high format-success -> pure Brier, no warmup. Low -> format warmup.")


if __name__ == "__main__":
    main()
