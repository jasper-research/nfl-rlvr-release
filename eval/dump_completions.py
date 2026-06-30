"""Diagnostic: dump raw completions + what extract_probability() pulls out, for the base model
vs a trained checkpoint, on the SAME held-out prompts. Tells us whether a decalibrated checkpoint
is (a) genuinely drifting pessimistic or (b) the parser is grabbing the wrong number as the RL
shifts the model's phrasing (which would make the eval numbers a parsing artifact, not real).

Run on the pod (GPU):
    python -m eval.dump_completions --adapter checkpoints/grpo-v8-qwen2.5-7b/checkpoint-50 --n 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward.extract import answer_region, extract_probability   # noqa: E402
from train.grpo_train import _transform                          # noqa: E402  (direct/masked prompt xform)


def load(split, n):
    return [json.loads(l) for _, l in zip(range(n), open(f"data/grpo/{split}.jsonl"))]


def gen(llm, sampling, prompts, lora):
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(msgs, sampling, lora_request=lora, use_tqdm=False)
    return [o.outputs[0].text for o in outs]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None, help="checkpoint dir to compare against base")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--direct", action="store_true", help="prompt with the direct (no-CoT) instruction")
    ap.add_argument("--masked", action="store_true", help="prompt with the WS2 masked-CoT instruction "
                    "(Probability: NN%) — use this for ws2 checkpoints so they see their trained format")
    ap.add_argument("--full", action="store_true", help="print the COMPLETE reasoning+answer (not just "
                    "the 220-char tail) — for judging whether the number follows from the reasoning")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = load(args.split, args.n)
    llm = LLM(model=args.base, enable_lora=bool(args.adapter), max_lora_rank=32,
              dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=2048, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)

    # Prompt both models with the SAME instruction the checkpoint was trained on, so the comparison
    # is apples-to-apples (and the masked checkpoint emits its learned "Probability: NN%" format).
    prompts = [_transform(r["prompt"], args.direct, args.masked) for r in rows]
    base_txt = gen(llm, sampling, prompts, None)
    ckpt_txt = gen(llm, sampling, prompts, LoRARequest("ckpt", 1, args.adapter)) if args.adapter else None

    for i, r in enumerate(rows):
        print("\n" + "=" * 100)
        print(f"[{i}] outcome={r['actual_outcome']}  target(p̂)={r.get('target')}  vegas={r['vegas_wp']:.3f}")
        print("PROMPT:", r["prompt"].split("Question:")[0].strip()[:300])
        for label, txt in [("BASE", base_txt[i])] + ([("CKPT", ckpt_txt[i])] if ckpt_txt else []):
            p = extract_probability(txt)
            ans = answer_region(txt)
            print(f"\n--- {label}  parsed_prob={p} ---")
            if args.full:
                print("FULL COMPLETION:\n" + txt.strip())
            else:
                print("ANSWER REGION (tail):", ans[-220:].replace("\n", " "))
    print("\n" + "=" * 100)
    print("Check: does `parsed_prob` match the probability the model actually states in the answer?")
    print("If parsed != stated -> parser artifact. If model genuinely says low numbers -> real drift.")


if __name__ == "__main__":
    main()
