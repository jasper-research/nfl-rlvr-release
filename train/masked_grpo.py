"""WS2 — masked-gradient GRPO: RL updates ONLY the answer (probability) tokens, not the reasoning.

Mechanism: TRL's GRPO multiplies the per-token policy-gradient loss (and the KL, and the length
normalizer) by `completion_mask`. We intersect that mask with an *answer-span* mask — 1 from the
"Probability:" marker to the end of the completion, 0 on the reasoning tokens before it. So the
gradient and KL touch only the final-probability tokens; the chain-of-thought is left at base
quality. This is the fix for the v6–v9 failure (RL corrupting the reasoning → decalibration); it
keeps the human-readable reasoning while still calibrating the number. DCPO-style decoupling, done
by narrowing the existing mask (no reimplementation of the loss → robust across TRL versions).

We DON'T reimplement the loss; we only override the input-prep hook to shrink `completion_mask`.
"""

from __future__ import annotations

MASK_MARKER = "Probability:"   # the masked-mode prompt makes the model emit "Probability: NN%"


def answer_prefix_len(text: str, marker: str, encode) -> int | None:
    """Number of tokens BEFORE the answer span = len(encode(text up to the last `marker`)).
    Returns None if the marker isn't present. `encode(str)->list` is the tokenizer's encoder.
    Pure/​testable (no torch)."""
    pos = text.rfind(marker)
    if pos < 0:
        return None
    return len(encode(text[:pos]))


def answer_token_mask(completion_ids, tokenizer, completion_mask, marker: str = MASK_MARKER):
    """0/1 tensor (same shape as completion_mask): 1 on answer-span tokens (marker→end), 0 on
    reasoning. Rows without the marker fall back to the original completion_mask (standard GRPO),
    so a parse miss never zeros a whole update. Never unmasks padding."""
    import torch
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    out = torch.zeros_like(completion_mask)
    ids = completion_ids.tolist()
    for i, row in enumerate(ids):
        n = int(completion_mask[i].sum().item())          # real (non-pad) length
        text = tokenizer.decode(row[:n], skip_special_tokens=True)
        k = answer_prefix_len(text, marker, enc)
        if k is None or k >= n:
            out[i, :n] = 1                                  # marker absent -> don't mask this row
        else:
            out[i, k:n] = 1                                 # answer span only
    return out * completion_mask                            # keep padding zeroed


def make_masked_trainer_cls():
    """Build MaskedGRPOTrainer lazily (so importing this module doesn't require trl at import time
    in the CPU test env)."""
    from trl import GRPOTrainer

    class MaskedGRPOTrainer(GRPOTrainer):
        _mask_checked = False

        def _apply_mask(self, d):
            if isinstance(d, dict) and "completion_ids" in d and "completion_mask" in d:
                d["completion_mask"] = answer_token_mask(
                    d["completion_ids"], self.processing_class, d["completion_mask"])
                if not MaskedGRPOTrainer._mask_checked:
                    frac = float(d["completion_mask"].float().mean())
                    print(f"[masked-grpo] answer-span masking ACTIVE (mean kept-token frac={frac:.3f})")
                    MaskedGRPOTrainer._mask_checked = True
            elif not MaskedGRPOTrainer._mask_checked:
                print("[masked-grpo] WARNING: completion_ids/completion_mask not found in inputs — "
                      "masking NOT applied (TRL internals changed; verify _prepare_inputs output).")
                MaskedGRPOTrainer._mask_checked = True
            return d

        def _prepare_inputs(self, generation_batch):
            prepared = super()._prepare_inputs(generation_batch)
            if isinstance(prepared, list):
                return [self._apply_mask(d) for d in prepared]
            return self._apply_mask(prepared)

    return MaskedGRPOTrainer
