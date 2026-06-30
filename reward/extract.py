"""Thinking-aware probability extraction from model responses.

Qwen3.6 emits a `<think>…</think>` block then a final answer. The probability we score is the
one in the *answer* region; we fall back to the whole text only if the answer has no parseable
number (e.g. the model put it solely inside the thinking block, or emitted no think tags).

Priority within a region (the answer is, by our prompt format, a trailing percentage):
  1) the last explicit percentage;
  2) a number after the last *strong* answer cue (handles "probability is 0.73" with no %);
  3) the last bare number that coerces into [0, 1].
Percentages and bare numbers > 1 are divided by 100. Returns None if nothing valid is found.
"""

from __future__ import annotations

import re
from typing import Optional

_PCT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_NUM_RE = re.compile(r"(-?\d*\.\d+|-?\d+(?:\.\d+)?)")
# Specific multi-token cues only (bare "final"/"p" occur in ordinary prose).
_STRONG_CUE = re.compile(
    r"\b(?:final\s+probability|final\s+answer|win\s+probability|"
    r"probability(?:\s+(?:is|of))?|answer)\b\s*[:=]?\s*",
    re.IGNORECASE,
)
_THINK_CLOSE = "</think>"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


def answer_region(text: str) -> str:
    """The text the model intends as its answer, with the thinking trace removed."""
    if not text:
        return ""
    if _THINK_CLOSE in text.lower():
        # Everything after the final close tag is the answer.
        idx = text.lower().rindex(_THINK_CLOSE) + len(_THINK_CLOSE)
        return text[idx:]
    # No close tag: strip any complete blocks, then drop a dangling open block.
    t = _THINK_BLOCK.sub(" ", text)
    t = _THINK_OPEN.sub(" ", t)
    return t


def _coerce(raw: str, is_pct: bool) -> Optional[float]:
    try:
        v = float(raw)
    except ValueError:
        return None
    if is_pct:
        v /= 100.0
    elif v > 1.0:  # a bare number > 1 (e.g. "72") is almost certainly a percent
        v /= 100.0
    return v if 0.0 <= v <= 1.0 else None


def _scan(text: str) -> Optional[float]:
    if not text:
        return None
    for m in reversed(list(_PCT_RE.finditer(text))):
        v = _coerce(m.group(1), is_pct=True)
        if v is not None:
            return v
    for cue in reversed(list(_STRONG_CUE.finditer(text))):
        m = _NUM_RE.search(text[cue.end():])
        if m:
            v = _coerce(m.group(1), is_pct=False)
            if v is not None:
                return v
    for raw in reversed(_NUM_RE.findall(text)):
        v = _coerce(raw, is_pct=False)
        if v is not None:
            return v
    return None


def extract_probability(text: str) -> Optional[float]:
    """Win probability in [0, 1] from the answer region, falling back to full text."""
    if not text:
        return None
    p = _scan(answer_region(text))
    if p is not None:
        return p
    return _scan(text)
