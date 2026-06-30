"""Generation backends for evaluation — all OpenAI-compatible /chat/completions.

  * oMLX (local)      : the served Qwen3.6-35B-A3B-4bit, base_url http://127.0.0.1:1234/v1
  * OpenRouter        : frontier reference (e.g. Sonnet), base_url https://openrouter.ai/api/v1
  * vLLM checkpoint   : the trained model served via an OpenAI-compatible vLLM endpoint

Responses are cached to disk (eval/cache/) keyed by (model, prompt, params), so re-runs and
added metrics are free and reproducible. The Vegas baseline is not a backend — it reads
`vegas_wp` directly in evaluate.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "cache"


class OpenAICompatBackend:
    def __init__(self, base_url, api_key="", model="", name=None, timeout=300, extra_headers=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name or model
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt, params):
        blob = json.dumps({"m": self.model, "p": prompt, **params}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()

    def generate(self, prompt, max_tokens=6144, temperature=0.7, top_p=0.95, system=""):
        params = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "sys": system}
        cache = CACHE_DIR / f"{self._key(prompt, params)}.json"
        if cache.exists():
            return json.loads(cache.read_text())

        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": temperature, "top_p": top_p,
                           "max_tokens": max_tokens}).encode()
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.load(r)
        ch = resp["choices"][0]
        msg = ch["message"]
        out = {
            "text": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "finish_reason": ch.get("finish_reason"),
        }
        cache.write_text(json.dumps(out))
        return out


def omlx(model=None):
    m = model or os.environ.get("OMLX_MODEL", "Qwen3.6-35B-A3B-4bit")
    name = re.sub(r"-MLX-4bit$", "", m, flags=re.IGNORECASE).lower()  # e.g. qwen2.5-7b-instruct
    return OpenAICompatBackend(
        base_url=os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:1234/v1"),
        api_key=os.environ.get("OMLX_API_KEY", ""),
        model=m,
        name=name,
    )


def openrouter(model="anthropic/claude-sonnet-4.6"):
    return OpenAICompatBackend(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        model=model,
        name=model.split("/")[-1],
        extra_headers={"HTTP-Referer": "https://github.com/nfl-rlvr",
                       "X-Title": "nfl-rlvr-calibration"},
    )


def vllm(base_url, model, name="qwen3.6-grpo"):
    return OpenAICompatBackend(base_url=base_url, api_key=os.environ.get("VLLM_API_KEY", ""),
                               model=model, name=name)


def deepseek(model="deepseek-v4-pro"):
    """DeepSeek native API (OpenAI-compatible). Key from DEEPSEEK_API_KEY. Strong+cheap frontier
    reference. deepseek-v4-pro = thinking-capable; the backend already reads reasoning_content."""
    return OpenAICompatBackend(
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        model=model,
        name=model,
    )
