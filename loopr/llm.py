"""Tiered LLM client: local Ollama first, then OpenAI-compatible, then Anthropic.

Local-first by design - Loopr runs offline at zero per-token cost, with a cloud
key as an optional fallback. A custom ``generate`` callable can be injected so
tests (and reproducible runs) never touch the network.

No third-party HTTP dependency: requests go through the standard library.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Optional


class LLMError(Exception):
    """Raised when every configured provider fails to produce a completion."""


def _post(url: str, body: dict, headers: Optional[dict] = None, timeout: int = 120) -> dict:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class LLMClient:
    """Generate completions from whichever backend is available.

    Provider resolution when ``provider="auto"`` (the default): try Ollama, then
    OpenAI (if ``OPENAI_API_KEY`` is set), then Anthropic (if ``ANTHROPIC_API_KEY``
    is set). Set ``LOOPR_PROVIDER`` to pin one explicitly.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        generate: Optional[Callable[[str, str], str]] = None,
        provider: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self._inject = generate
        self.model = model or os.getenv("LOOPR_MODEL", "qwen2.5:14b")
        self.temperature = temperature
        self.provider = provider or os.getenv("LOOPR_PROVIDER", "auto")
        self.timeout = timeout
        self.last_provider: Optional[str] = None

    def generate(self, prompt: str, system: str = "") -> str:
        """Return the model's completion for ``prompt`` (with optional ``system``)."""
        if self._inject is not None:
            return self._inject(prompt, system)

        errors = []
        for name in self._provider_order():
            try:
                out = getattr(self, f"_{name}")(prompt, system)
                self.last_provider = name
                return out
            except Exception as exc:  # try the next tier
                errors.append(f"{name}: {exc}")
        raise LLMError("all providers failed -> " + " | ".join(errors))

    def _provider_order(self) -> list[str]:
        if self.provider != "auto":
            return [self.provider]
        order = ["ollama"]
        if os.getenv("OPENAI_API_KEY"):
            order.append("openai")
        if os.getenv("ANTHROPIC_API_KEY"):
            order.append("anthropic")
        return order

    def _ollama(self, prompt: str, system: str) -> str:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        body = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = _post(f"{host}/api/generate", body, timeout=self.timeout)
        return (data.get("response") or "").strip()

    def _openai(self, prompt: str, system: str) -> str:
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ["OPENAI_API_KEY"]
        model = os.getenv("LOOPR_OPENAI_MODEL", "gpt-4o-mini")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = {"model": model, "messages": messages, "temperature": self.temperature}
        data = _post(
            f"{base}/chat/completions",
            body,
            headers={"Authorization": f"Bearer {key}"},
            timeout=self.timeout,
        )
        return data["choices"][0]["message"]["content"].strip()

    def _anthropic(self, prompt: str, system: str) -> str:
        key = os.environ["ANTHROPIC_API_KEY"]
        model = os.getenv("LOOPR_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        body = {
            "model": model,
            "max_tokens": 1024,
            "system": system or "",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        data = _post("https://api.anthropic.com/v1/messages", body, headers=headers, timeout=self.timeout)
        return "".join(block.get("text", "") for block in data.get("content", [])).strip()
