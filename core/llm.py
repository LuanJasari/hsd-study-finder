"""Provider-Abstraktion: Claude, OpenAI, Ollama oder regelbasierter Fallback (kein LLM).

``get_chat_model()`` liefert ``(Anzeigename, Modell)``. Das Modell ist ``None``, wenn kein LLM
verfügbar ist – die Aufrufer greifen dann auf ihre regelbasierten Varianten zurück.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

import httpx

from config import settings

log = logging.getLogger(__name__)
FALLBACK_LABEL = "Regelbasierter Fallback (kein LLM)"


class ChatModel(Protocol):
    label: str

    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...


class AnthropicChat:
    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_timeout)
        self.label = f"Claude ({settings.anthropic_model})"

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        message = self._client.messages.create(
            model=settings.anthropic_model,
            max_tokens=700,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in message.content if block.type == "text")


class OpenAIChat:
    def __init__(self) -> None:
        import openai

        self._client = openai.OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout)
        self.label = f"OpenAI ({settings.openai_model})"

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = self._client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        )
        return response.choices[0].message.content or ""


class OllamaChat:
    def __init__(self) -> None:
        self.label = f"Ollama ({settings.ollama_model}, lokal)"

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        payload = {
            "model": settings.ollama_model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            payload["format"] = "json"
        response = httpx.post(f"{settings.ollama_base_url}/api/chat", json=payload, timeout=settings.llm_timeout * 4)
        response.raise_for_status()
        return response.json()["message"]["content"]


def _ollama_running() -> bool:
    try:
        return httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=0.7).is_success
    except httpx.HTTPError:
        return False


def _build(provider: str) -> ChatModel | None:
    try:
        if provider == "anthropic" and settings.anthropic_api_key:
            return AnthropicChat()
        if provider == "openai" and settings.openai_api_key:
            return OpenAIChat()
        if provider == "ollama" and _ollama_running():
            return OllamaChat()
    except Exception as exc:  # fehlendes SDK o. Ä. darf die Demo nicht verhindern
        log.warning("LLM-Provider %s nicht nutzbar: %s", provider, exc)
    return None


@lru_cache(maxsize=1)
def get_chat_model() -> tuple[str, ChatModel | None]:
    wanted = settings.llm_provider
    if wanted == "none":
        return FALLBACK_LABEL, None
    for provider in ([wanted] if wanted != "auto" else ["anthropic", "openai", "ollama"]):
        model = _build(provider)
        if model is not None:
            return model.label, model
    if wanted != "auto":
        log.warning("LLM_PROVIDER=%s ist nicht verfügbar – nutze den regelbasierten Fallback.", wanted)
    return FALLBACK_LABEL, None
