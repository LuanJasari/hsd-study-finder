"""Provider-Abstraktion: Claude, OpenAI, Ollama oder regelbasierter Fallback (kein LLM).

``get_chat_model()`` liefert ``(Anzeigename, Modell)``. Das Modell ist ``None``, wenn kein LLM
verfügbar ist – die Aufrufer greifen dann auf ihre regelbasierten Varianten zurück.

Transparenz: Die Auswahl und jeder LLM-Aufruf werden im Logger ``hsd.llm`` protokolliert (Terminal,
Stufe per ``LOG_LEVEL``). ``provider_notes()`` liefert dieselben Gründe für die Anzeige in der App.
Schneller Check der Konfiguration: ``python -m core.llm``.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Protocol

import httpx

from config import settings

log = logging.getLogger("hsd.llm")
FALLBACK_LABEL = "Regelbasierter Fallback (kein LLM)"
_PROVIDERS = ("anthropic", "openai", "ollama")


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


class LoggedChat:
    """Reicht Aufrufe an das echte Modell durch und protokolliert Dauer, Länge und Fehler."""

    def __init__(self, inner: ChatModel) -> None:
        self._inner = inner
        self.label = inner.label

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        start = time.perf_counter()
        try:
            reply = self._inner.complete(system, user, json_mode)
        except Exception as exc:
            log.warning("%s: Aufruf fehlgeschlagen nach %.1f s (%s: %s)",
                        self.label, time.perf_counter() - start, type(exc).__name__, exc)
            raise
        log.info("%s: Antwort in %.1f s (%d Zeichen)", self.label, time.perf_counter() - start, len(reply))
        log.debug("Prompt-Länge system=%d user=%d", len(system), len(user))
        return reply


def _ollama_running() -> bool:
    try:
        return httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=0.7).is_success
    except httpx.HTTPError:
        return False


def _build(provider: str) -> tuple[ChatModel | None, str]:
    """Versucht den Provider zu starten; liefert (Modell, Grund) – der Grund erklärt auch Fehlschläge."""
    try:
        if provider == "anthropic":
            if not settings.anthropic_api_key:
                return None, "ANTHROPIC_API_KEY nicht gesetzt"
            return AnthropicChat(), "API-Key vorhanden"
        if provider == "openai":
            if not settings.openai_api_key:
                return None, "OPENAI_API_KEY nicht gesetzt"
            return OpenAIChat(), "API-Key vorhanden"
        if provider == "ollama":
            if not _ollama_running():
                return None, f"Ollama nicht erreichbar unter {settings.ollama_base_url}"
            return OllamaChat(), f"Ollama läuft unter {settings.ollama_base_url}"
    except Exception as exc:  # fehlendes SDK o. Ä. darf die Demo nicht verhindern
        return None, f"Start fehlgeschlagen ({type(exc).__name__}: {exc})"
    return None, f"unbekannter Provider '{provider}'"


@lru_cache(maxsize=1)
def _select() -> tuple[str, ChatModel | None, tuple[str, ...]]:
    wanted = settings.llm_provider
    notes: list[str] = []
    if wanted == "none":
        notes.append("LLM_PROVIDER=none: LLM bewusst deaktiviert")
        log.info("LLM deaktiviert (LLM_PROVIDER=none) – regelbasierter Fallback aktiv.")
        return FALLBACK_LABEL, None, tuple(notes)
    candidates = list(_PROVIDERS) if wanted == "auto" else [wanted]
    for provider in candidates:
        model, reason = _build(provider)
        notes.append(f"{provider}: {reason}")
        if model is not None:
            log.info("LLM-Provider aktiv: %s (LLM_PROVIDER=%s; %s)", model.label, wanted, reason)
            return model.label, LoggedChat(model), tuple(notes)
    log.warning("Kein LLM verfügbar (LLM_PROVIDER=%s) – regelbasierter Fallback aktiv. %s", wanted, "; ".join(notes))
    return FALLBACK_LABEL, None, tuple(notes)


def get_chat_model() -> tuple[str, ChatModel | None]:
    label, model, _ = _select()
    return label, model


def provider_notes() -> list[str]:
    """Warum welcher Provider (nicht) aktiv ist, eine Zeile je geprüftem Provider."""
    return list(_select()[2])


def main() -> None:
    label, model = get_chat_model()
    print(f"Aktiver Provider: {label}")
    for note in provider_notes():
        print(f"  - {note}")
    if model is None:
        print("Hinweis: Ohne LLM nutzt der Study-Finder regelbasierte Fragen. Key in .env eintragen oder Ollama starten.")
        return
    print("Testaufruf …")
    print("Antwort:", model.complete("Antworte mit genau einem Wort.", "Sag OK.").strip()[:80])


if __name__ == "__main__":
    main()
