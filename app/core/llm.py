"""LLM client wrapper.

This module is the single source of LLM access. It uses the official OpenAI Python
SDK (`from openai import OpenAI`). The same SDK is used for two providers:

* OpenAI (default) — when `OPENAI_API_KEY` is set
* Alibaba DashScope (OpenAI-compatible) — when only `ALIBABA_API_KEY` is set, the
  OpenAI client is pointed at the Alibaba endpoint via `base_url`

The LangChain agents consume the same provider through `langchain_openai.ChatOpenAI`,
which internally uses the OpenAI SDK. The raw `openai.OpenAI` client exposed here is
intended for non-agent code (ingestion scripts, evaluators, custom utilities).

If neither key is configured, all calls raise `LLMUnavailable` and callers should fall
back to deterministic behaviour.
"""
import json
import logging
import threading
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


_client_lock = threading.Lock()
_client: Optional[OpenAI] = None


def _build_client() -> OpenAI:
    api_key = settings.llm_api_key()
    if not api_key:
        raise LLMUnavailable("No LLM API key configured (set OPENAI_API_KEY or ALIBABA_API_KEY)")
    base_url = settings.llm_base_url() or None
    return OpenAI(api_key=api_key, base_url=base_url)


def get_client() -> OpenAI:
    """Return a process-wide OpenAI client (built lazily, thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _build_client()
    return _client


def reset_client() -> None:
    """Drop the cached client. Useful in tests after env changes."""
    global _client
    _client = None


def is_configured() -> bool:
    return settings.has_llm_key


def _default_model() -> str:
    return settings.llm_model()


def _to_messages(messages: List[Dict[str, str]]):
    """Pass-through: OpenAI SDK accepts the same {role, content} dicts we already use."""
    return messages


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    response_format_json: bool = False,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Call the chat completions endpoint via the OpenAI SDK.

    Returns a dict in a stable shape that the rest of the codebase already understands
    (it mirrors the previous httpx response: `{"choices": [...], "usage": {...}}`).
    """
    client = get_client()
    chosen_model = model or _default_model()
    kwargs: Dict[str, Any] = {
        "model": chosen_model,
        "messages": _to_messages(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.warning("llm_call_failed model=%s err=%s", chosen_model, str(e)[:200])
        raise LLMUnavailable(f"LLM call failed: {e}") from e

    # Normalise to the legacy dict shape used elsewhere in the codebase.
    choice = resp.choices[0] if resp.choices else None
    content = choice.message.content if choice and choice.message else ""
    return {
        "choices": [{"message": {"role": "assistant", "content": content or ""}}],
        "usage": {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
        },
        "model": getattr(resp, "model", chosen_model),
    }


def extract_content(data: Dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def extract_usage(data: Dict[str, Any]) -> Dict[str, int]:
    usage = data.get("usage", {}) or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
    }


async def chat_json(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 600,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Call the LLM and parse JSON from the response (handles non-strict providers)."""
    data = await _run_chat_async(
        messages, model=model, temperature=temperature,
        max_tokens=max_tokens, response_format_json=True, timeout=timeout,
    )
    content = extract_content(data)
    try:
        return json.loads(content)
    except Exception:
        return _extract_json_fallback(content)


async def _run_chat_async(
    messages: List[Dict[str, str]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Async wrapper around the sync OpenAI client.

    The OpenAI SDK has both sync and async clients; the sync one is enough for our
    throughput. We run it in the default executor to avoid blocking the event loop.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: chat(messages, **kwargs))


def _extract_json_fallback(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"_raw": text}
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return {"_raw": text}


def build_langchain_chat_openai():
    """Build a `langchain_openai.ChatOpenAI` instance using the same provider.

    Returns None if no LLM is configured (so callers can skip LangChain paths).
    """
    if not is_configured():
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model(),
        api_key=settings.llm_api_key(),
        base_url=settings.llm_base_url() or None,
        temperature=0.2,
        timeout=30,
    )
