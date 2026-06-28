"""LLM client wrapper for Alibaba DashScope-compatible API."""
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


def _endpoint() -> str:
    base = settings.ALIBABA_URL.rstrip("/")
    return f"{base}/chat/completions"


def is_configured() -> bool:
    return bool(settings.ALIBABA_API_KEY)


async def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    response_format_json: bool = False,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Call the chat completions endpoint. Returns the parsed JSON response."""
    if not is_configured():
        raise LLMUnavailable("ALIBABA_API_KEY is not set")
    model = model or settings.ALIBABA_LLM_MODEL
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {settings.ALIBABA_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(_endpoint(), json=payload, headers=headers)
        if r.status_code >= 400:
            logger.warning("llm_http_error status=%s body=%s", r.status_code, r.text[:300])
            raise LLMUnavailable(f"LLM HTTP {r.status_code}")
        data = r.json()
    return data


def extract_content(data: Dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"]
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
    try:
        data = await chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=True,
            timeout=timeout,
        )
    except LLMUnavailable:
        raise
    content = extract_content(data)
    try:
        return json.loads(content)
    except Exception:
        return _extract_json_fallback(content)


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
