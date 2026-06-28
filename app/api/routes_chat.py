"""Main /chat endpoint with multi-agent pipeline."""
import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, Request

from app.agents.orchestrator import run_orchestrator
from app.core.security import new_request_id
from app.db.schemas import ChatRequest, ChatResponse, GuardrailInfo, SourceInfo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    request_id = getattr(request.state, "request_id", None) or new_request_id()
    started = time.time()
    result: Dict[str, Any] = await run_orchestrator(
        message=req.message,
        request_id=request_id,
        customer_id=req.customer_id,
    )

    sources: list = []
    seen = set()
    for r in result.get("agent_results", []):
        for blk in r.get("data", []) or []:
            if isinstance(blk, dict) and "products" in blk:
                for p in blk["products"][:3]:
                    key = ("sql", p.get("sku", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    sources.append(SourceInfo(type="sql", name=f"products#{p.get('sku','')}"))
    for c in result.get("policy_chunks", [])[:4]:
        key = ("policy_doc", c.get("source", ""))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            SourceInfo(
                type="policy_doc",
                name=c.get("source", ""),
                section=c.get("section"),
                score=c.get("score"),
            )
        )

    gr = result.get("guardrail", {})
    return ChatResponse(
        answer=result.get("answer", ""),
        intent=result.get("intent", "unknown"),
        agents_called=result.get("agents_called", []),
        tools_called=result.get("tools_called", []),
        sources=sources,
        guardrail=GuardrailInfo(
            input=gr.get("input", "passed"),
            output=gr.get("output", "passed"),
            reason=gr.get("reason"),
        ),
        request_id=request_id,
        latency_ms=result.get("latency_ms", int((time.time() - started) * 1000)),
        token_usage=result.get("token_usage"),
        requires_human=result.get("requires_human", False),
        debug={
            "pii_redacted": result.get("pii_redacted", False),
            "agents_executed": [
                r.get("agent") for r in result.get("agent_results", [])
            ],
        },
    )
