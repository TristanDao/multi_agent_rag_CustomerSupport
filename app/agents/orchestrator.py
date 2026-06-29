"""Orchestrator entrypoint.

The previous build had a hand-rolled `run_orchestrator` that called the agents in a
sequence. In the new design, the actual orchestration is done by the LangGraph
StateGraph in `app.agents.graph`. This module:

* runs the input guardrail + PII redaction before invoking the graph
* invokes the compiled graph with the user's `thread_id`
* translates the final `SupportState` back into the dict shape that the
  `/chat` endpoint and existing tests already understand

The signature of `run_orchestrator` is intentionally kept compatible with the
previous build so existing tests (e.g. `tests/test_api_chat.py`) still work.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
from uuid import uuid4

from app.agents.graph import get_orchestrator_graph
from app.agents.state import empty_state
from app.config import settings
from app.core.observability import trace_request
from app.guardrails.input_guardrail import run_input_guardrail

logger = logging.getLogger(__name__)


async def run_orchestrator(
    message: str,
    request_id: str,
    customer_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the multi-agent pipeline via the LangGraph orchestrator.

    Returns a dict with: `answer`, `intent`, `agents_called`, `tools_called`,
    `policy_chunks`, `agent_results`, `guardrail`, `request_id`, `latency_ms`,
    `token_usage`, `requires_human`, `pii_redacted`, `thread_id`, `checkpoint_id`.
    """
    started = time.time()
    thread_id = thread_id or f"thread-{uuid4().hex}"
    final_thread_id = thread_id
    final_checkpoint_id: Optional[str] = None

    with trace_request(request_id, message) as tracer:
        tracer.set_metadata("customer_id", customer_id)
        tracer.set_metadata("thread_id", thread_id)

        # --- 1. Input guardrail ---
        if settings.ENABLE_INPUT_GUARDRAIL:
            input_gr, processed_message = run_input_guardrail(message)
            tracer.set_metadata("input_guardrail", input_gr)
            if input_gr["blocked"]:
                return _blocked_response(
                    request_id=request_id,
                    thread_id=thread_id,
                    safe_response=input_gr["safe_response"],
                    reason=input_gr["reason"],
                    started=started,
                    pii_redacted=input_gr.get("pii_redacted", False),
                )
        else:
            processed_message = message
            input_gr = {"blocked": False, "pii_redacted": False}

        # --- 2. Build initial state ---
        state = empty_state(
            user_message=processed_message,
            request_id=request_id,
            thread_id=thread_id,
            customer_id=customer_id,
        )
        state["pii_redacted"] = bool(input_gr.get("pii_redacted", False))
        state["guardrail"] = {"input": "passed", "output": "passed", "reason": None}

        # --- 3. Run the graph ---
        graph = get_orchestrator_graph()
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result_state = await graph.ainvoke(state, config=config)
        except Exception as e:
            logger.exception("graph_invoke_failed err=%s", str(e))
            return _error_response(
                request_id=request_id,
                thread_id=thread_id,
                started=started,
                error=str(e),
            )

        # The checkpointer stores the latest checkpoint under this thread_id.
        try:
            cp_state = graph.get_state(config)
            if cp_state and cp_state.config:
                final_checkpoint_id = (
                    cp_state.config.get("configurable", {}).get("checkpoint_id")
                    or cp_state.config.get("configurable", {}).get("thread_ts")
                )
        except Exception:
            final_checkpoint_id = None

        # --- 4. Translate state → API response ---
        answer = result_state.get("final_answer") or ""
        intent = result_state.get("intent", "unknown")
        agent_results_dict: Dict[str, Any] = result_state.get("agent_results") or {}
        agent_results_list = list(agent_results_dict.values())
        tools_called: list = list(result_state.get("tools_called") or [])
        nodes_visited: list = list(result_state.get("nodes_visited") or [])
        policy_chunks: list = []
        for r in agent_results_list:
            if r.get("agent") == "policy_rag_agent":
                policy_chunks = r.get("data") or []
                break
        requires_human = bool(result_state.get("requires_human", False))
        # If refund decision says it needs human review, propagate that flag.
        refund_result = agent_results_dict.get("refund_decision_agent") or {}
        refund_decision = refund_result.get("decision") or {}
        if refund_decision.get("requires_human_review"):
            requires_human = True

        guardrail = result_state.get("guardrail") or {"input": "passed", "output": "passed", "reason": None}
        latency_ms = int(result_state.get("latency_ms") or int((time.time() - started) * 1000))
        token_usage = result_state.get("token_usage") or {"input_tokens": 0, "output_tokens": 0}
        pii_redacted = bool(result_state.get("pii_redacted", False))

        tracer.set_metadata("intent", intent)
        tracer.set_metadata("agents_called", nodes_visited)
        tracer.set_metadata("tools_called", tools_called)
        tracer.set_metadata("latency_ms", latency_ms)
        tracer.set_metadata("thread_id", thread_id)
        tracer.set_token_usage(token_usage.get("input_tokens", 0), token_usage.get("output_tokens", 0))

        return {
            "answer": answer,
            "intent": intent,
            "agents_called": nodes_visited,
            "tools_called": tools_called,
            "policy_chunks": policy_chunks,
            "agent_results": agent_results_list,
            "guardrail": guardrail,
            "request_id": request_id,
            "latency_ms": latency_ms,
            "token_usage": token_usage,
            "requires_human": requires_human,
            "pii_redacted": pii_redacted,
            "thread_id": thread_id,
            "checkpoint_id": final_checkpoint_id,
        }


def _blocked_response(
    request_id: str,
    thread_id: str,
    safe_response: str,
    reason: str,
    started: float,
    pii_redacted: bool,
) -> Dict[str, Any]:
    return {
        "answer": safe_response,
        "intent": "blocked",
        "agents_called": [],
        "tools_called": [],
        "policy_chunks": [],
        "agent_results": [],
        "guardrail": {"input": "blocked", "output": "passed", "reason": reason},
        "request_id": request_id,
        "latency_ms": int((time.time() - started) * 1000),
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
        "requires_human": False,
        "pii_redacted": pii_redacted,
        "thread_id": thread_id,
        "checkpoint_id": None,
    }


def _error_response(
    request_id: str,
    thread_id: str,
    started: float,
    error: str,
) -> Dict[str, Any]:
    return {
        "answer": "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
        "intent": "error",
        "agents_called": [],
        "tools_called": [],
        "policy_chunks": [],
        "agent_results": [],
        "guardrail": {"input": "passed", "output": "passed", "reason": error},
        "request_id": request_id,
        "latency_ms": int((time.time() - started) * 1000),
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
        "requires_human": False,
        "pii_redacted": False,
        "thread_id": thread_id,
        "checkpoint_id": None,
    }
