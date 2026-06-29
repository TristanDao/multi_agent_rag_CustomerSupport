"""response_agent LangGraph node (terminal). Composes the final user-facing answer.

The actual composition logic lives in `app.agents.response_agent.response_agent` and
is unchanged. This node:
  * calls the legacy response_agent
  * writes the answer to `state["final_answer"]`
  * accumulates token usage and latency (via the reducers declared in SupportState)
  * applies the output guardrail
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

from app.agents.response_agent import response_agent as _response_agent_fn
from app.agents.state import SupportState
from app.config import settings
from app.guardrails.output_guardrail import run_output_guardrail

logger = logging.getLogger(__name__)


async def response_agent_node(state: SupportState) -> Dict[str, Any]:
    started = time.time()
    intent = state.get("intent", "")
    user_message = state.get("user_message", "")
    agent_results = list((state.get("agent_results") or {}).values())
    policy_chunks: list = []
    for r in agent_results:
        if r.get("agent") == "policy_rag_agent":
            policy_chunks = r.get("data") or []
        elif r.get("agent") in ("refund_decision_agent", "product_agent"):
            # These agents may have pulled policy data internally. Look for a
            # `policy` block inside their `data` list (legacy shape from
            # refund_decision_agent) or for chunks sitting alongside other data.
            for blk in r.get("data") or []:
                if isinstance(blk, dict) and "policy" in blk and isinstance(blk["policy"], list):
                    policy_chunks.extend(blk["policy"])
                if isinstance(blk, dict) and blk.get("source") and blk.get("content"):
                    policy_chunks.append(blk)
    # Deduplicate by doc_id
    seen: set = set()
    deduped: list = []
    for c in policy_chunks:
        cid = c.get("doc_id") or (c.get("source"), c.get("section"), c.get("content", "")[:50])
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(c)

    extra_context = {
        "requires_human": state.get("requires_human", False),
        "entities": state.get("entities") or {},
    }

    resp = await _response_agent_fn(
        intent=intent,
        user_message=user_message,
        agent_results=agent_results,
        policy_chunks=deduped,
        extra_context=extra_context,
    )
    answer = resp.get("answer", "")
    token_usage = resp.get("token_usage") or {"input_tokens": 0, "output_tokens": 0}

    guardrail_patch: Dict[str, Any] = {}
    if settings.ENABLE_OUTPUT_GUARDRAIL:
        out_gr = run_output_guardrail(answer)
        if out_gr.get("blocked"):
            answer = out_gr.get("rewritten_answer") or answer
            guardrail_patch["output"] = "blocked"
            guardrail_patch["reason"] = out_gr.get("reason")
    else:
        guardrail_patch["output"] = "passed"

    elapsed = int((time.time() - started) * 1000)

    # Reducers in SupportState handle accumulation; we just return the new
    # contribution for this node.
    return {
        "final_answer": answer,
        "guardrail": guardrail_patch,
        "nodes_visited": ["response_agent"],
        "latency_ms": elapsed,
        "token_usage": token_usage,
    }
