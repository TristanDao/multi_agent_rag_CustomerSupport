"""Orchestrator agent: routes intent, runs specialized agents, builds final answer."""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from app.agents.intent_classifier import classify_intent
from app.agents.order_agent import order_agent
from app.agents.policy_rag_agent import policy_rag_agent
from app.agents.product_agent import product_agent
from app.agents.refund_decision_agent import refund_decision_agent
from app.agents.response_agent import response_agent
from app.agents.sales_recommendation_agent import sales_recommendation_agent
from app.config import settings
from app.core.observability import trace_request
from app.guardrails.input_guardrail import run_input_guardrail
from app.guardrails.output_guardrail import run_output_guardrail

logger = logging.getLogger(__name__)


AGENT_MAP = {
    "product_agent": product_agent,
    "order_agent": order_agent,
    "policy_rag_agent": policy_rag_agent,
    "sales_recommendation_agent": sales_recommendation_agent,
    "refund_decision_agent": refund_decision_agent,
}


async def run_orchestrator(
    message: str,
    request_id: str,
    customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    started = time.time()
    with trace_request(request_id, message) as tracer:
        tracer.set_metadata("customer_id", customer_id)

        # --- 1. Input guardrail ---
        if settings.ENABLE_INPUT_GUARDRAIL:
            input_gr, processed_message = run_input_guardrail(message)
            tracer.set_metadata("input_guardrail", input_gr)
            if input_gr["blocked"]:
                return _blocked_response(
                    request_id=request_id,
                    safe_response=input_gr["safe_response"],
                    reason=input_gr["reason"],
                    started=started,
                    pii_redacted=input_gr.get("pii_redacted", False),
                )
        else:
            processed_message = message
            input_gr = {"blocked": False, "pii_redacted": False}

        # --- 2. Intent classification ---
        try:
            decision = await classify_intent(processed_message, customer_id=customer_id)
        except Exception as e:
            logger.warning("intent_classifier_failed err=%s", str(e))
            decision = {"intent": "unknown", "required_agents": ["response_agent"], "entities": {}, "confidence": 0.0}
        intent = decision["intent"]
        entities = decision.get("entities", {}) or {}
        entities.setdefault("customer_id", customer_id)
        agents = decision.get("required_agents") or ["response_agent"]
        tracer.set_metadata("intent", intent)
        tracer.set_metadata("entities", entities)
        tracer.set_metadata("confidence", decision.get("confidence", 0.0))

        # --- 3. Run specialized agents in order ---
        agent_results: List[Dict[str, Any]] = []
        tools_called: List[str] = []
        policy_chunks: List[Dict[str, Any]] = []
        requires_human = False

        for agent_name in agents:
            if agent_name == "product_agent":
                r = product_agent(intent, entities, processed_message)
                agent_results.append(r)
                tools_called.extend(r.get("tools_called", []))
            elif agent_name == "order_agent":
                r = order_agent(intent, entities, processed_message)
                agent_results.append(r)
                tools_called.extend(r.get("tools_called", []))
            elif agent_name == "policy_rag_agent":
                r = policy_rag_agent(intent, processed_message)
                agent_results.append(r)
                policy_chunks = r["data"] or []
                tools_called.extend(r.get("tools_called", []))
            elif agent_name == "sales_recommendation_agent":
                r = sales_recommendation_agent(intent, entities, processed_message)
                agent_results.append(r)
                tools_called.extend(r.get("tools_called", []))
            elif agent_name == "refund_decision_agent":
                r = await refund_decision_agent(intent, entities, processed_message)
                agent_results.append(r)
                tools_called.extend(r.get("tools_called", []))
                if r.get("decision", {}).get("requires_human_review"):
                    requires_human = True
            # response_agent is always called last and handled below
            elif agent_name == "response_agent":
                continue
            else:
                logger.warning("unknown_agent name=%s", agent_name)

        if intent == "human_escalation":
            requires_human = True

        # --- 4. Response generation ---
        resp = await response_agent(
            intent=intent,
            user_message=processed_message,
            agent_results=agent_results,
            policy_chunks=policy_chunks,
            extra_context={"requires_human": requires_human, "entities": entities},
        )
        answer = resp["answer"]
        token_usage = resp.get("token_usage", {})

        # --- 5. Output guardrail ---
        if settings.ENABLE_OUTPUT_GUARDRAIL:
            out_gr = run_output_guardrail(answer)
            if out_gr.get("blocked"):
                answer = out_gr.get("rewritten_answer") or answer
        else:
            out_gr = {"passed": True, "blocked": False}

        latency_ms = int((time.time() - started) * 1000)
        tracer.set_metadata("agents_called", agents)
        tracer.set_metadata("tools_called", tools_called)
        tracer.set_metadata("latency_ms", latency_ms)
        tracer.set_token_usage(
            token_usage.get("input_tokens", 0), token_usage.get("output_tokens", 0)
        )

        return {
            "answer": answer,
            "intent": intent,
            "agents_called": agents,
            "tools_called": tools_called,
            "policy_chunks": policy_chunks,
            "agent_results": agent_results,
            "guardrail": {
                "input": "passed" if not input_gr.get("blocked") else "blocked",
                "output": "passed" if out_gr.get("passed") else "blocked",
                "reason": out_gr.get("reason"),
            },
            "request_id": request_id,
            "latency_ms": latency_ms,
            "token_usage": token_usage,
            "requires_human": requires_human,
            "pii_redacted": input_gr.get("pii_redacted", False),
        }


def _blocked_response(
    request_id: str,
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
    }
