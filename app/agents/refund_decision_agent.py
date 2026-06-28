"""Refund decision agent: combines order data + return policy + LLM reasoning."""
import logging
from datetime import datetime
from typing import Any, Dict, List

from app.core.llm import LLMUnavailable, chat_json
from app.agents.policy_rag_agent import policy_rag_agent
from app.prompts import REFUND_DECISION_SYSTEM, RAG_UNTRUSTED_DOC_RULE
from app.tools.sql_tools import call_tool

logger = logging.getLogger(__name__)


def _summarise_chunks(chunks: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[source: {c.get('source','?')} | section: {c.get('section','?')}]\n{c.get('content','')}"
        for c in chunks[:4]
    )


def _deterministic_decision(order_id: str, order: Dict[str, Any], policy_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply deterministic rules when LLM is not available."""
    eligible = False
    reason = "Không đủ thông tin để quyết định."
    requires_human = True

    if not order.get("found"):
        return {
            "eligible": False,
            "reason": f"Không tìm thấy đơn hàng {order_id}.",
            "policy_sources": [c.get("source") for c in policy_chunks[:2]],
            "requires_human_review": True,
        }

    delivered_at = order.get("delivered_at")
    if delivered_at:
        try:
            delivered_dt = datetime.fromisoformat(delivered_at)
            days_since = (datetime.utcnow() - delivered_dt).days
            if days_since <= 7:
                eligible = True
                reason = f"Đơn đã giao {days_since} ngày trước, còn trong thời hạn đổi trả 7 ngày."
                requires_human = False
            else:
                reason = f"Đơn đã giao {days_since} ngày trước, vượt quá thời hạn đổi trả 7 ngày cho bán lẻ."
        except Exception:
            reason = "Không xác định được ngày giao hàng."
    else:
        reason = "Đơn chưa được giao, chưa thể xét đổi trả."

    return {
        "eligible": eligible,
        "reason": reason,
        "policy_sources": [c.get("source") for c in policy_chunks[:2]],
        "requires_human_review": requires_human,
    }


async def refund_decision_agent(intent: str, entities: Dict[str, Any], message: str) -> Dict[str, Any]:
    order_id = entities.get("order_id")
    customer_id = entities.get("customer_id")
    policy = policy_rag_agent("return_refund", message)

    blocks = {"policy": policy["data"]}
    order_info: Dict[str, Any] = {}
    if order_id:
        order_info = call_tool("get_order_details", order_id=order_id)
        blocks["order"] = order_info

    decision = _deterministic_decision(order_id or "(unknown)", order_info, policy["data"])

    try:
        result = await _llm_refund_reasoning(
            message=message,
            order_info=order_info,
            policy_chunks=policy["data"],
            base_decision=decision,
        )
        if isinstance(result, dict) and "eligible" in result:
            decision = {
                "eligible": bool(result.get("eligible", decision["eligible"])),
                "reason": str(result.get("reason", decision["reason"])),
                "policy_sources": result.get("policy_sources") or decision["policy_sources"],
                "requires_human_review": bool(
                    result.get("requires_human_review", decision["requires_human_review"])
                ),
            }
    except LLMUnavailable:
        pass
    except Exception as e:
        logger.warning("refund_llm_error err=%s", str(e))

    return {
        "agent": "refund_decision_agent",
        "intent": intent,
        "tools_called": ["get_order_details", "policy_retrieval"],
        "data": blocks,
        "decision": decision,
    }


async def _llm_refund_reasoning(
    message: str,
    order_info: Dict[str, Any],
    policy_chunks: List[Dict[str, Any]],
    base_decision: Dict[str, Any],
) -> Dict[str, Any]:
    policy_text = _summarise_chunks(policy_chunks)
    system = REFUND_DECISION_SYSTEM + "\n" + RAG_UNTRUSTED_DOC_RULE
    user = (
        f"Yêu cầu khách: {message}\n"
        f"Thông tin đơn hàng: {order_info}\n"
        f"Chính sách liên quan:\n{policy_text}\n"
        f"Quyết định khởi tạo (deterministic): {base_decision}\n"
        "Hãy trả về JSON: {eligible: bool, reason: str, policy_sources: [str], requires_human_review: bool}"
    )
    return await chat_json(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=400,
    )
