"""Intent classification + entity extraction for the orchestrator."""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.core.llm import LLMUnavailable, chat_json
from app.prompts import INTENT_DECISION_USER_TEMPLATE, ORCHESTRATOR_SYSTEM

logger = logging.getLogger(__name__)


VALID_INTENTS = {
    "product_search",
    "product_comparison",
    "inventory_check",
    "wholesale_pricing",
    "order_tracking",
    "return_refund",
    "shipping_policy",
    "payment_terms",
    "warranty_policy",
    "sales_recommendation",
    "human_escalation",
    "general_faq",
    "unknown",
}


INTENT_KEYWORDS = {
    "product_search": ["tìm", "có bán", "có loại", "có ", "sản phẩm", "giá dưới", "giá trên", "giá bao nhiêu", "giá rẻ", "giấy", "bút", "sổ", "bìa", "mực", "máy tính", "bảng", "khay"],
    "product_comparison": ["so sánh", "khác nhau", "nên chọn", "tốt hơn"],
    "inventory_check": ["còn hàng", "tồn kho", "hết hàng", "còn không", "số lượng tồn"],
    "wholesale_pricing": ["giá sỉ", "mua sỉ", "số lượng lớn", "báo giá sỉ", "giá buôn", "thùng", "hộp", "sỉ"],
    "order_tracking": ["đơn hàng", "đơn của tôi", "đơn dh", "đang ở đâu", "kiểm tra đơn", "lịch sử đơn", "đơn nào"],
    "return_refund": ["đổi trả", "hoàn tiền", "trả hàng", "đổi hàng", "refund", "return", "đổi", "trả lại"],
    "shipping_policy": ["phí ship", "vận chuyển", "giao hàng bao lâu", "ship", "giao tận nơi", "giao khi nào", "địa chỉ", "mất bao lâu", "thời gian giao", "chuẩn bị hàng"],
    "payment_terms": ["thanh toán", "công nợ", "net 7", "net 15", "net 30", "trả góp", "đặt cọc", "trả chậm"],
    "warranty_policy": ["bảo hành", "lỗi", "hỏng", "sửa chữa", "warranty"],
    "sales_recommendation": ["gợi ý", "nên mua thêm", "kèm theo", "thường mua", "combo", "bundle", "mua kèm"],
    "human_escalation": ["nói chuyện với người", "nhân viên", "tư vấn", "gặp nhân viên", "tổng đài viên", "gọi lại", "chuyển nhân viên", "hủy đơn", "cần duyệt", "có cần duyệt", "thẩm quyền"],
    "general_faq": ["là gì", "như thế nào", "tại sao", "có thể", "được không", "bao nhiêu ram", "tương thích", "êm tay", "gợi ý"],
}


def _heuristic_intent(message: str) -> Dict[str, Any]:
    """Keyword-based fallback intent classifier."""
    msg = (message or "").lower()
    scores: Dict[str, int] = {}
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in msg:
                scores[intent] = scores.get(intent, 0) + 1
    if not scores:
        return {"intent": "unknown", "confidence": 0.2, "required_agents": ["response_agent"], "entities": {}}

    # Prefer specific intents over the generic "general_faq" fallback when scores tie.
    priority = [
        "wholesale_pricing",
        "return_refund",
        "warranty_policy",
        "payment_terms",
        "shipping_policy",
        "human_escalation",
        "sales_recommendation",
        "inventory_check",
        "order_tracking",
        "product_search",
        "product_comparison",
        "general_faq",
        "unknown",
    ]
    best_intent = None
    best_score = -1
    for intent in priority:
        if intent in scores and scores[intent] > best_score:
            best_intent = intent
            best_score = scores[intent]
    if best_intent is None:
        best_intent = "unknown"
    return {
        "intent": best_intent,
        "confidence": min(0.9, 0.4 + 0.15 * best_score),
        "required_agents": _default_agents_for_intent(best_intent),
        "entities": _heuristic_entities(message),
    }


def _default_agents_for_intent(intent: str) -> List[str]:
    table = {
        "product_search": ["product_agent", "response_agent"],
        "product_comparison": ["product_agent", "response_agent"],
        "inventory_check": ["product_agent", "response_agent"],
        "wholesale_pricing": ["product_agent", "policy_rag_agent", "response_agent"],
        "order_tracking": ["order_agent", "response_agent"],
        "return_refund": ["refund_decision_agent", "policy_rag_agent", "response_agent"],
        "shipping_policy": ["policy_rag_agent", "response_agent"],
        "payment_terms": ["policy_rag_agent", "response_agent"],
        "warranty_policy": ["policy_rag_agent", "response_agent"],
        "sales_recommendation": ["sales_recommendation_agent", "product_agent", "response_agent"],
        "human_escalation": ["response_agent"],
        "general_faq": ["policy_rag_agent", "response_agent"],
        "unknown": ["response_agent"],
    }
    return table.get(intent, ["response_agent"])


def _heuristic_entities(message: str) -> Dict[str, Any]:
    entities: Dict[str, Any] = {}
    if not message:
        return entities
    m = re.search(r"(\d+)\s*(thùng|ram|hộp|cái|chiếc|cuốn|xấp|gói)", message, re.IGNORECASE)
    if m:
        entities["quantity"] = int(m.group(1))
    m = re.search(r"đơn\s*([A-Za-z0-9\-_]+)", message, re.IGNORECASE)
    if m:
        entities["order_id"] = m.group(1)
    m = re.search(r"(sku[\-: ]?\w+)", message, re.IGNORECASE)
    if m:
        entities["sku"] = m.group(1).replace(" ", "")
    if "sỉ" in message.lower() or "wholesale" in message.lower():
        entities["customer_type"] = "wholesale"
    elif "lẻ" in message.lower() or "retail" in message.lower():
        entities["customer_type"] = "retail"
    return entities


async def classify_intent(message: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
    """Use LLM when available; fall back to deterministic heuristic otherwise."""
    fallback = _heuristic_intent(message)
    fallback["entities"]["customer_id"] = customer_id
    try:
        result = await chat_json(
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                {
                    "role": "user",
                    "content": INTENT_DECISION_USER_TEMPLATE.format(
                        message=message, customer_id=customer_id or "(không có)"
                    ),
                },
            ],
            max_tokens=400,
        )
    except LLMUnavailable:
        return fallback
    except Exception as e:
        logger.warning("intent_llm_error err=%s", str(e))
        return fallback

    intent = result.get("intent") or fallback["intent"]
    if intent not in VALID_INTENTS:
        intent = fallback["intent"]
    agents = result.get("required_agents") or _default_agents_for_intent(intent)
    entities = result.get("entities") or {}
    if isinstance(entities, dict):
        entities = {**fallback["entities"], **entities}
    confidence = float(result.get("confidence", fallback["confidence"]))
    return {
        "intent": intent,
        "required_agents": agents,
        "entities": entities,
        "confidence": max(0.0, min(1.0, confidence)),
    }
