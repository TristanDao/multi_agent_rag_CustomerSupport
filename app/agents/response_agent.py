"""Response generation agent: compose final user-facing answer."""
import logging
from typing import Any, Dict, List, Optional

from app.core.llm import LLMUnavailable, chat, extract_content, extract_usage
from app.prompts import RESPONSE_SYSTEM, RAG_UNTRUSTED_DOC_RULE

logger = logging.getLogger(__name__)


def _format_data_block(agent_results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for r in agent_results:
        name = r.get("agent", "agent")
        data = r.get("data")
        if data:
            lines.append(f"[{name}] data: {data}")
        if r.get("decision"):
            lines.append(f"[{name}] decision: {r['decision']}")
    return "\n".join(lines) or "(no data)"


def _deterministic_answer(
    intent: str,
    agent_results: List[Dict[str, Any]],
    policy_chunks: List[Dict[str, Any]],
) -> str:
    """Compose a concise answer without calling the LLM."""
    parts: List[str] = []

    for r in agent_results:
        if r["agent"] == "product_agent":
            for blk in r["data"]:
                if "products" in blk and blk["products"]:
                    items = blk["products"][:3]
                    summary = ", ".join(
                        f"{p['product_name']} ({p['sku']}) - {p.get('base_price', 'N/A')}đ, còn {p.get('stock_available', 0)}"
                        for p in items
                    )
                    parts.append(f"Một số sản phẩm phù hợp: {summary}.")
                elif "line_total" in blk:
                    parts.append(
                        f"{blk.get('product_name', blk.get('sku',''))}: đơn giá {blk.get('unit_price')}đ "
                        f"cho {blk.get('quantity', 1)} đơn vị, tổng {blk['line_total']}đ "
                        f"(giảm {blk.get('discount_percent', 0)}%)."
                    )
                elif "total_available" in blk:
                    parts.append(
                        f"Tồn kho {blk.get('sku','')}: còn {blk['total_available']} đơn vị."
                    )
        elif r["agent"] == "order_agent":
            for blk in r["data"]:
                if blk.get("found") and "status" in blk:
                    parts.append(
                        f"Đơn {blk['order_id']}: trạng thái {blk['status']}, "
                        f"thanh toán {blk.get('payment_status')}, vận chuyển {blk.get('shipping_status')}."
                    )
        elif r["agent"] == "refund_decision_agent":
            d = r.get("decision") or {}
            if d:
                parts.append(
                    f"Quyết định đổi trả: {'Đủ điều kiện' if d.get('eligible') else 'Không đủ điều kiện'}. "
                    f"Lý do: {d.get('reason', 'N/A')}."
                )
                if d.get("requires_human_review"):
                    parts.append("Trường hợp này sẽ được chuyển cho nhân viên hỗ trợ xem xét.")
        elif r["agent"] == "sales_recommendation_agent":
            for blk in r["data"]:
                if blk.get("related", {}).get("products"):
                    rel = blk["related"]["products"][:3]
                    parts.append(
                        "Gợi ý thêm: " + ", ".join(p["product_name"] for p in rel) + "."
                    )
                if blk.get("search", {}).get("products"):
                    rel = blk["search"]["products"][:3]
                    parts.append(
                        "Các sản phẩm trong danh mục: " + ", ".join(p["product_name"] for p in rel) + "."
                    )

    if policy_chunks:
        # include up to 2 short policy citations
        cites = []
        for c in policy_chunks[:2]:
            content = c.get("content", "")
            content = content.replace("\n", " ")[:200]
            cites.append(f"[{c.get('source','')}#{c.get('section','')}] {content}")
        if cites:
            parts.append("Tham khảo chính sách: " + " || ".join(cites) + ".")

    if not parts:
        parts.append(
            "Tôi chưa tìm thấy thông tin phù hợp. Vui lòng cung cấp thêm chi tiết (mã SKU, mã đơn, số lượng)."
        )

    return " ".join(parts)


async def response_agent(
    intent: str,
    user_message: str,
    agent_results: List[Dict[str, Any]],
    policy_chunks: List[Dict[str, Any]],
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    extra_context = extra_context or {}
    deterministic = _deterministic_answer(intent, agent_results, policy_chunks)

    try:
        system_msg = RESPONSE_SYSTEM + "\n" + RAG_UNTRUSTED_DOC_RULE
        data_text = _format_data_block(agent_results)
        policy_text = "\n".join(
            f"[source: {c.get('source','')} | section: {c.get('section','')}] {c.get('content','')}"
            for c in policy_chunks[:4]
        )
        user_msg = (
            f"Câu hỏi khách: {user_message}\n"
            f"Intent: {intent}\n"
            f"Dữ liệu từ các agent:\n{data_text}\n\n"
            f"Chính sách tham khảo:\n{policy_text or '(không có)'}\n\n"
            f"Bối cảnh bổ sung: {extra_context}\n"
        )
        data = await chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=700,
        )
        answer = extract_content(data).strip() or deterministic
        usage = extract_usage(data)
        return {"answer": answer, "token_usage": usage, "used_llm": True}
    except LLMUnavailable:
        return {"answer": deterministic, "token_usage": {"input_tokens": 0, "output_tokens": 0}, "used_llm": False}
    except Exception as e:
        logger.warning("response_agent_llm_error err=%s", str(e))
        return {"answer": deterministic, "token_usage": {"input_tokens": 0, "output_tokens": 0}, "used_llm": False}
