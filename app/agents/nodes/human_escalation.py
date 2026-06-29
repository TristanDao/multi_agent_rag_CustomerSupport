"""human_escalation LangGraph node (terminal).

Sets `requires_human=True` and produces a fixed safe answer that tells the user a
human will follow up. An operator can later resume the same `thread_id` via
`graph.update_state(...)` + `graph.ainvoke(None, config)` to continue the flow.
"""
from __future__ import annotations

from typing import Any, Dict

from app.agents.state import SupportState


SAFE_ANSWER = (
    "Yêu cầu của bạn cần được nhân viên hỗ trợ xem xét thêm. "
    "Chúng tôi sẽ liên hệ lại trong thời gian sớm nhất. "
    "Cảm ơn bạn đã kiên nhẫn."
)


async def human_escalation_node(state: SupportState) -> Dict[str, Any]:
    return {
        "final_answer": SAFE_ANSWER,
        "requires_human": True,
        "nodes_visited": ["human_escalation"],
    }
