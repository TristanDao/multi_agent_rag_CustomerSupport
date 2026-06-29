"""Agent Registry — single source of truth for every specialized agent.

The orchestrator reads this registry to:
  * build its `route_intent` system prompt
  * register each agent as a LangGraph node
  * validate that no intent is claimed by more than one agent
  * serve `GET /admin/agents` and `GET /admin/agents/{key}` from the API
  * generate the README Mermaid diagram via `scripts/generate_agent_graph.py`

The `node_fn` field is set lazily by `app.agents.nodes` to avoid import cycles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# --- Output schemas used by the registry ----------------------------
# Kept here as forward references; the real Pydantic models can be defined
# alongside the agents and re-exported later if needed.


class AgentInput(BaseModel):
    """Minimal base schema for an agent's input."""

    user_message: str = ""


class AgentOutput(BaseModel):
    """Minimal base schema for an agent's output."""

    summary: str = ""


@dataclass
class AgentSpec:
    """Declarative description of a single specialized agent."""

    key: str
    name: str
    description: str
    capabilities: List[str]
    intents: List[str]
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    tools: List[str] = field(default_factory=list)
    example_queries: List[str] = field(default_factory=list)
    node_fn: Optional[Callable] = None  # set at build time, references the LangGraph node

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "intents": list(self.intents),
            "tools": list(self.tools),
            "example_queries": list(self.example_queries),
            "input_schema": self.input_schema.__name__,
            "output_schema": self.output_schema.__name__,
            "has_node_fn": self.node_fn is not None,
        }


# --- Concrete input/output schemas for each agent -------------------


class ProductAgentInput(AgentInput):
    intent: str = "product_search"
    entities: Dict[str, Any] = field(default_factory=dict)


class ProductAgentOutput(AgentOutput):
    products: List[Dict[str, Any]] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)


class OrderAgentInput(AgentInput):
    intent: str = "order_tracking"
    entities: Dict[str, Any] = field(default_factory=dict)


class OrderAgentOutput(AgentOutput):
    order_id: Optional[str] = None
    status: Optional[str] = None
    tools_called: List[str] = field(default_factory=list)


class PolicyRAGAgentInput(AgentInput):
    intent: str = "general_faq"


class PolicyRAGAgentOutput(AgentOutput):
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)


class SalesRecommendationAgentInput(AgentInput):
    intent: str = "sales_recommendation"
    entities: Dict[str, Any] = field(default_factory=dict)


class SalesRecommendationAgentOutput(AgentOutput):
    related: List[Dict[str, Any]] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)


class RefundDecisionAgentInput(AgentInput):
    intent: str = "return_refund"
    entities: Dict[str, Any] = field(default_factory=dict)


class RefundDecisionAgentOutput(AgentOutput):
    eligible: bool = False
    reason: str = ""
    policy_sources: List[str] = field(default_factory=list)
    requires_human_review: bool = False
    tools_called: List[str] = field(default_factory=list)


# --- The Registry --------------------------------------------------

AGENT_REGISTRY: Dict[str, AgentSpec] = {
    "product_agent": AgentSpec(
        key="product_agent",
        name="Product Agent",
        description=(
            "Tìm kiếm sản phẩm theo tên/danh mục, kiểm tra tồn kho, "
            "tra bảng giá sỉ/lẻ theo số lượng, gợi ý sản phẩm liên quan."
        ),
        capabilities=["product_search", "inventory_check", "wholesale_pricing", "product_comparison"],
        intents=["product_search", "product_comparison", "inventory_check", "wholesale_pricing"],
        input_schema=ProductAgentInput,
        output_schema=ProductAgentOutput,
        tools=["search_products", "check_inventory", "get_price_for_quantity", "get_product_by_sku", "get_related_products"],
        example_queries=[
            "Tìm giấy A4 giá dưới 400k còn hàng ở HCM",
            "Mua 50 thùng giấy A4 thì giá sỉ bao nhiêu?",
        ],
    ),
    "order_agent": AgentSpec(
        key="order_agent",
        name="Order Agent",
        description=(
            "Tra cứu trạng thái đơn hàng, chi tiết đơn, lịch sử mua của khách hàng."
        ),
        capabilities=["order_tracking", "order_details", "customer_order_history"],
        intents=["order_tracking"],
        input_schema=OrderAgentInput,
        output_schema=OrderAgentOutput,
        tools=["get_order_status", "get_order_details", "get_customer_order_history"],
        example_queries=[
            "Đơn DH1024 của tôi đang ở đâu?",
            "Cho tôi xem lịch sử đơn hàng của khách C001.",
        ],
    ),
    "policy_rag_agent": AgentSpec(
        key="policy_rag_agent",
        name="Policy RAG Agent",
        description=(
            "Truy xuất các đoạn chính sách (vận chuyển, thanh toán, bảo hành) và FAQ "
            "từ vector DB để trả lời câu hỏi chính sách. Các intent về bán sỉ và đổi "
            "trả được xử lý trực tiếp bởi product_agent và refund_decision_agent, các "
            "agent đó tự gọi policy retrieval nội bộ khi cần."
        ),
        capabilities=["policy_retrieval", "faq_retrieval", "grounded_qa"],
        intents=[
            "shipping_policy",
            "payment_terms",
            "warranty_policy",
            "general_faq",
        ],
        input_schema=PolicyRAGAgentInput,
        output_schema=PolicyRAGAgentOutput,
        tools=["retrieve_policy_chunks"],
        example_queries=[
            "Phí ship nội thành HCM là bao nhiêu?",
            "Điều khoản NET 30 áp dụng khi nào?",
        ],
    ),
    "sales_recommendation_agent": AgentSpec(
        key="sales_recommendation_agent",
        name="Sales Recommendation Agent",
        description=(
            "Gợi ý sản phẩm liên quan, sản phẩm bán kèm (cross-sell), sản phẩm thay thế "
            "dựa trên danh mục, lịch sử mua và tồn kho."
        ),
        capabilities=["cross_sell", "bundle_suggestion", "reorder_suggestion", "alternative_suggestion"],
        intents=["sales_recommendation"],
        input_schema=SalesRecommendationAgentInput,
        output_schema=SalesRecommendationAgentOutput,
        tools=["get_customer_order_history", "search_products", "get_related_products"],
        example_queries=[
            "Khách nhà sách thường mua giấy A4 thì nên gợi ý thêm gì?",
        ],
    ),
    "refund_decision_agent": AgentSpec(
        key="refund_decision_agent",
        name="Refund Decision Agent",
        description=(
            "Kiểm tra đơn hàng, áp dụng chính sách đổi trả/hoàn tiền, đưa ra quyết định "
            "eligible và đề xuất leo thang nhân viên khi không chắc chắn."
        ),
        capabilities=["refund_decision", "return_eligibility", "policy_reasoning", "human_escalation_decision"],
        intents=["return_refund"],
        input_schema=RefundDecisionAgentInput,
        output_schema=RefundDecisionAgentOutput,
        tools=["get_order_details", "retrieve_policy_chunks"],
        example_queries=[
            "Đơn DH1024 giao 10 ngày rồi, tôi muốn trả lại được không?",
        ],
    ),
}

# `response_agent` and `human_escalation` are terminal nodes, not specialized agents.
# They are still addressable in the graph but are not part of AGENT_REGISTRY.
TERMINAL_NODES = ("response_agent", "human_escalation")


def set_node_fn(key: str, fn: Callable) -> None:
    """Bind a node function to a registered agent. Called once at startup."""
    if key not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent key: {key!r}. Add it to AGENT_REGISTRY first.")
    AGENT_REGISTRY[key].node_fn = fn


def validate_registry() -> None:
    """Fail fast at startup if the registry is inconsistent.

    * every entry must have a node_fn
    * intents must be unique across all entries
    """
    seen_intents: Dict[str, str] = {}
    for key, spec in AGENT_REGISTRY.items():
        if spec.node_fn is None:
            raise RuntimeError(
                f"Agent {key!r} has no node_fn. Call set_node_fn() for every registry entry at startup."
            )
        for intent in spec.intents:
            if intent in seen_intents and seen_intents[intent] != key:
                raise RuntimeError(
                    f"Intent {intent!r} is claimed by both {seen_intents[intent]!r} and {key!r}. "
                    f"Intents must be globally unique."
                )
            seen_intents[intent] = key


def registry_for_api() -> Dict[str, Any]:
    """Serialisable view of the registry for the admin API."""
    return {
        "agents": [spec.to_dict() for spec in AGENT_REGISTRY.values()],
        "total": len(AGENT_REGISTRY),
        "terminal_nodes": list(TERMINAL_NODES),
    }


def get_agent(key: str) -> Optional[AgentSpec]:
    return AGENT_REGISTRY.get(key)


def routing_prompt_block() -> str:
    """Build the routing block that gets prepended to the orchestrator system prompt.

    The orchestrator LLM reads this to decide which agent (node) to route to.
    """
    lines = ["Bạn có thể gọi một trong các agent sau đây:"]
    for spec in AGENT_REGISTRY.values():
        lines.append(
            f"- key: {spec.key} | name: {spec.name} | intents: {', '.join(spec.intents)}"
        )
        lines.append(f"  description: {spec.description}")
        if spec.example_queries:
            lines.append(f"  examples: {' | '.join(spec.example_queries[:2])}")
    return "\n".join(lines)
