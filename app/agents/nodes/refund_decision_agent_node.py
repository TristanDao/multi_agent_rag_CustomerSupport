"""refund_decision_agent LangGraph node. Wraps `app.agents.refund_decision_agent`."""
from app.agents.nodes._helpers import make_node
from app.agents.refund_decision_agent import refund_decision_agent as _refund_fn

refund_decision_agent_node = make_node("refund_decision_agent", _refund_fn)
