"""order_agent LangGraph node. Wraps `app.agents.order_agent.order_agent`."""
from app.agents.nodes._helpers import make_node
from app.agents.order_agent import order_agent as _order_agent_fn

order_agent_node = make_node("order_agent", _order_agent_fn)
