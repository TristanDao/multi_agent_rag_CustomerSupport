"""product_agent LangGraph node. Wraps `app.agents.product_agent.product_agent`."""
from app.agents.nodes._helpers import make_node
from app.agents.product_agent import product_agent as _product_agent_fn

product_agent_node = make_node("product_agent", _product_agent_fn)
