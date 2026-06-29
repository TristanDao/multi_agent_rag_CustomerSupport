"""sales_recommendation_agent LangGraph node. Wraps `app.agents.sales_recommendation_agent`."""
from app.agents.nodes._helpers import make_node
from app.agents.sales_recommendation_agent import sales_recommendation_agent as _sales_fn

sales_recommendation_agent_node = make_node("sales_recommendation_agent", _sales_fn)
