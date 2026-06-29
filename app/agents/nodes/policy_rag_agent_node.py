"""policy_rag_agent LangGraph node. Wraps `app.agents.policy_rag_agent.policy_rag_agent`."""
from app.agents.nodes._helpers import make_node
from app.agents.policy_rag_agent import policy_rag_agent as _policy_fn

policy_rag_agent_node = make_node("policy_rag_agent", _policy_fn)
