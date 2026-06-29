"""LangGraph node implementations.

Each node is a pure function of `SupportState`: it reads fields from the input state
and returns a partial dict of fields to update. No module-level mutable state.

Node names (must match the keys used in `app.agents.registry` and `app.agents.graph`):
    * route_intent           — classifies user intent + entities
    * product_agent          — wraps the existing product_agent()
    * order_agent            — wraps the existing order_agent()
    * policy_rag_agent       — wraps the existing policy_rag_agent()
    * sales_recommendation_agent — wraps the existing sales_recommendation_agent()
    * refund_decision_agent  — wraps the existing refund_decision_agent()
    * response_agent         — terminal node, composes final answer
    * human_escalation       — terminal node, sets requires_human=True

All non-terminal nodes share the same shape: they call the existing agent function
(keeping the legacy behaviour and tests intact), then write the result into
`state["agent_results"][<node_name>]` and append any tool names to `state["tools_called"]`.
"""
