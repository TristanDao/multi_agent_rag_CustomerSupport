"""Helper to wrap the legacy agent functions into LangGraph node functions.

The existing `app.agents.*` modules return dicts in the shape:
    {"agent": "<key>", "intent": ..., "tools_called": [...], "data": [...], ...}

These helpers bridge that shape into the `SupportState` patch that each node returns.

State accumulation is handled by LangGraph reducers declared in `app.agents.state`:
* `agent_results` is shallow-merged, so the node just needs to return its own entry
* `tools_called` and `nodes_visited` are append-only via the `add_str_list` reducer
* `nodes_visited` is append-only via the same reducer
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List


def make_node(node_name: str, agent_fn: Callable[..., Dict[str, Any]]) -> Callable:
    """Return a LangGraph-compatible node function for the given legacy agent.

    The returned coroutine reads `intent`, `entities`, `user_message` from state and
    returns a partial update with:
        * `agent_results` → {node_name: <legacy agent dict>}
        * `tools_called`  → list of tool names that the agent invoked
        * `nodes_visited` → [node_name]

    Reducers in `SupportState` take care of appending/merging into the running state.
    """

    async def _node(state):
        intent = state.get("intent", "")
        entities = state.get("entities") or {}
        user_message = state.get("user_message", "")
        result = agent_fn(intent=intent, entities=entities, message=user_message)
        if hasattr(result, "__await__"):
            result = await result
        return {
            "agent_results": {node_name: result},
            "tools_called": list(result.get("tools_called") or []),
            "nodes_visited": [node_name],
        }

    _node.__name__ = f"{node_name}_node"
    return _node
