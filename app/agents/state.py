"""LangGraph shared state schema for the orchestrator.

Every node in the LangGraph StateGraph receives this state and returns a partial
update. Fields fall into three groups:

* Inputs (set by the entrypoint, never mutated by nodes): `messages`, `customer_id`,
  `request_id`, `thread_id`, `user_message`.
* Routing: `intent`, `entities`, `next_node`, `requires_human`.
* Accumulated outputs: `agent_results`, `sources`, `final_answer`, `tools_called`,
  `nodes_visited`, `latency_ms`, `token_usage`, `guardrail`, `pii_redacted`.

Nodes must be pure functions of state: they read fields and return only the fields
they want to update. The Postgres checkpointer is the only source of truth for
cross-request persistence.

Reducers:
* `add_messages`     — append-only list of BaseMessage
* `add_str_list`     — append-only list of strings (tools_called, nodes_visited)
* `add_dict`         — shallow merge into a dict (agent_results)
* `sum_int`          — accumulate ints (latency_ms, token_usage sub-fields)
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# --- Reducers ------------------------------------------------------


def add_str_list(existing: Optional[List[str]], new: Optional[List[str]]) -> List[str]:
    """Append-only reducer for lists of strings (e.g. tools_called, nodes_visited)."""
    if not existing:
        existing = []
    if not new:
        return existing
    return list(existing) + list(new)


def add_dict(existing: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Shallow-merge reducer for dicts (e.g. agent_results)."""
    merged: Dict[str, Any] = dict(existing or {})
    if new:
        merged.update(new)
    return merged


def sum_int(existing: Optional[int], new: Optional[int]) -> int:
    return int(existing or 0) + int(new or 0)


def add_token_usage(
    existing: Optional[Dict[str, int]],
    new: Optional[Dict[str, int]],
) -> Dict[str, int]:
    out: Dict[str, int] = dict(existing or {"input_tokens": 0, "output_tokens": 0})
    if new:
        out["input_tokens"] = out.get("input_tokens", 0) + int(new.get("input_tokens", 0) or 0)
        out["output_tokens"] = out.get("output_tokens", 0) + int(new.get("output_tokens", 0) or 0)
    return out


# --- State ---------------------------------------------------------


class SupportState(TypedDict, total=False):
    # ---- Inputs -----------------------------------------------------------
    messages: Annotated[List[BaseMessage], add_messages]
    user_message: str
    customer_id: Optional[str]
    request_id: str
    thread_id: str

    # ---- Routing ----------------------------------------------------------
    intent: str
    entities: Dict[str, Any]
    next_node: Optional[str]  # name of the next node chosen by route_intent
    requires_human: bool

    # ---- Accumulated outputs (use reducers so each node appends, not overwrites) ----
    agent_results: Annotated[Dict[str, Any], add_dict]
    sources: Annotated[List[Dict[str, Any]], add_str_list]
    tools_called: Annotated[List[str], add_str_list]
    nodes_visited: Annotated[List[str], add_str_list]
    final_answer: Optional[str]

    # ---- Telemetry / observability --------------------------------------
    latency_ms: Annotated[int, sum_int]
    token_usage: Annotated[Dict[str, int], add_token_usage]
    guardrail: Annotated[Dict[str, Any], add_dict]
    pii_redacted: bool

    # ---- Control flags ---------------------------------------------------
    blocked: bool  # True if input guardrail blocked the request
    block_reason: Optional[str]
    block_safe_response: Optional[str]


def empty_state(
    user_message: str,
    request_id: str,
    thread_id: str,
    customer_id: Optional[str] = None,
) -> SupportState:
    """Construct a fresh SupportState for a new turn."""
    return SupportState(
        messages=[],
        user_message=user_message,
        customer_id=customer_id,
        request_id=request_id,
        thread_id=thread_id,
        intent="",
        entities={},
        next_node=None,
        requires_human=False,
        agent_results={},
        sources=[],
        tools_called=[],
        nodes_visited=[],
        final_answer=None,
        latency_ms=0,
        token_usage={"input_tokens": 0, "output_tokens": 0},
        guardrail={"input": "passed", "output": "passed", "reason": None},
        pii_redacted=False,
        blocked=False,
        block_reason=None,
        block_safe_response=None,
    )
