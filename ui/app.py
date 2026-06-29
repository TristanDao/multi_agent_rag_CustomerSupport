"""Streamlit chat UI for manual agent testing.

The UI always talks to the FastAPI backend over HTTP (`BACKEND_URL` env var,
default `http://localhost:8000`). It is intentionally thin: it sends messages
to `/chat` and renders the response (answer + intent + agents + sources).

A session-scoped `thread_id` is held in `st.session_state` so multi-turn
conversations can be exercised end-to-end (the LangGraph checkpointer stores
state on the backend).
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Multi-Agent RAG Test Console",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- session state ------------------------------------------------------


def _new_thread_id() -> str:
    return f"thread-{uuid.uuid4().hex}"


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = _new_thread_id()
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of {role, content, meta?}


# --- helpers ------------------------------------------------------------


def _post_chat(message: str, thread_id: str, customer_id: str | None = None) -> Dict[str, Any]:
    """POST /chat. Returns the parsed JSON body, or an error dict."""
    payload: Dict[str, Any] = {"message": message, "thread_id": thread_id}
    if customer_id:
        payload["customer_id"] = customer_id
    try:
        r = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {
            "answer": f"Backend unreachable: {e}",
            "intent": "error",
            "agents_called": [],
            "tools_called": [],
            "sources": [],
            "guardrail": {"input": "error", "output": "error", "reason": str(e)},
            "thread_id": thread_id,
            "checkpoint_id": None,
            "latency_ms": 0,
            "token_usage": None,
            "requires_human": False,
        }


def _backend_health() -> Dict[str, Any]:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "down", "error": str(e)}


def _list_agents() -> list:
    try:
        r = requests.get(f"{BACKEND_URL}/admin/agents", timeout=5)
        r.raise_for_status()
        return r.json().get("agents", [])
    except Exception:
        return []


# --- sidebar ------------------------------------------------------------


with st.sidebar:
    st.markdown("## Agent Test Console")
    st.caption("Multi-Agent RAG · LangGraph + Langfuse + Streamlit")

    health = _backend_health()
    healthy = health.get("status") == "ok"
    st.markdown(
        f"**Backend** `{'OK' if healthy else 'DOWN'}`  ·  `{BACKEND_URL}`"
    )

    st.text_input(
        "Thread ID",
        value=st.session_state["thread_id"],
        key="_thread_id_display",
        disabled=True,
        help="Conversation ID sent on every /chat call. Use the same value across turns to keep state.",
    )
    if st.button("New conversation", use_container_width=True):
        st.session_state["thread_id"] = _new_thread_id()
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.markdown("**Customer ID** (optional)")
    customer_id = st.text_input(
        "customer_id",
        value="",
        key="_customer_id",
        label_visibility="collapsed",
        placeholder="e.g. C00001",
    )

    st.divider()
    st.markdown("**Registered agents**")
    agents = _list_agents()
    if agents:
        for a in agents:
            with st.expander(f"`{a['key']}` — {a['name']}"):
                st.write(a.get("description", ""))
                st.caption("Intents: " + ", ".join(a.get("intents", [])))
                st.caption("Tools: " + ", ".join(a.get("tools", [])))
    else:
        st.caption("(unavailable)")

    st.divider()
    st.markdown("**Other UIs**")
    st.markdown("- [FastAPI Swagger](http://localhost:8000/docs)")
    st.markdown("- [Langfuse](http://localhost:3000)")
    st.markdown("- [Qdrant](http://localhost:6333/dashboard)")


# --- main pane ----------------------------------------------------------


st.markdown("# Chat")
st.caption("Ask the multi-agent retail assistant. Multi-turn context is preserved via `thread_id`.")

# message history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        meta = m.get("meta")
        if meta and m["role"] == "assistant":
            with st.expander("Response details"):
                # Two columns: routing on the left, sources on the right
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Routing**")
                    st.json(
                        {
                            "intent": meta.get("intent"),
                            "thread_id": meta.get("thread_id"),
                            "checkpoint_id": meta.get("checkpoint_id"),
                            "agents_called": meta.get("agents_called", []),
                            "tools_called": meta.get("tools_called", []),
                            "latency_ms": meta.get("latency_ms"),
                            "token_usage": meta.get("token_usage"),
                            "requires_human": meta.get("requires_human"),
                            "guardrail": meta.get("guardrail"),
                            "request_id": meta.get("request_id"),
                        }
                    )
                with col2:
                    st.markdown("**Sources**")
                    sources = meta.get("sources", []) or []
                    if sources:
                        for s in sources:
                            st.markdown(
                                f"- `{s.get('type', '?')}` · **{s.get('name', '')}**"
                                + (f" — `{s.get('section')}`" if s.get("section") else "")
                                + (f" · score `{s.get('score')}`" if s.get("score") is not None else "")
                            )
                    else:
                        st.caption("(no sources)")

# chat input
prompt = st.chat_input("Ask anything (e.g. `Tìm giấy A4 giá dưới 400k còn hàng ở HCM`)")
if prompt:
    cid = customer_id or None
    with st.spinner("Calling the orchestrator..."):
        body = _post_chat(prompt, st.session_state["thread_id"], customer_id=cid)
    # If the server returned a new thread_id (e.g. first turn, no client value),
    # keep the sidebar in sync.
    new_tid = body.get("thread_id")
    if new_tid and new_tid != st.session_state["thread_id"]:
        st.session_state["thread_id"] = new_tid
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["messages"].append(
        {"role": "assistant", "content": body.get("answer", ""), "meta": body}
    )
    st.rerun()
