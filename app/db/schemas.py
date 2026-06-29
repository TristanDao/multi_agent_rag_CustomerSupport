"""Pydantic schemas for API requests/responses."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    customer_id: Optional[str] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None  # legacy alias, kept for back-compat


class SourceInfo(BaseModel):
    type: str
    name: str
    section: Optional[str] = None
    score: Optional[float] = None


class GuardrailInfo(BaseModel):
    input: str
    output: str
    reason: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    intent: str
    agents_called: List[str]
    tools_called: List[str] = []
    sources: List[SourceInfo] = []
    guardrail: GuardrailInfo
    request_id: str
    thread_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    latency_ms: int
    token_usage: Optional[Dict[str, int]] = None
    requires_human: bool = False
    debug: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    database: str
    vector_db: str
    llm: str


# --- Agent Registry & Graph introspection --------------------------


class AgentSpecOut(BaseModel):
    key: str
    name: str
    description: str
    capabilities: List[str]
    intents: List[str]
    tools: List[str]
    example_queries: List[str]
    input_schema: str
    output_schema: str
    has_node_fn: bool


class AgentListResponse(BaseModel):
    agents: List[AgentSpecOut]
    total: int
    terminal_nodes: List[str]


class GraphResponse(BaseModel):
    format: str = "mermaid"
    diagram: str


class CheckpointInfo(BaseModel):
    checkpoint_id: Optional[str] = None
    node: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    checkpoints: List[CheckpointInfo]
    total: int
