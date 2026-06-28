"""Pydantic schemas for API requests/responses."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    customer_id: Optional[str] = None
    session_id: Optional[str] = None


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
    latency_ms: int
    token_usage: Optional[Dict[str, int]] = None
    requires_human: bool = False
    debug: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    database: str
    vector_db: str
    llm: str
