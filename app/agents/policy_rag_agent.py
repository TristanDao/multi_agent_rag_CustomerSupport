"""Policy RAG agent: retrieve relevant policy/FAQ chunks and summarize."""
import logging
from typing import Any, Dict, List

from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


POLICY_INTENTS = {
    "shipping_policy": ["shipping"],
    "payment_terms": ["payment"],
    "warranty_policy": ["warranty"],
    "return_refund": ["return", "refund"],
    "wholesale_pricing": ["wholesale"],
    "general_faq": [],
    "human_escalation": ["escalation"],
    "sales_recommendation": [],
}


def policy_rag_agent(intent: str, message: str) -> Dict[str, Any]:
    """Retrieve top-k chunks relevant to the question and current intent."""
    sources = POLICY_INTENTS.get(intent, [])
    queries: List[str] = [message]
    if intent in ("return_refund", "wholesale_pricing", "shipping_policy", "payment_terms", "warranty_policy"):
        # add intent-specific keywords to bias retrieval
        queries.append(" ".join([intent.replace("_", " "), *sources]))
    seen: set = set()
    chunks: List[Dict[str, Any]] = []
    for q in queries:
        for hit in retrieve(q):
            if hit["doc_id"] in seen:
                continue
            seen.add(hit["doc_id"])
            chunks.append(hit)
        if len(chunks) >= 6:
            break
    return {
        "agent": "policy_rag_agent",
        "intent": intent,
        "tools_called": ["retrieve"],
        "data": chunks[:6],
    }
