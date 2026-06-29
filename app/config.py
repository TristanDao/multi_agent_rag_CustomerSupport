"""Application configuration.

LLM provider: Alibaba DashScope is the sole LLM provider. We access it via the
OpenAI Python SDK pointed at the Alibaba OpenAI-compatible endpoint through
`base_url`. There is no `OPENAI_API_KEY` anywhere — the SDK is purely a
transport, Alibaba is the model.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    APP_NAME: str = "multi_agent_rag_retail"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # --- Alibaba (sole LLM + embedding + rerank provider) -----------------
    # The OpenAI Python SDK is used as HTTP transport via these settings.
    ALIBABA_API_KEY: str = ""
    ALIBABA_URL: str = "https://ws-v9y2oinbtzzm4ey9.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    ALIBABA_LLM_MODEL: str = "qwen-mt-flash"
    ALIBABA_LLM_BACKUP: str = "qwen3.5-flash"
    # DashScope has a separate, non-OpenAI-compatible endpoint for rerank.
    ALIBABA_RERANK_URL: str = "https://dashscope-intl.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    # --- Embedding (RAG) ---------------------------------------------------
    # Primary: Alibaba text-embedding-v3 via OpenAI-compatible /embeddings.
    # Fallback: sentence-transformers if EMBEDDING_PROVIDER=sentence_transformers
    #   or if the Alibaba call fails.
    EMBEDDING_PROVIDER: str = "alibaba"           # "alibaba" | "sentence_transformers"
    EMBEDDING_MODEL: str = "text-embedding-v3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_FALLBACK_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_FALLBACK_DIM: int = 384

    # --- Rerank (RAG) ------------------------------------------------------
    # Primary: Alibaba gte-rerank (separate REST endpoint).
    # Fallback: skip rerank, return merged candidates by hybrid score.
    RERANK_PROVIDER: str = "alibaba"             # "alibaba" | "none"
    RERANK_MODEL: str = "gte-rerank"
    RERANK_TOP_N: int = 20                       # candidates to rerank

    # --- Hybrid retrieval weights -----------------------------------------
    HYBRID_SPARSE_WEIGHT: float = 0.4
    HYBRID_DENSE_WEIGHT: float = 0.6
    HYBRID_TOP_N: int = 20                       # candidates per leg (BM25 + dense)
    RERANK_FINAL_K: int = 5                      # final top-K after rerank
    TOP_K_RETRIEVAL: int = 5                     # legacy alias, used by some callers

    # --- PostgreSQL (app data + LangGraph checkpointer) -------------------
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "retail"
    POSTGRES_PASSWORD: str = "retail_password"
    POSTGRES_DB: str = "retail_db"
    DATABASE_URL: str = "postgresql+psycopg2://retail:retail_password@postgres:5432/retail_db"

    # --- Qdrant (vector store) --------------------------------------------
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "policy_docs"

    # --- Langfuse (LLM observability, required) ---------------------------
    LANGFUSE_ENABLED: bool = True
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://langfuse:3000"
    LANGFUSE_NEXTAUTH_SECRET: str = "please-change-me"
    LANGFUSE_USER_EMAIL: str = "admin@example.com"
    LANGFUSE_USER_PASSWORD: str = "admin_password"

    # --- Guardrails -------------------------------------------------------
    ENABLE_INPUT_GUARDRAIL: bool = True
    ENABLE_OUTPUT_GUARDRAIL: bool = True
    ENABLE_PII_REDACTION: bool = True

    # --- Chunking ---------------------------------------------------------
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100

    # --- LangGraph checkpointer -------------------------------------------
    CHECKPOINT_BACKEND: str = "memory"            # "memory" | "postgres"
    CHECKPOINT_TTL_DAYS: int = 30

    @property
    def has_alibaba_key(self) -> bool:
        return bool(self.ALIBABA_API_KEY)

    @property
    def has_llm_key(self) -> bool:
        return self.has_alibaba_key

    def llm_base_url(self) -> str:
        return self.ALIBABA_URL

    def llm_api_key(self) -> str:
        return self.ALIBABA_API_KEY

    def llm_model(self) -> str:
        return self.ALIBABA_LLM_MODEL


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
