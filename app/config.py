"""Application configuration."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    APP_NAME: str = "multi_agent_rag_retail"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    ALIBABA_API_KEY: str = ""
    ALIBABA_URL: str = "https://ws-v9y2oinbtzzm4ey9.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    ALIBABA_LLM_MODEL: str = "qwen-mt-flash"
    ALIBABA_LLM_BACKUP: str = "qwen3.5-flash"

    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM: int = 384

    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "retail"
    POSTGRES_PASSWORD: str = "retail_password"
    POSTGRES_DB: str = "retail_db"
    DATABASE_URL: str = "postgresql+psycopg2://retail:retail_password@postgres:5432/retail_db"

    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "policy_docs"

    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://langfuse:3000"
    LANGFUSE_ENABLED: bool = False

    ENABLE_INPUT_GUARDRAIL: bool = True
    ENABLE_OUTPUT_GUARDRAIL: bool = True
    ENABLE_PII_REDACTION: bool = True

    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    TOP_K_RETRIEVAL: int = 4

    @property
    def has_llm_key(self) -> bool:
        return bool(self.ALIBABA_API_KEY)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
