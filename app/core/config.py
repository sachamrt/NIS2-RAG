"""Application settings, loaded from the environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["local", "openai", "anthropic", "mistral"]
EmbeddingProvider = Literal["local", "openai", "mistral"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- provider selection -------------------------------------------------
    llm_provider: LLMProvider = "mistral"
    embedding_provider: EmbeddingProvider = "mistral"

    # --- credentials --------------------------------------------------------
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None

    # --- per-provider models ------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"

    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    anthropic_llm_model: str = "claude-sonnet-5"

    mistral_llm_model: str = "ministral-8b-latest"
    mistral_embedding_model: str = "mistral-embed"

    # --- generation defaults (provider-independent) -------------------------
    llm_temperature: float = 0.0
    llm_max_tokens: int | None = None
    llm_timeout: int = 60
    llm_max_retries: int = 3

    # --- retrieval ----------------------------------------------------------
    retrieval_k: int = 4

    # --- api ----------------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # --- vector store -------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "nis2"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so settings are parsed once per process."""
    return Settings()
