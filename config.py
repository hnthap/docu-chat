"""
Centralized configuration management for the RAG application.

This module uses Pydantic's BaseSettings to load and validate configuration
from environment variables and .env files.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Defines the application's configuration settings."""
    # RAG Core Settings
    LLM: str = 'gemini-1.5-flash'
    LLM_PROVIDER: str = 'google_genai'
    LLM_API_KEY_ENV_KEY: str = 'GOOGLE_API_KEY'
    HF_EMBEDDINGS_MODEL_NAME: str = 'sentence-transformers/all-mpnet-base-v2'
    MAX_RETRIEVED_DOCUMENTS: int = 10
    CONTEXT_MAX_CHARS: int = 15_000
    HISTORY_MAX_SIZE: int = 5

    # Indexing Settings
    CORPUS_DIR_PATH: str = 'data'
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    INDEXING_BATCH_SIZE: int = 1000

    # PGVector Settings (loaded from environment files)
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    TABLE_NAME: str
    VECTOR_SIZE: int = 768  # Corresponds to all-mpnet-base-v2

    model_config = SettingsConfigDict(
        env_file=(
            '.env',
            Path('pgvector') / '.env'
        ),
        env_file_encoding='utf-8',
        extra='allow'
    )

settings = Settings()
