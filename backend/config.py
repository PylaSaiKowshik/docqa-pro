"""
config.py — Centralised application settings via pydantic-settings.
All values can be overridden by environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_model_name: str = "google/flan-t5-large"
    openai_api_key: str = ""          # Optional — enables OpenAI fallback & RAGAS

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 102          # ~20 % overlap

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 5
    rrf_k: int = 60                   # RRF fusion constant

    # ── Storage ───────────────────────────────────────────────────────────────
    faiss_index_path: str = "data/vectorstores/faiss_index"
    chroma_persist_path: str = "data/vectorstores/chroma"
    bm25_pickle_path: str = "data/vectorstores/bm25.pkl"
    upload_dir: str = "data/uploads"
    eval_results_path: str = "data/eval_results.json"

    # ── Backend ───────────────────────────────────────────────────────────────
    backend_url: str = "http://localhost:8000"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key)

    def ensure_dirs(self) -> None:
        """Create all required data directories if they don't exist."""
        for p in [
            self.faiss_index_path,
            self.chroma_persist_path,
            Path(self.bm25_pickle_path).parent,
            self.upload_dir,
            Path(self.eval_results_path).parent,
        ]:
            Path(p).mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
settings = Settings()
