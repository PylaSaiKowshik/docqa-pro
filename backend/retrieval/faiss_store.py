"""
retrieval/faiss_store.py — Dense FAISS vector store.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) with FAISS IndexFlatIP
(cosine similarity via normalised inner product).

Supports:
  - add_documents() — index new chunks
  - similarity_search_with_score() — top-K retrieval
  - save() / load() — disk persistence
  - clear() — wipe index
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from backend.config import settings

logger = logging.getLogger(__name__)


class FAISSStore:
    """Thin wrapper around LangChain FAISS with persistence helpers."""

    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # cosine via IP
        )
        self._store: FAISS | None = None
        self._index_path = settings.faiss_index_path

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Add (or extend) the FAISS index with new chunks."""
        if not documents:
            logger.warning("add_documents called with empty list — skipping")
            return

        if self._store is None:
            logger.info("Building FAISS index from %d chunks…", len(documents))
            self._store = FAISS.from_documents(documents, self._embeddings)
        else:
            logger.info("Extending FAISS index with %d chunks…", len(documents))
            self._store.add_documents(documents)

        self.save()

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Document]:
        """Return top-K documents without scores."""
        results = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """Return list of (Document, cosine_score) tuples."""
        self._require_store()
        k = k or settings.top_k
        return self._store.similarity_search_with_score(query, k=k)  # type: ignore[union-attr]

    def save(self) -> None:
        """Persist FAISS index to disk."""
        if self._store is None:
            return
        os.makedirs(self._index_path, exist_ok=True)
        self._store.save_local(self._index_path)
        logger.info("FAISS index saved → %s", self._index_path)

    def load(self) -> bool:
        """
        Load FAISS index from disk.
        Returns True if loaded successfully, False if no index exists.
        """
        index_file = os.path.join(self._index_path, "index.faiss")
        if not os.path.exists(index_file):
            logger.info("No existing FAISS index found at %s", self._index_path)
            return False
        self._store = FAISS.load_local(
            self._index_path,
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded from %s", self._index_path)
        return True

    def clear(self) -> None:
        """Discard in-memory index (does not delete files)."""
        self._store = None
        logger.info("FAISS in-memory store cleared")

    @property
    def is_empty(self) -> bool:
        return self._store is None

    def as_retriever(self, k: int | None = None):
        """Return a LangChain retriever interface for LCEL chains."""
        self._require_store()
        return self._store.as_retriever(  # type: ignore[union-attr]
            search_type="similarity",
            search_kwargs={"k": k or settings.top_k},
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _require_store(self) -> None:
        if self._store is None:
            raise RuntimeError(
                "FAISS store is empty. Call add_documents() or load() first."
            )


# Module-level singleton
faiss_store = FAISSStore()
