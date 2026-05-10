"""
retrieval/bm25_store.py — BM25 sparse retriever.

Uses rank_bm25 (BM25Okapi) with simple whitespace tokenisation.
The corpus and tokenised texts are serialised to disk with pickle
so the index survives server restarts without rebuilding.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from backend.config import settings

logger = logging.getLogger(__name__)

_TOKENIZE_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    """Lowercase + whitespace tokeniser (no stopword removal for now)."""
    return _TOKENIZE_RE.findall(text.lower())


class BM25Store:
    """In-memory BM25 index with disk persistence via pickle."""

    def __init__(self) -> None:
        self._documents: List[Document] = []
        self._tokenized: List[List[str]] = []
        self._bm25: BM25Okapi | None = None
        self._pickle_path = settings.bm25_pickle_path

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Append documents and rebuild BM25 index."""
        if not documents:
            return
        self._documents.extend(documents)
        self._tokenized.extend([_tokenize(d.page_content) for d in documents])
        self._rebuild()
        self.save()

    def search(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Return top-K (Document, bm25_score) pairs.
        Scores are raw BM25 values (not normalised).
        """
        if self._bm25 is None or not self._documents:
            return []
        k = k or settings.top_k
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair each doc with its score and sort descending
        ranked = sorted(
            zip(self._documents, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]

    def save(self) -> None:
        """Serialise index state to disk."""
        os.makedirs(os.path.dirname(self._pickle_path), exist_ok=True)
        payload = {
            "documents": self._documents,
            "tokenized": self._tokenized,
        }
        with open(self._pickle_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 index saved → %s (%d docs)", self._pickle_path, len(self._documents))

    def load(self) -> bool:
        """
        Load index from disk.
        Returns True on success, False if no saved index found.
        """
        if not os.path.exists(self._pickle_path):
            logger.info("No existing BM25 index at %s", self._pickle_path)
            return False
        with open(self._pickle_path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301
        self._documents = payload["documents"]
        self._tokenized = payload["tokenized"]
        self._rebuild()
        logger.info("BM25 index loaded — %d documents", len(self._documents))
        return True

    def clear(self) -> None:
        self._documents = []
        self._tokenized = []
        self._bm25 = None
        logger.info("BM25 store cleared")

    @property
    def is_empty(self) -> bool:
        return self._bm25 is None or len(self._documents) == 0

    # ── Private ───────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        """Rebuild BM25Okapi from current tokenised corpus."""
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
            logger.debug("BM25 index rebuilt with %d docs", len(self._tokenized))


# Module-level singleton
bm25_store = BM25Store()
