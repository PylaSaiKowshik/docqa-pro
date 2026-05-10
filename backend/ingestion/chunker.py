"""
ingestion/chunker.py — Sliding-window semantic chunker.

Uses LangChain's RecursiveCharacterTextSplitter with:
  - chunk_size   = configurable (default 512 tokens ≈ chars)
  - chunk_overlap = ~20 % of chunk_size
  - Semantic boundary separators: paragraph > newline > sentence > word
  - Full source metadata preservation on every chunk
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings

logger = logging.getLogger(__name__)

# Semantic boundary order: prefer splitting at paragraph breaks first
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Return a configured RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        add_start_index=True,   # records char offset in metadata
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Split a list of Documents into overlapping chunks.

    Each output chunk carries:
      - doc.metadata  (source, page, doc_type, …)
      - chunk_index   (position within the original document's chunks)
      - start_index   (character offset within original page content)
      - chunk_id      (unique identifier: "<source>::<chunk_index>")

    Args:
        documents:      Raw documents from the loader.
        chunk_size:     Override global chunk size.
        chunk_overlap:  Override global overlap.

    Returns:
        Flat list of chunk Documents with enriched metadata.
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: List[Document] = []

    for doc in documents:
        splits = splitter.split_documents([doc])
        for idx, chunk in enumerate(splits):
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = f"{source}::{idx}"
            chunk.metadata["total_chunks_in_doc"] = len(splits)
            chunks.append(chunk)

    logger.info(
        "Chunked %d document(s) → %d chunks "
        "(size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size or settings.chunk_size,
        chunk_overlap or settings.chunk_overlap,
    )
    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """Return descriptive statistics about a list of chunks."""
    if not chunks:
        return {"count": 0}
    lengths = [len(c.page_content) for c in chunks]
    return {
        "count": len(chunks),
        "avg_length": round(sum(lengths) / len(lengths), 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_chars": sum(lengths),
    }
