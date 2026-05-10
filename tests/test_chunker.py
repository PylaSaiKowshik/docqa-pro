"""tests/test_chunker.py — Unit tests for the sliding-window chunker."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from backend.ingestion.chunker import chunk_documents, get_chunk_stats, build_splitter


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_doc(text: str, source: str = "test.pdf", page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


def _long_text(n_words: int = 300) -> str:
    return " ".join(["word"] * n_words)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildSplitter:
    def test_defaults(self):
        splitter = build_splitter()
        assert splitter._chunk_size > 0
        assert splitter._chunk_overlap > 0

    def test_custom_params(self):
        splitter = build_splitter(chunk_size=100, chunk_overlap=20)
        assert splitter._chunk_size == 100
        assert splitter._chunk_overlap == 20


class TestChunkDocuments:
    def test_short_doc_single_chunk(self):
        doc = _make_doc("This is a short document.")
        chunks = chunk_documents([doc])
        assert len(chunks) >= 1

    def test_long_doc_multiple_chunks(self):
        doc = _make_doc(_long_text(500))
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=40)
        assert len(chunks) > 1

    def test_metadata_preserved(self):
        doc = _make_doc("Some text.", source="my_doc.pdf", page=3)
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "my_doc.pdf"
            assert chunk.metadata["page"] == 3

    def test_chunk_id_assigned(self):
        doc = _make_doc("Hello world " * 50)
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "::" in chunk.metadata["chunk_id"]

    def test_chunk_index_sequential(self):
        doc = _make_doc(_long_text(400))
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=40)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_input(self):
        result = chunk_documents([])
        assert result == []

    def test_multiple_documents(self):
        docs = [_make_doc(f"Document {i} " * 10, source=f"doc{i}.pdf") for i in range(3)]
        chunks = chunk_documents(docs)
        sources = {c.metadata["source"] for c in chunks}
        assert sources == {"doc0.pdf", "doc1.pdf", "doc2.pdf"}

    def test_overlap_creates_shared_content(self):
        """Chunks from a long doc should share some words at boundaries."""
        text = " ".join([f"word{i}" for i in range(200)])
        doc = _make_doc(text)
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=30)
        if len(chunks) >= 2:
            # The end of chunk[0] and start of chunk[1] should share tokens
            end_of_first = chunks[0].page_content[-50:]
            start_of_second = chunks[1].page_content[:50]
            # There should be some overlap (shared words)
            words_first = set(end_of_first.split())
            words_second = set(start_of_second.split())
            assert len(words_first & words_second) > 0


class TestGetChunkStats:
    def test_empty(self):
        stats = get_chunk_stats([])
        assert stats["count"] == 0

    def test_basic_stats(self):
        chunks = [
            Document(page_content="hello world", metadata={}),
            Document(page_content="foo bar baz qux", metadata={}),
        ]
        stats = get_chunk_stats(chunks)
        assert stats["count"] == 2
        assert stats["min_length"] == len("hello world")
        assert stats["max_length"] == len("foo bar baz qux")
        assert stats["total_chars"] == len("hello world") + len("foo bar baz qux")
