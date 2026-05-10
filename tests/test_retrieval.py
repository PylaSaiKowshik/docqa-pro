"""tests/test_retrieval.py — Unit tests for BM25, FAISS, and Hybrid RRF retrieval."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from backend.retrieval.bm25_store import BM25Store, _tokenize
from backend.retrieval.hybrid import reciprocal_rank_fusion, _get_chunk_id


# ── BM25 Tests ────────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_basic(self):
        assert _tokenize("Hello World!") == ["hello", "world"]

    def test_numbers(self):
        tokens = _tokenize("GPT-4 is version 4.0")
        assert "gpt" in tokens or "gpt4" in tokens or "4" in tokens

    def test_empty(self):
        assert _tokenize("") == []


class TestBM25Store:
    @pytest.fixture
    def store(self):
        s = BM25Store()
        s._pickle_path = "data/vectorstores/test_bm25.pkl"
        return s

    def _doc(self, text: str, cid: str = "") -> Document:
        return Document(page_content=text, metadata={"chunk_id": cid or text[:10]})

    def test_add_and_search(self, store):
        docs = [
            self._doc("The quick brown fox jumps over the lazy dog", "d1"),
            self._doc("Python is a great programming language", "d2"),
            self._doc("Machine learning uses neural networks", "d3"),
        ]
        store.add_documents(docs)
        results = store.search("programming language Python", k=2)
        assert len(results) == 2
        top_doc, top_score = results[0]
        assert "python" in top_doc.page_content.lower() or top_score > 0

    def test_empty_store_returns_empty(self, store):
        results = store.search("anything", k=3)
        assert results == []

    def test_is_empty_flag(self, store):
        assert store.is_empty
        store.add_documents([self._doc("some text", "t1")])
        assert not store.is_empty

    def test_clear(self, store):
        store.add_documents([self._doc("text", "t1")])
        store.clear()
        assert store.is_empty

    def test_scores_are_ranked_descending(self, store):
        docs = [
            self._doc("cat sat on mat", "c1"),
            self._doc("dog barked loudly in the yard", "c2"),
            self._doc("cat chased the cat around the cat mat", "c3"),
        ]
        store.add_documents(docs)
        results = store.search("cat mat", k=3)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


# ── RRF Fusion Tests ──────────────────────────────────────────────────────────

class TestRRF:
    def _doc(self, cid: str) -> Document:
        return Document(page_content=f"content of {cid}", metadata={"chunk_id": cid})

    def test_single_list_preserves_rank(self):
        docs = [self._doc(f"d{i}") for i in range(5)]
        result = reciprocal_rank_fusion([docs])
        ids = [_get_chunk_id(d) for d, _ in result]
        # Highest rank should have highest RRF score
        assert ids[0] == "d0"

    def test_fusion_boosts_shared_docs(self):
        """A doc ranked high in both lists should beat a doc in only one."""
        # d1 is ranked #1 in both lists → should win fusion
        list_a = [self._doc("d1"), self._doc("d2"), self._doc("d3")]
        list_b = [self._doc("d1"), self._doc("d4"), self._doc("d5")]
        result = reciprocal_rank_fusion([list_a, list_b])
        top_id = _get_chunk_id(result[0][0])
        assert top_id == "d1"

    def test_deduplication(self):
        """A doc appearing in both lists should appear only once in output."""
        shared = self._doc("shared")
        list_a = [shared, self._doc("a1")]
        list_b = [shared, self._doc("b1")]
        result = reciprocal_rank_fusion([list_a, list_b])
        ids = [_get_chunk_id(d) for d, _ in result]
        assert ids.count("shared") == 1

    def test_empty_lists(self):
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_scores_positive(self):
        docs = [self._doc(f"d{i}") for i in range(3)]
        result = reciprocal_rank_fusion([docs])
        for _, score in result:
            assert score > 0

    def test_rrf_k_parameter(self):
        """Higher k → scores closer together (less rank-sensitive)."""
        docs = [self._doc(f"d{i}") for i in range(3)]
        result_small_k = reciprocal_rank_fusion([docs], k=1)
        result_large_k = reciprocal_rank_fusion([docs], k=1000)
        # With k=1000 the gap between ranks is smaller
        gap_small = result_small_k[0][1] - result_small_k[-1][1]
        gap_large = result_large_k[0][1] - result_large_k[-1][1]
        assert gap_small > gap_large
