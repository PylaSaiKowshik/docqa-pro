"""tests/test_chain.py — Unit tests for the LCEL RAG chain helpers."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from backend.chain.rag_chain import _format_docs, PROMPT


# ── Format docs tests ─────────────────────────────────────────────────────────

class TestFormatDocs:
    def _doc(self, text: str, source: str = "test.pdf", page=1) -> Document:
        return Document(page_content=text, metadata={"source": source, "page": page})

    def test_single_doc(self):
        doc = self._doc("Some content here.", "report.pdf", 2)
        result = _format_docs([doc])
        assert "Some content here." in result
        assert "report.pdf" in result
        assert "p.2" in result

    def test_multiple_docs_separated(self):
        docs = [self._doc(f"Content {i}", f"doc{i}.pdf") for i in range(3)]
        result = _format_docs(docs)
        assert result.count("---") == 2   # 3 docs → 2 separators
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_empty_list(self):
        assert _format_docs([]) == ""

    def test_no_page_number(self):
        doc = Document(page_content="text", metadata={"source": "web.html"})
        result = _format_docs([doc])
        assert "p." not in result
        assert "web.html" in result


# ── Prompt template tests ─────────────────────────────────────────────────────

class TestPromptTemplate:
    def test_prompt_has_required_variables(self):
        assert "context" in PROMPT.input_variables
        assert "question" in PROMPT.input_variables

    def test_prompt_formats_correctly(self):
        rendered = PROMPT.format(context="Some context.", question="What is it?")
        assert "Some context." in rendered
        assert "What is it?" in rendered
        assert "Answer:" in rendered

    def test_grounding_instruction_present(self):
        rendered = PROMPT.format(context="ctx", question="q")
        assert "only" in rendered.lower() or "context" in rendered.lower()


# ── run_with_sources mock test ────────────────────────────────────────────────

class TestRunWithSources:
    def test_structure_of_output(self):
        """Mock the LLM + retriever to verify the output schema."""
        mock_doc = Document(
            page_content="RAG combines retrieval with generation.",
            metadata={"source": "paper.pdf", "page": 1, "chunk_id": "paper.pdf::0"},
        )

        with patch("backend.chain.rag_chain._get_llm_pipeline") as mock_llm_fn, \
             patch("backend.retrieval.hybrid.hybrid_search") as mock_hybrid:

            mock_hybrid.return_value = [(mock_doc, 0.9)]
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "RAG stands for Retrieval-Augmented Generation."
            mock_llm_fn.return_value = mock_llm

            from backend.chain.rag_chain import run_with_sources
            result = run_with_sources("What is RAG?")

        assert "answer" in result
        assert "sources" in result
        assert "question" in result
        assert isinstance(result["sources"], list)
        assert result["sources"][0]["source"] == "paper.pdf"
