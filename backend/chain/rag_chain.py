"""
chain/rag_chain.py — LCEL-based RAG chain with Flan-T5.

Architecture:
  {context: retriever | format_docs, question: passthrough}
  → PromptTemplate
  → HuggingFacePipeline (Flan-T5-Large)
  → StrOutputParser

The chain also exposes run_with_sources() which returns both the
answer string and the source chunks used for attribution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from backend.config import settings
from backend.retrieval.hybrid import hybrid_search_docs

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """You are a precise question-answering assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this question."
Do not make up information. Keep your answer concise and factual.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_PROMPT_TEMPLATE,
)

# ── Lazy-loaded pipeline ──────────────────────────────────────────────────────
_pipeline_cache: Any = None


def _get_llm_pipeline():
    """Load Flan-T5 pipeline (cached after first call)."""
    global _pipeline_cache  # noqa: PLW0603
    if _pipeline_cache is not None:
        return _pipeline_cache

    logger.info("Loading LLM: %s (this may take a moment…)", settings.llm_model_name)

    if settings.use_openai:
        # OpenAI fallback (better quality)
        try:
            from langchain_openai import ChatOpenAI
            _pipeline_cache = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=settings.openai_api_key,
            )
            logger.info("Using OpenAI GPT-3.5-turbo as LLM backend")
            return _pipeline_cache
        except Exception as exc:
            logger.warning("OpenAI init failed (%s) — falling back to Flan-T5", exc)

    # Local Flan-T5 via HuggingFace
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    from langchain_community.llms import HuggingFacePipeline

    tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.llm_model_name)

    hf_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
    )
    _pipeline_cache = HuggingFacePipeline(pipeline=hf_pipe)
    logger.info("Flan-T5 pipeline ready")
    return _pipeline_cache


# ── Formatting helper ─────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    """Concatenate chunk contents with source markers for the prompt."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        loc = f"{source}" + (f" (p.{page})" if page != "" else "")
        parts.append(f"[{i}] {loc}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Main chain factory ────────────────────────────────────────────────────────

def build_rag_chain(retriever=None):
    """
    Build and return an LCEL RAG chain.

    Args:
        retriever: Optional LangChain retriever. Defaults to hybrid_search_docs
                   wrapped as a Runnable.
    """
    llm = _get_llm_pipeline()

    if retriever is None:
        retriever = RunnableLambda(
            lambda q: hybrid_search_docs(q, k=settings.top_k)
        )

    chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ── Convenience function with source attribution ──────────────────────────────

def run_with_sources(
    question: str,
    k: int | None = None,
) -> Dict[str, Any]:
    """
    Run the RAG chain and return answer + source chunks.

    Returns:
        {
            "answer": str,
            "sources": [{"content": str, "source": str, "page": ..., "chunk_id": str}],
            "question": str,
        }
    """
    from backend.retrieval.hybrid import hybrid_search

    k = k or settings.top_k
    results = hybrid_search(question, k=k)
    docs = [doc for doc, _ in results]

    # Build chain with fixed docs (avoids double retrieval)
    llm = _get_llm_pipeline()
    context_str = _format_docs(docs)

    prompt_text = PROMPT.format(context=context_str, question=question)
    raw_answer = llm.invoke(prompt_text)

    # Handle both string and AIMessage responses
    if hasattr(raw_answer, "content"):
        answer = raw_answer.content
    else:
        answer = str(raw_answer)

    sources = [
        {
            "content": doc.page_content[:400],
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", ""),
            "chunk_id": doc.metadata.get("chunk_id", ""),
        }
        for doc in docs
    ]

    return {"answer": answer, "sources": sources, "question": question}
