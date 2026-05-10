"""
main.py — FastAPI application for DocQA-Pro.

Endpoints:
  GET  /health          — Health check + system status
  POST /ingest/file     — Upload file (PDF/DOCX)
  POST /ingest/url      — Index a URL
  POST /query           — Ask a question, get answer + sources
  POST /evaluate        — Run RAGAS evaluation
  POST /evaluate/build  — Auto-build test data then evaluate
  GET  /evaluate/history— Eval run history
  GET  /sources         — List all indexed documents
  DELETE /clear         — Wipe all indexes
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import settings
from backend.ingestion.loader import load_document
from backend.ingestion.chunker import chunk_documents, get_chunk_stats
from backend.retrieval.faiss_store import faiss_store
from backend.retrieval.bm25_store import bm25_store
from backend.retrieval.hybrid import get_retrieval_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocQA-Pro API",
    description="Hybrid RAG Q&A system with FAISS + BM25 + RRF fusion",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_indexed_sources: List[Dict] = []


@app.on_event("startup")
async def startup() -> None:
    settings.ensure_dirs()
    faiss_store.load()
    bm25_store.load()
    logger.info("DocQA-Pro startup complete.")


# ── Schemas ───────────────────────────────────────────────────────────────────

class URLIngestRequest(BaseModel):
    url: str
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = None
    include_comparison: bool = False


class QAPair(BaseModel):
    question: str
    ground_truth: str


class EvaluateRequest(BaseModel):
    test_data: List[Dict[str, Any]]
    run_name: Optional[str] = None


class EvaluateBuildRequest(BaseModel):
    qa_pairs: List[QAPair]
    run_name: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> Dict:
    return {
        "status": "ok",
        "faiss_ready": not faiss_store.is_empty,
        "bm25_ready": not bm25_store.is_empty,
        "indexed_sources": len(_indexed_sources),
        "llm_model": settings.llm_model_name,
        "openai_fallback": settings.use_openai,
    }


@app.get("/sources", tags=["Documents"])
async def list_sources() -> Dict:
    return {"sources": _indexed_sources, "count": len(_indexed_sources)}


@app.post("/ingest/file", tags=["Documents"])
async def ingest_file(file: UploadFile = File(...)) -> Dict:
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest = upload_dir / safe_name
    try:
        with open(dest, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        docs = load_document(str(dest))
        chunks = chunk_documents(docs)
        stats = get_chunk_stats(chunks)
        faiss_store.add_documents(chunks)
        bm25_store.add_documents(chunks)
        entry = {
            "id": safe_name,
            "original_name": file.filename,
            "type": Path(file.filename).suffix.lower(),
            "chunks": stats["count"],
            "stats": stats,
        }
        _indexed_sources.append(entry)
        logger.info("Indexed %d chunks from '%s'", stats["count"], file.filename)
        return {"status": "indexed", **entry}
    except Exception as exc:
        logger.error("Ingest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest/url", tags=["Documents"])
async def ingest_url(request: URLIngestRequest) -> Dict:
    try:
        docs = load_document(request.url)
        chunks = chunk_documents(docs)
        stats = get_chunk_stats(chunks)
        faiss_store.add_documents(chunks)
        bm25_store.add_documents(chunks)
        entry = {
            "id": uuid.uuid4().hex,
            "original_name": request.url,
            "type": "web",
            "chunks": stats["count"],
            "stats": stats,
        }
        _indexed_sources.append(entry)
        logger.info("Indexed %d chunks from URL: %s", stats["count"], request.url)
        return {"status": "indexed", **entry}
    except Exception as exc:
        logger.error("URL ingest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query", tags=["Q&A"])
async def query(request: QueryRequest) -> Dict:
    if faiss_store.is_empty and bm25_store.is_empty:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")
    try:
        from backend.chain.rag_chain import run_with_sources
        result = run_with_sources(request.question, k=request.k)
        if request.include_comparison:
            result["retrieval_comparison"] = get_retrieval_comparison(
                request.question, k=request.k
            )
        return result
    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/evaluate", tags=["Evaluation"])
async def evaluate(request: EvaluateRequest) -> Dict:
    if not request.test_data:
        raise HTTPException(status_code=400, detail="test_data cannot be empty")
    try:
        from backend.evaluation.ragas_eval import run_evaluation
        return run_evaluation(request.test_data, run_name=request.run_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/evaluate/build", tags=["Evaluation"])
async def evaluate_build(request: EvaluateBuildRequest) -> Dict:
    if not request.qa_pairs:
        raise HTTPException(status_code=400, detail="qa_pairs cannot be empty")
    try:
        from backend.evaluation.ragas_eval import build_test_data_from_qa_pairs, run_evaluation
        pairs = [p.dict() for p in request.qa_pairs]
        test_data = build_test_data_from_qa_pairs(pairs)
        return run_evaluation(test_data, run_name=request.run_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/evaluate/history", tags=["Evaluation"])
async def eval_history() -> Dict:
    from backend.evaluation.ragas_eval import load_eval_history
    history = load_eval_history()
    return {"history": history, "count": len(history)}


@app.delete("/clear", tags=["System"])
async def clear_indexes() -> Dict:
    faiss_store.clear()
    bm25_store.clear()
    _indexed_sources.clear()
    return {"status": "cleared"}
