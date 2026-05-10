"""
evaluation/ragas_eval.py — RAGAS evaluation harness.

Metrics computed:
  - faithfulness        (answer grounded in context)
  - answer_relevancy    (answer relevant to question)
  - context_recall      (context covers ground-truth answer)
  - context_precision   (retrieved chunks are relevant)

RAGAS uses an LLM for faithfulness + relevancy scoring.
If OPENAI_API_KEY is set, it uses GPT-3.5-turbo.
Otherwise, it degrades gracefully (these two metrics are skipped).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import settings

logger = logging.getLogger(__name__)


def _load_ragas_dataset(test_data: List[Dict]) -> Any:
    """
    Convert a list of dicts into a RAGAS-compatible HuggingFace Dataset.

    Expected keys per record:
      - question      (str)
      - ground_truth  (str)     — reference answer
      - answer        (str)     — model's answer
      - contexts      (list[str]) — retrieved chunks
    """
    from datasets import Dataset  # type: ignore

    return Dataset.from_list(test_data)


def run_evaluation(
    test_data: List[Dict],
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on a test dataset.

    Args:
        test_data:  List of dicts with keys: question, answer,
                    contexts (list of strings), ground_truth.
        run_name:   Optional label for this evaluation run.

    Returns:
        Dict with metric scores + metadata, also persisted to disk.
    """
    try:
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
    except ImportError as exc:
        logger.error("RAGAS not installed: %s", exc)
        return {"error": str(exc)}

    logger.info("Starting RAGAS evaluation on %d samples…", len(test_data))

    # Choose which metrics to run based on LLM availability
    if settings.use_openai:
        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
        logger.info("Using OpenAI for LLM-based RAGAS metrics")
    else:
        # Without an LLM, only context_recall and context_precision work
        metrics = [context_recall, context_precision]
        logger.warning(
            "OPENAI_API_KEY not set — skipping faithfulness & answer_relevancy. "
            "Set OPENAI_API_KEY to enable all RAGAS metrics."
        )

    dataset = _load_ragas_dataset(test_data)

    try:
        if settings.use_openai:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        result = evaluate(dataset, metrics=metrics)
    except Exception as exc:  # noqa: BLE001
        logger.error("RAGAS evaluation failed: %s", exc)
        return {"error": str(exc)}

    scores: Dict[str, float] = {}
    for metric in metrics:
        name = metric.name  # type: ignore[attr-defined]
        val = result.get(name)  # type: ignore[attr-defined]
        if val is not None:
            scores[name] = round(float(val), 4)

    report = {
        "run_name": run_name or f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(test_data),
        "metrics": scores,
        "llm_scoring": settings.use_openai,
    }

    _persist_result(report)
    logger.info("RAGAS evaluation complete: %s", scores)
    return report


def _persist_result(report: Dict) -> None:
    """Append evaluation result to the JSON results file."""
    path = Path(settings.eval_results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict] = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            history = []

    history.append(report)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("Results persisted → %s", path)


def load_eval_history() -> List[Dict]:
    """Return all past evaluation runs from the results JSON file."""
    path = Path(settings.eval_results_path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def build_test_data_from_qa_pairs(
    qa_pairs: List[Dict],
    retriever_fn=None,
    answer_fn=None,
) -> List[Dict]:
    """
    Helper to automatically build RAGAS test data from Q&A pairs
    by running the retriever and chain over each question.

    Args:
        qa_pairs:     List of {"question": ..., "ground_truth": ...}
        retriever_fn: callable(question) -> List[str]  (chunk texts)
        answer_fn:    callable(question) -> str         (model answer)
    """
    if retriever_fn is None:
        from backend.retrieval.hybrid import hybrid_search_docs
        retriever_fn = lambda q: [d.page_content for d in hybrid_search_docs(q)]  # noqa: E731

    if answer_fn is None:
        from backend.chain.rag_chain import run_with_sources
        answer_fn = lambda q: run_with_sources(q)["answer"]  # noqa: E731

    records = []
    for pair in qa_pairs:
        question = pair["question"]
        ground_truth = pair.get("ground_truth", "")
        logger.info("Building test record for: %s", question[:60])
        contexts = retriever_fn(question)
        answer = answer_fn(question)
        records.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )
    return records
