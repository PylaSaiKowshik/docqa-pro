"""
conftest.py — pytest configuration and shared fixtures for DocQA-Pro tests.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Override config paths to avoid touching real data dirs during tests ────────
os.environ.setdefault("FAISS_INDEX_PATH", "data/vectorstores/test_faiss")
os.environ.setdefault("CHROMA_PERSIST_PATH", "data/vectorstores/test_chroma")
os.environ.setdefault("BM25_PICKLE_PATH", "data/vectorstores/test_bm25.pkl")
os.environ.setdefault("UPLOAD_DIR", "data/uploads/test")
os.environ.setdefault("EVAL_RESULTS_PATH", "data/test_eval_results.json")
