"""
ingestion/loader.py — Multi-format document loader.

Supports:
  - PDF   via pypdf (PyPDFLoader)
  - DOCX  via python-docx (Docx2txtLoader)
  - URLs  via trafilatura + BeautifulSoup fallback
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── Public helpers ────────────────────────────────────────────────────────────

def load_pdf(file_path: str | Path) -> List[Document]:
    """Load a PDF file; each page becomes a Document."""
    path = str(file_path)
    logger.info("Loading PDF: %s", path)
    loader = PyPDFLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({"doc_type": "pdf", "source": path})
    logger.info("  → %d pages loaded", len(docs))
    return docs


def load_docx(file_path: str | Path) -> List[Document]:
    """Load a DOCX file as a single Document."""
    path = str(file_path)
    logger.info("Loading DOCX: %s", path)
    loader = Docx2txtLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({"doc_type": "docx", "source": path})
    logger.info("  → %d section(s) loaded", len(docs))
    return docs


def load_url(url: str) -> List[Document]:
    """
    Load a web page.
    Primary: trafilatura (extracts main content, ignores nav/ads).
    Fallback: LangChain WebBaseLoader (BeautifulSoup).
    """
    logger.info("Loading URL: %s", url)
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text:
                doc = Document(
                    page_content=text,
                    metadata={"doc_type": "web", "source": url},
                )
                logger.info("  → trafilatura: %d chars", len(text))
                return [doc]
    except Exception as exc:  # noqa: BLE001
        logger.warning("trafilatura failed (%s), falling back to WebBaseLoader", exc)

    # Fallback
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({"doc_type": "web", "source": url})
    logger.info("  → WebBaseLoader: %d doc(s)", len(docs))
    return docs


def load_document(source: str | Path) -> List[Document]:
    """
    Auto-detect source type and load accordingly.

    Args:
        source: A file path (PDF/DOCX) or a URL string.

    Returns:
        List of LangChain Documents with populated metadata.
    """
    source_str = str(source)

    if source_str.startswith("http://") or source_str.startswith("https://"):
        return load_url(source_str)

    path = Path(source_str)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in (".docx", ".doc"):
        return load_docx(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Supported types: .pdf, .docx, .doc, or an http/https URL."
        )
