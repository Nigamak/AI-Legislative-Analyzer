"""
pipeline.py
-----------
High-level orchestrator called by the API routes.
Accepts raw bytes (PDF) or plain text and returns a structured result.
"""

from __future__ import annotations

from app.utils.pdf_parser import extract_text_from_bytes
from app.services.summarizer import summarise_document


def run_pipeline(
    content: bytes | str,
    is_pdf: bool = True,
    keep_ratio: float = 0.45,
) -> dict:
    """
    Parameters
    ----------
    content    : Raw PDF bytes or plain-text string.
    is_pdf     : If True, extract text from PDF first.
    keep_ratio : Fraction of sentences to keep during compression (0–1).

    Returns
    -------
    {
        "summary"    : str,
        "key_points" : list[str],
        "stats"      : dict,
        "word_count" : int,
    }
    """
    # 1. Parse
    if is_pdf:
        text = extract_text_from_bytes(content)
    else:
        text = content if isinstance(content, str) else content.decode("utf-8", errors="replace")

    if not text.strip():
        return {
            "summary": "No readable text could be extracted from the document.",
            "key_points": [],
            "stats": {},
            "word_count": 0,
        }

    word_count = len(text.split())

    # 2. Summarise (with stats)
    result = summarise_document(text, keep_ratio=keep_ratio, return_stats=True)
    result["word_count"] = word_count

    return result