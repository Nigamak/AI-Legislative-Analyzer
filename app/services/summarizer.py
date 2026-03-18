"""
summarizer.py
-------------
Wrapper around the HuggingFace pipeline that produces citizen-friendly
summaries from (possibly large) compressed legal text.
"""

from __future__ import annotations

from app.models.hf_models import get_pipeline, get_tokenizer
from app.services.chunking import chunk_document
from app.services.compression import extractive_compress, compression_stats

# ── Summarisation parameters ─────────────────────────────────────────────────
_MIN_SUMMARY_LEN = 80
_MAX_SUMMARY_LEN = 300   # tokens per chunk summary
_CHUNK_MAX_TOKENS = 900  # must be < model encoder limit


def _summarise_chunk(text: str) -> str:
    pipe = get_pipeline()
    result = pipe(
        text,
        min_length=_MIN_SUMMARY_LEN,
        max_length=_MAX_SUMMARY_LEN,
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"]


def summarise_document(
    text: str,
    keep_ratio: float = 0.45,
    return_stats: bool = False,
) -> dict:
    """
    Full pipeline:
      raw text → token compression → chunking → per-chunk summarisation
      → hierarchical re-summarisation (if many chunks) → final output.

    Returns
    -------
    {
        "summary": str,
        "key_points": list[str],
        "stats": dict  (only if return_stats=True)
    }
    """
    tokenizer = get_tokenizer()

    # ── Step 1: Token Compression ────────────────────────────────────────────
    compressed = extractive_compress(text, keep_ratio=keep_ratio, tokenizer=tokenizer)
    stats = compression_stats(text, compressed, tokenizer) if return_stats else {}

    # ── Step 2: Chunk ────────────────────────────────────────────────────────
    chunks = chunk_document(compressed, tokenizer, max_tokens=_CHUNK_MAX_TOKENS)

    # ── Step 3: Summarise each chunk ─────────────────────────────────────────
    chunk_summaries: list[str] = [_summarise_chunk(c) for c in chunks]

    # ── Step 4: Hierarchical re-summarisation if > 1 chunk ───────────────────
    if len(chunk_summaries) == 1:
        final_summary = chunk_summaries[0]
    else:
        combined = " ".join(chunk_summaries)
        # If combined summaries still exceed the model limit, compress again
        combined_tokens = len(tokenizer.encode(combined, add_special_tokens=False))
        if combined_tokens > _CHUNK_MAX_TOKENS:
            combined = extractive_compress(
                combined,
                keep_ratio=0.6,
                tokenizer=tokenizer,
                max_tokens=_CHUNK_MAX_TOKENS,
            )
        final_summary = _summarise_chunk(combined)

    # ── Step 5: Extract key points (sentence-level from final summary) ────────
    import nltk
    sentences = nltk.sent_tokenize(final_summary)
    key_points = [s.strip() for s in sentences if len(s.strip()) > 30]

    result: dict = {
        "summary": final_summary,
        "key_points": key_points,
    }
    if return_stats:
        result["stats"] = stats
    return result