"""
chunking.py
-----------
Split very long documents into overlapping chunks that fit within the
model's max input length.  Works at the *sentence* level so chunks never
cut mid-sentence.
"""

from __future__ import annotations

import re
from typing import Iterator

import nltk

# Download punkt tokeniser silently on first use
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# ── Constants ────────────────────────────────────────────────────────────────
# BART-large-cnn has a 1 024-token encoder limit.
# We leave headroom and use a stride to keep context across chunk boundaries.
DEFAULT_MAX_TOKENS = 900
DEFAULT_STRIDE_TOKENS = 100  # overlap between consecutive chunks


def _count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK punkt."""
    return nltk.sent_tokenize(text)


def chunk_document(
    text: str,
    tokenizer,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stride_tokens: int = DEFAULT_STRIDE_TOKENS,
) -> list[str]:
    """
    Return a list of text chunks, each ≤ max_tokens tokens.

    Parameters
    ----------
    text        : Full document text.
    tokenizer   : HuggingFace tokenizer (used for accurate token counting).
    max_tokens  : Hard ceiling per chunk (should be ≤ model encoder limit).
    stride_tokens: Overlap between consecutive chunks for context continuity.
    """
    sentences = split_into_sentences(text)
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = _count_tokens(sentence, tokenizer)

        # If a single sentence exceeds the limit, hard-split it by words
        if s_tokens > max_tokens:
            for sub in _hard_split(sentence, tokenizer, max_tokens):
                chunks.append(sub)
            continue

        if current_tokens + s_tokens > max_tokens:
            # Flush current chunk
            chunks.append(" ".join(current_sentences))

            # Stride: keep the tail sentences whose total ≤ stride_tokens
            tail: list[str] = []
            tail_tokens = 0
            for s in reversed(current_sentences):
                t = _count_tokens(s, tokenizer)
                if tail_tokens + t <= stride_tokens:
                    tail.insert(0, s)
                    tail_tokens += t
                else:
                    break
            current_sentences = tail
            current_tokens = tail_tokens

        current_sentences.append(sentence)
        current_tokens += s_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return [c.strip() for c in chunks if c.strip()]


def _hard_split(sentence: str, tokenizer, max_tokens: int) -> Iterator[str]:
    """Split a single oversized sentence by words."""
    words = sentence.split()
    buf: list[str] = []
    buf_tokens = 0
    for word in words:
        w_tokens = _count_tokens(word, tokenizer)
        if buf_tokens + w_tokens > max_tokens:
            if buf:
                yield " ".join(buf)
            buf = [word]
            buf_tokens = w_tokens
        else:
            buf.append(word)
            buf_tokens += w_tokens
    if buf:
        yield " ".join(buf)