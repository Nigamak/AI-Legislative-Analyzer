"""
compression.py
--------------
Token Compression  —  the key technique required by the problem statement.

Goal: maximise Information Density = value delivered / tokens consumed.

Strategy
--------
1. **Extractive pre-compression**: score every sentence with TF-IDF and
   keep only the top-k% most information-dense sentences before sending
   anything to the generative model.  This alone can cut token usage by
   50-70 % on verbose legal text.

2. **Redundancy pruning**: deduplicate near-identical sentences using
   cosine similarity on simple bag-of-words vectors so the LLM never
   sees repeated clauses.

3. **Legal boilerplate stripping**: regex patterns for common Indian
   legislative boilerplate ("Whereas", "Be it enacted", etc.) that adds
   tokens without adding citizen-relevant meaning.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

_STOPWORDS = set(stopwords.words("english"))

# ── Legal boilerplate patterns (Indian legislative style) ────────────────────
_BOILERPLATE_PATTERNS = [
    r"^whereas[\s,]",
    r"be it enacted by",
    r"it is hereby enacted",
    r"passed by both houses",
    r"received the assent of the president",
    r"published in the gazette",
    r"no\.\s+\d+\s+of\s+\d{4}",          # "No. 15 of 2023"
    r"the\s+\w+\s+act,?\s+\d{4}$",        # short-title lines
    r"arrangement of (?:sections|clauses)",
    r"^chapter\s+[IVXLCDM\d]+\s*$",        # standalone chapter headings
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE | re.MULTILINE)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tfidf_scores(sentences: list[str]) -> list[float]:
    """Compute a simple TF-IDF importance score for each sentence."""
    # Term frequency per sentence
    tf: list[Counter] = [Counter(_tokenize(s)) for s in sentences]
    n = len(sentences)

    # Document frequency
    df: Counter = Counter()
    for counter in tf:
        df.update(counter.keys())

    scores: list[float] = []
    for counter in tf:
        if not counter:
            scores.append(0.0)
            continue
        score = sum(
            freq * math.log((n + 1) / (df[term] + 1))
            for term, freq in counter.items()
        )
        scores.append(score / max(len(counter), 1))
    return scores


def _cosine_sim(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v ** 2 for v in a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Public API ───────────────────────────────────────────────────────────────

def strip_boilerplate(text: str) -> str:
    """Remove common Indian legislative boilerplate sentences."""
    sentences = nltk.sent_tokenize(text)
    kept = [s for s in sentences if not _BOILERPLATE_RE.search(s)]
    return " ".join(kept)


def deduplicate_sentences(
    sentences: list[str], sim_threshold: float = 0.85
) -> list[str]:
    """Remove near-duplicate sentences (cosine similarity ≥ threshold)."""
    bows: list[Counter] = [Counter(_tokenize(s)) for s in sentences]
    kept_indices: list[int] = []

    for i, bow_i in enumerate(bows):
        is_dup = False
        for j in kept_indices:
            if _cosine_sim(bow_i, bows[j]) >= sim_threshold:
                is_dup = True
                break
        if not is_dup:
            kept_indices.append(i)

    return [sentences[i] for i in kept_indices]


def extractive_compress(
    text: str,
    keep_ratio: float = 0.45,
    tokenizer=None,
    max_tokens: int | None = None,
) -> str:
    """
    Extractively compress *text* to approximately *keep_ratio* of its
    original sentences, ranked by TF-IDF importance.

    If *tokenizer* and *max_tokens* are supplied the function also enforces
    a hard token ceiling (useful before feeding chunks to a generative model).
    """
    # 1. Strip boilerplate first
    text = strip_boilerplate(text)

    # 2. Sentence split & dedup
    sentences = nltk.sent_tokenize(text)
    sentences = deduplicate_sentences(sentences)

    if not sentences:
        return text

    # 3. Score
    scores = _tfidf_scores(sentences)

    # 4. Keep top keep_ratio% by score (preserve original order)
    n_keep = max(1, int(len(sentences) * keep_ratio))
    top_indices = sorted(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_keep]
    )
    compressed = " ".join(sentences[i] for i in top_indices)

    # 5. Optional hard token cap
    if tokenizer is not None and max_tokens is not None:
        tokens = tokenizer.encode(compressed, add_special_tokens=False)
        if len(tokens) > max_tokens:
            decoded = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            compressed = decoded

    return compressed


def compression_stats(original: str, compressed: str, tokenizer=None) -> dict:
    """Return token/character reduction metrics."""
    orig_chars = len(original)
    comp_chars = len(compressed)
    stats: dict = {
        "original_chars": orig_chars,
        "compressed_chars": comp_chars,
        "char_reduction_pct": round((1 - comp_chars / max(orig_chars, 1)) * 100, 1),
    }
    if tokenizer:
        orig_tokens = len(tokenizer.encode(original, add_special_tokens=False))
        comp_tokens = len(tokenizer.encode(compressed, add_special_tokens=False))
        stats.update(
            {
                "original_tokens": orig_tokens,
                "compressed_tokens": comp_tokens,
                "token_reduction_pct": round(
                    (1 - comp_tokens / max(orig_tokens, 1)) * 100, 1
                ),
                "information_density_score": round(comp_tokens / max(orig_tokens, 1), 3),
            }
        )
    return stats