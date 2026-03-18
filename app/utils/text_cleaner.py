import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Normalise raw text extracted from a legal PDF or plain-text document.

    Steps
    -----
    1. Unicode normalisation (NFKC).
    2. Remove page-headers / footers that repeat across pages.
    3. Collapse excessive whitespace.
    4. Strip boilerplate artefacts (line numbers, dotted leaders, etc.).
    """
    # 1. Unicode normalise
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove common PDF artefacts (form-feeds, null chars)
    text = text.replace("\x0c", "\n").replace("\x00", "")

    # 3. Remove standalone page numbers like  "— 12 —" or "Page 12 of 34"
    text = re.sub(r"(?i)(page\s+\d+\s+of\s+\d+|—\s*\d+\s*—|\[\s*\d+\s*\])", "", text)

    # 4. Remove dotted leaders used in ToC  "Chapter 1 ............... 3"
    text = re.sub(r"\.{4,}", " ", text)

    # 5. Collapse multiple blank lines → single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # 7. Strip leading/trailing whitespace
    text = text.strip()

    return text


def remove_headers_footers(pages: list[str]) -> list[str]:
    """
    Given a list of per-page text strings, detect and strip repeated lines
    that appear on ≥ 60 % of pages (likely headers/footers).
    """
    if not pages:
        return pages

    from collections import Counter

    line_freq: Counter = Counter()
    for page in pages:
        for line in page.splitlines():
            stripped = line.strip()
            if stripped:
                line_freq[stripped] += 1

    threshold = max(2, int(len(pages) * 0.6))
    boilerplate = {line for line, cnt in line_freq.items() if cnt >= threshold}

    cleaned = []
    for page in pages:
        filtered_lines = [
            l for l in page.splitlines() if l.strip() not in boilerplate
        ]
        cleaned.append("\n".join(filtered_lines))
    return cleaned