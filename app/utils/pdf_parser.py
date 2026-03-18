import io
from pathlib import Path
from typing import Union

import pdfplumber

from app.utils.text_cleaner import clean_text, remove_headers_footers


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract and clean text from raw PDF bytes."""
    pages_text: list[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            pages_text.append(raw)

    # Remove repeated headers/footers
    pages_text = remove_headers_footers(pages_text)

    # Join pages with a separator, then global clean
    full_text = "\n\n".join(pages_text)
    return clean_text(full_text)


def extract_text_from_path(path: Union[str, Path]) -> str:
    """Extract and clean text from a PDF file path."""
    with open(path, "rb") as f:
        return extract_text_from_bytes(f.read())