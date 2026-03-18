from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.pipeline import run_pipeline

router = APIRouter()


class TextRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Raw document text to summarise.")
    keep_ratio: float = Field(
        default=0.45,
        ge=0.1,
        le=1.0,
        description="Fraction of sentences to keep during compression (0.1–1.0).",
    )


class PDFRequest(BaseModel):
    # base64-encoded PDF content for JSON-body uploads
    pdf_base64: str
    keep_ratio: float = Field(default=0.45, ge=0.1, le=1.0)


@router.post("/summarize/text")
async def summarize_text(body: TextRequest):
    """Summarise plain text (pre-extracted)."""
    try:
        result = run_pipeline(body.text, is_pdf=False, keep_ratio=body.keep_ratio)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/summarize/pdf-base64")
async def summarize_pdf_base64(body: PDFRequest):
    """Summarise a base64-encoded PDF."""
    import base64

    try:
        pdf_bytes = base64.b64decode(body.pdf_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF data.")

    try:
        result = run_pipeline(pdf_bytes, is_pdf=True, keep_ratio=body.keep_ratio)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))