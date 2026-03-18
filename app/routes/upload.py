from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

ALLOWED_TYPES = {"application/pdf", "text/plain"}
MAX_SIZE_MB = 50


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a PDF or plain-text file and return its extracted text.
    The text can then be POSTed to /api/summarize.
    """
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Only PDF and plain text are accepted.",
        )

    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed is {MAX_SIZE_MB} MB.",
        )

    from app.utils.pdf_parser import extract_text_from_bytes
    from app.utils.text_cleaner import clean_text

    if content_type == "application/pdf":
        text = extract_text_from_bytes(data)
    else:
        text = clean_text(data.decode("utf-8", errors="replace"))

    return JSONResponse(
        content={
            "filename": file.filename,
            "size_mb": round(size_mb, 2),
            "word_count": len(text.split()),
            "char_count": len(text),
            "text_preview": text[:500],
            "full_text": text,
        }
    )