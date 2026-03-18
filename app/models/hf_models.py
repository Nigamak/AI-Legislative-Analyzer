from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ── Model config ────────────────────────────────────────────────────────────
# Best open-source models for long legal summarisation (swap as needed):
#   "facebook/bart-large-cnn"      – fast, good quality, English
#   "google/long-t5-tglobal-base"  – handles very long inputs natively
#   "allenai/led-base-16384"       – Longformer Encoder-Decoder
MODEL_NAME = "facebook/bart-large-cnn"

_tokenizer = None
_model = None
_pipe = None


def _load():
    global _tokenizer, _model, _pipe
    if _pipe is not None:
        return _pipe

    device = 0 if torch.cuda.is_available() else -1
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    _pipe = pipeline(
        "summarization",
        model=_model,
        tokenizer=_tokenizer,
        device=device,
    )
    return _pipe


def get_pipeline():
    """Return a cached summarisation pipeline (lazy-loaded on first call)."""
    return _load()


def get_tokenizer():
    """Return the tokenizer (loads models if not already loaded)."""
    _load()
    return _tokenizer