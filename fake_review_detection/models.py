# fake_review_detection/models.py

import logging
import threading
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Same rule as the original code: use the GPU when PyTorch can see one
device = "cuda" if torch.cuda.is_available() else "cpu"

_load_lock = threading.Lock()
_sbert: Any = None
_clip: Any = None


class _Serialized:
    """Lets claims checked in parallel share one model by running its calls one at a time."""

    def __init__(self, model: Any):
        self._model = model
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._model, name)
        if not callable(attribute):
            return attribute

        def call(*args, **kwargs):
            with self._lock:
                return attribute(*args, **kwargs)

        return call


def get_sbert() -> Optional[Any]:
    """The SentenceTransformer model shared by both checkers, loaded on first use (None if it fails)."""
    global _sbert
    with _load_lock:
        if _sbert is None:
            from sentence_transformers import SentenceTransformer
            try:
                logger.info("🔧 Loading SentenceTransformer model...")
                _sbert = _Serialized(SentenceTransformer("all-MiniLM-L6-v2"))
                logger.info("✅ SentenceTransformer model loaded.")
            except Exception as e:
                logger.error(f"❌ Failed to load SentenceTransformer model: {e}")
                _sbert = False
        return _sbert or None


def get_clip() -> Tuple[Optional[Any], Optional[Any]]:
    """The CLIP model and its image preprocessing, loaded on first use ((None, None) if it fails)."""
    global _clip
    with _load_lock:
        if _clip is None:
            import clip
            try:
                logger.info("🔧 Loading CLIP model for web search checker...")
                clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
                _clip = (_Serialized(clip_model), clip_preprocess)
                logger.info("✅ CLIP model loaded.")
            except Exception as e:
                logger.error(f"❌ Failed to load CLIP model: {e}")
                _clip = (None, None)
        return _clip
