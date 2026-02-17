import logging
from os import path
from pathlib import Path

import numpy as np
from rapidocr import RapidOCR, EngineType

logger = logging.getLogger(__name__)

# ── Model paths (Paddle format — only usable with the Paddle backend) ────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DET_MODEL = MODEL_DIR / "det" / "inference.pdmodel"
DET_PARAMS = MODEL_DIR / "det" / "inference.pdiparams"
REC_MODEL = MODEL_DIR / "rec" / "inference.pdmodel"
REC_PARAMS = MODEL_DIR / "rec" / "inference.pdiparams"

_ocr_engine: RapidOCR | None = None


# ── Internal helpers ─────────────────────────────────────────────────────────

def _has_paddle_gpu() -> bool:
    """Return True when paddlepaddle-gpu is installed AND a CUDA device is visible."""
    try:
        import paddle
        return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def _custom_paddle_models_exist() -> bool:
    """Return True when all four Paddle model files are present on disk."""
    return all(p.exists() for p in (DET_MODEL, DET_PARAMS, REC_MODEL, REC_PARAMS))


def _build_engine() -> RapidOCR:
    """Build the OCR engine.

    Strategy
    --------
    * **GPU available (Paddle + CUDA)** — use the Paddle backend so we can also
      load custom ``.pdmodel`` / ``.pdiparams`` models when they exist.
    * **No GPU** — use ONNX Runtime (CPU).  Custom Paddle-format models are
      *not* compatible with ONNX Runtime, so RapidOCR's bundled models are used
      instead.  If ONNX Runtime also fails, fall back to bare RapidOCR defaults.
    """

    # ── Path A: Paddle + CUDA (GPU) ──────────────────────────────────────
    if _has_paddle_gpu():
        use_custom = _custom_paddle_models_exist()
        try:
            params: dict = {
                "Global.text_score": 0.5,
                "Global.use_cls": False,
                "Det.engine_type": EngineType.PADDLE,
                "Rec.engine_type": EngineType.PADDLE,
                "EngineConfig.paddle.use_cuda": True,
            }
            if use_custom:
                params.update({
                    "Det.model_path": str(DET_MODEL),
                    "Det.params_path": str(DET_PARAMS),
                    "Rec.model_path": str(REC_MODEL),
                    "Rec.params_path": str(REC_PARAMS),
                })
            engine = RapidOCR(params=params)
            logger.info(
                "OCR engine: Paddle + CUDA%s",
                " (custom models)" if use_custom else "",
            )
            return engine
        except Exception as exc:
            logger.warning("Paddle GPU init failed (%s) — falling back to ONNX Runtime", exc)

    # ── Path B: ONNX Runtime (CPU, no GPU) ───────────────────────────────
    try:
        engine = RapidOCR(params={
            "Global.text_score": 0.5,
            "Global.use_cls": False,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
        })
        logger.info("OCR engine: ONNX Runtime (CPU)")
        return engine
    except Exception as exc:
        logger.warning("ONNX Runtime init failed (%s) — using RapidOCR defaults", exc)

    # ── Path C: last-resort bare defaults ────────────────────────────────
    logger.warning("All preferred backends unavailable — using RapidOCR defaults")
    return RapidOCR()


# ── Public API ───────────────────────────────────────────────────────────────

def get_ocr_engine() -> RapidOCR:
    """Return a shared OCR engine, created on first call."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = _build_engine()
    return _ocr_engine


def run_ocr(img: np.ndarray) -> dict:
    """Run OCR on an image.

    Args:
        img: BGR numpy array (OpenCV format).

    Returns:
        dict with "texts", "boxes", and "scores" lists.
    """
    if img is None or img.size == 0:
        return {"texts": [], "boxes": [], "scores": []}

    engine = get_ocr_engine()
    result = engine(img)

    if not result.txts:
        return {"texts": [], "boxes": [], "scores": []}

    return {
        "texts": list(result.txts),
        "boxes": [box.tolist() for box in result.boxes],
        "scores": list(result.scores),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2
    path = r"image.png"
    data = run_ocr(cv2.imread(path))
    print(data)