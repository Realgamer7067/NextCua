import logging
from pathlib import Path

import numpy as np
from rapidocr import RapidOCR, EngineType

logger = logging.getLogger(__name__)

# ── Model paths ──────────────────────────────────────────────────────────────

MODEL_DIR = "models"
DET_MODEL = f"{MODEL_DIR}/det/inference.pdmodel"
DET_PARAMS = f"{MODEL_DIR}/det/inference.pdiparams"
REC_MODEL = f"{MODEL_DIR}/rec/inference.pdmodel"
REC_PARAMS = f"{MODEL_DIR}/rec/inference.pdiparams"

_ocr_engine: RapidOCR | None = None


# ── Internal helpers ─────────────────────────────────────────────────────────

def _has_paddle_gpu() -> bool:
    try:
        import paddle
        return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def _has_paddle() -> bool:
    try:
        import paddle  # noqa: F401
        return True
    except ImportError:
        return False


def _custom_models_exist() -> bool:
    return all(Path(p).exists() for p in (DET_MODEL, DET_PARAMS, REC_MODEL, REC_PARAMS))


def _build_engine() -> RapidOCR:
    """Build the best available OCR engine, falling back gracefully."""
    use_custom = _custom_models_exist()

    # Try 1: Paddle + CUDA
    if _has_paddle_gpu():
        try:
            params = {
                "Global.text_score": 0.5,
                "Global.use_cls": False,
                "Det.engine_type": EngineType.PADDLE,
                "Rec.engine_type": EngineType.PADDLE,
                "EngineConfig.paddle.use_cuda": True,
            }
            if use_custom:
                params.update({
                    "Det.model_path": DET_MODEL,
                    "Det.params_path": DET_PARAMS,
                    "Rec.model_path": REC_MODEL,
                    "Rec.params_path": REC_PARAMS,
                })
            engine = RapidOCR(params=params)
            logger.info("OCR engine: Paddle + CUDA%s", " (custom models)" if use_custom else "")
            return engine
        except Exception as e:
            logger.warning("Paddle GPU failed (%s), trying next…", e)

    # Try 2: Paddle + CPU
    if _has_paddle():
        try:
            params = {
                "Global.text_score": 0.5,
                "Global.use_cls": False,
                "Det.engine_type": EngineType.PADDLE,
                "Rec.engine_type": EngineType.PADDLE,
                "EngineConfig.paddle.use_cuda": False,
            }
            if use_custom:
                params.update({
                    "Det.model_path": DET_MODEL,
                    "Det.params_path": DET_PARAMS,
                    "Rec.model_path": REC_MODEL,
                    "Rec.params_path": REC_PARAMS,
                })
            engine = RapidOCR(params=params)
            logger.info("OCR engine: Paddle + CPU%s", " (custom models)" if use_custom else "")
            return engine
        except Exception as e:
            logger.warning("Paddle CPU failed (%s), trying next…", e)

    # Try 3: ONNX Runtime (auto-detects CUDA/CPU)
    try:
        engine = RapidOCR(params={
            "Global.text_score": 0.5,
            "Global.use_cls": False,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
        })
        logger.info("OCR engine: ONNX Runtime")
        return engine
    except Exception as e:
        logger.warning("ONNX Runtime failed (%s), trying next…", e)

    # Try 4: bare defaults
    logger.warning("All preferred backends failed — using RapidOCR defaults")
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
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        print("Usage: python image_processing.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: could not read '{sys.argv[1]}'")
        sys.exit(1)

    result = run_ocr(img)

    if result["texts"]:
        for txt, score in zip(result["texts"], result["scores"]):
            print(f"  [{score:.2f}] {txt}")
    else:
        print("No text detected.")
