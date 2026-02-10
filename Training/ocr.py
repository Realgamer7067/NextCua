import numpy as np
from rapidocr import RapidOCR, EngineType


def create_ocr_engine() -> RapidOCR:
    """Create and return a RapidOCR engine with Paddle CUDA and custom models."""
    return RapidOCR(params={
        "Global.text_score": 0.5,
        "Global.use_cls": False,

        "Det.engine_type": EngineType.PADDLE,
        "Rec.engine_type": EngineType.PADDLE,

        "Det.model_path": "models/det/inference.pdmodel",
        "Det.params_path": "models/det/inference.pdiparams",

        "Rec.model_path": "models/rec/inference.pdmodel",
        "Rec.params_path": "models/rec/inference.pdiparams",

        "EngineConfig.paddle.use_cuda": True,
    })


def run_ocr(engine: RapidOCR, img: np.ndarray):
    """Run OCR on an image and return the result.

    Args:
        engine: A RapidOCR engine instance.
        img: A BGR numpy array (OpenCV format).

    Returns:
        RapidOCROutput with .boxes, .txts, and .scores.
    """
    return engine(img)


if __name__ == "__main__":
    import cv2, sys

    engine = create_ocr_engine()

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
    else:
        print("Usage: python ocr.py <image_path>")
        sys.exit(1)

    result = run_ocr(engine, img)
    if result.txts:
        for box, txt, score in zip(result.boxes, result.txts, result.scores):
            print(f"[{score:.2f}] {txt}")
    else:
        print("No text detected.")
