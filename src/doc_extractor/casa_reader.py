"""
DOC-01 CASA reader: runs PaddleOCR-VL on a page and applies the DOC-01 rules.

The model is loaded once and reused. It runs on CUDA when available (fp16), otherwise on CPU.
The model is asked for a short JSON object only, so generation stops after a few dozen tokens.
Rules and review routing come from casa_form.normalise_casa_fields.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image

from doc_extractor.casa_form import (
    CasaFormResult,
    PROMPT,
    fields_from_ocr_text,
    ground_fields,
    normalise_casa_fields,
    parse_model_json,
)

logger = logging.getLogger(__name__)

MODEL_ID = "PaddlePaddle/PaddleOCR-VL"


class PaddleVLCasaReader:
    """Holds the PaddleOCR-VL model and reads one CASA form page per call."""

    def __init__(self, model_id: str = MODEL_ID, max_new_tokens: int = 320) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=dtype).to(self._device).eval()
        logger.info("PaddleOCR-VL loaded on %s", self._device)

    def generate_text(self, image: Image.Image, prompt: str = PROMPT) -> str:
        """One model call on a page image. Returns the raw text the model produced."""
        import torch

        self._load()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[chat], images=[image.convert("RGB")], return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self._processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def ocr_tokens(image: Image.Image) -> list:
    """Printed-text tokens with boxes, from RapidOCR (CPU). Used only as evidence (FR-EXT-004)."""
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    global _RAPID
    if "_RAPID" not in globals() or _RAPID is None:
        _RAPID = RapidOCR()
    res, _ = _RAPID(np.array(image.convert("RGB"))[:, :, ::-1])  # RapidOCR expects BGR
    tokens = []
    for box, text, _score in res or []:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        tokens.append({"text": text, "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))], "page": 1})
    return tokens


_RAPID = None


def read_casa_form(
    source: Union[str, Path, Image.Image],
    reader: Optional[PaddleVLCasaReader] = None,
) -> Dict[str, Any]:
    """
    Reads one CASA form and returns the DOC-01 result as a plain dict, with timing.
    Model failures do not raise: every field is returned as null and sent to review.
    """
    image = source if isinstance(source, Image.Image) else Image.open(source)
    reader = reader or PaddleVLCasaReader()

    started = time.perf_counter()
    try:
        raw_text = reader.generate_text(image)
        model_error = None
    except Exception as exc:  # fail closed: no guessed values
        logger.exception("PaddleOCR-VL failed")
        raw_text = ""
        model_error = str(exc)
    model_secs = round(time.perf_counter() - started, 2)

    # The model may answer with JSON (if it follows the prompt) or with plain page text
    # (the PaddleOCR-VL "OCR:" behaviour). Use JSON when present, otherwise read the labels.
    model_fields = parse_model_json(raw_text)
    if model_fields is None and raw_text.strip():
        model_fields = fields_from_ocr_text(raw_text)
    result: CasaFormResult = normalise_casa_fields(model_fields if model_fields is not None else raw_text)
    try:
        result = ground_fields(result, ocr_tokens(image))
    except Exception as exc:  # no OCR evidence means no value can be trusted
        logger.exception("Evidence grounding failed")
        result.flags.append({"code": "EVIDENCE_UNAVAILABLE", "message": str(exc)})
        for f in result.fields.values():
            if f.value is not None:
                f.value = None
                f.null_reason = "absent"
                f.review_required = True
                f.error = "EVIDENCE_UNAVAILABLE"
        result.status = "needs_review"
    out = result.to_dict()
    out["timing"] = {"model_seconds": model_secs}
    out["raw_model_output"] = raw_text
    if model_error:
        out["flags"].append({"code": "MODEL_ERROR", "message": model_error})
        out["status"] = "needs_review"
    return out
