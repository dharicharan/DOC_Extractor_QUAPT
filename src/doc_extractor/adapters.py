"""
Specialist model adapters (PRD v1.4: FR-RTE-004, FR-RTE-006, D14).

Every route runs behind one interface, so the routing table can send a category to a
different kind of model without changing the app:

- VLMAdapter:      a vision-language model that reads the page and returns schema JSON with boxes
                   (Qwen3-VL-8B, the default).
- OCRTextAdapter:  an OCR model reads the page to text, then a text LLM of comparable size maps the
                   text to the schema (for example GLM-OCR + Qwen3-4B-Instruct-2507). The text LLM cannot see the
                   page, so checkboxes and character grids should stay on the VLM route.

OCRTextAdapter is UNTESTED against the real models: it follows the published model-card usage and
needs a GPU run (Kaggle) to confirm. The OCR step gives no coordinates, so values from this route
carry no evidence box (grounded=False) and the app flags them for review.
"""

from abc import ABC, abstractmethod
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from doc_extractor.contracts_c import ExecutionMode, ExecutionProvenance
from doc_extractor.extractor import gpu_dtype, load_capped_image

logger = logging.getLogger("doc_extractor.adapters")

DEFAULT_OCR_MODEL = "zai-org/GLM-OCR"
DEFAULT_TEXT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # pure text, non-thinking only
OCR_PROMPT = "Text Recognition:"


def _side_device() -> str:
    """OCR and text models go on the second GPU when there is one, leaving GPU 0 to the main VLM."""
    import torch

    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    return "auto"


class ModelAdapter(ABC):
    """Common interface for every routed specialist."""

    kind: str = "base"

    @property
    @abstractmethod
    def model_label(self) -> str:
        """Human-readable name(s) of the model(s) used, recorded in the envelope."""

    @abstractmethod
    def extract(self, image_path: str, system_prompt: str) -> Tuple[Dict[str, Any], ExecutionProvenance, Dict[str, Any]]:
        """Returns (raw_fields, provenance, meta). meta['grounded'] says whether boxes are real."""


class VLMAdapter(ModelAdapter):
    kind = "vlm"

    def __init__(self, extractor) -> None:
        self._extractor = extractor

    @property
    def model_label(self) -> str:
        return self._extractor.model_name

    def extract(self, image_path, system_prompt):
        raw, prov = self._extractor.extract_page_image(image_path, None, system_prompt=system_prompt)
        return raw, prov, {"grounded": True, "models": {"specialist": self.model_label}}


class OCRTextAdapter(ModelAdapter):
    kind = "ocr_text"

    def __init__(
        self,
        ocr_model: str = DEFAULT_OCR_MODEL,
        text_model: str = DEFAULT_TEXT_MODEL,
        max_ocr_tokens: int = 4096,
        max_json_tokens: int = 2048,
        load_in_4bit: bool = True,
    ) -> None:
        self.ocr_model = ocr_model
        self.text_model = text_model
        self.max_ocr_tokens = max_ocr_tokens
        self.max_json_tokens = max_json_tokens
        self.load_in_4bit = load_in_4bit
        self._ocr = None
        self._ocr_proc = None
        self._txt = None
        self._tok = None

    @property
    def model_label(self) -> str:
        return f"{self.ocr_model} + {self.text_model}"

    # -- loading ---------------------------------------------------------
    def _load_ocr(self):
        if self._ocr is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self._ocr_proc = AutoProcessor.from_pretrained(self.ocr_model, trust_remote_code=True)
            self._ocr = AutoModelForImageTextToText.from_pretrained(
                self.ocr_model, torch_dtype=gpu_dtype(), device_map=_side_device(), trust_remote_code=True,
                attn_implementation="sdpa",
            )
            self._ocr.eval()
        return self._ocr, self._ocr_proc

    def _load_text(self):
        if self._txt is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self._tok = AutoTokenizer.from_pretrained(self.text_model, trust_remote_code=True)
            kwargs: Dict[str, Any] = {"device_map": _side_device(), "trust_remote_code": True}
            if self.load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=gpu_dtype()
                )
            else:
                kwargs["torch_dtype"] = gpu_dtype()
            self._txt = AutoModelForCausalLM.from_pretrained(self.text_model, **kwargs)
            self._txt.eval()
        return self._txt, self._tok

    # -- steps -----------------------------------------------------------
    def read_text(self, image_path: str) -> str:
        """Step 1: OCR model turns the page image into text."""
        import torch

        model, proc = self._load_ocr()
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": load_capped_image(image_path)}, {"type": "text", "text": OCR_PROMPT}],
        }]
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_ocr_tokens, do_sample=False)
        return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def text_to_fields(self, ocr_text: str, system_prompt: str) -> Dict[str, Any]:
        """Step 2: text LLM maps OCR text to the schema. It cannot see the page, so bbox is null."""
        import torch

        model, tok = self._load_text()
        sys_msg = (
            system_prompt
            + "\n\nIMPORTANT: you are given OCR text of the page, not the image. Use only what appears in the "
              "text. Set every \"bbox\" to null. Use null for any value not present in the text."
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"OCR TEXT:\n{ocr_text}\n\nReturn the JSON object only."},
        ]
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([prompt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_json_tokens, do_sample=False)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        cleaned = re.sub(r"^```json\s*", "", text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        return json.loads(cleaned) if cleaned else {}

    def extract(self, image_path, system_prompt):
        ocr_text = self.read_text(image_path)
        raw = self.text_to_fields(ocr_text, system_prompt)
        prov = ExecutionProvenance(
            mode=ExecutionMode.GENUINE_LLM,
            model_name=self.model_label,
            provider="transformers_ocr_text",
            fallback_occurred=False,
            details={"ocr_chars": len(ocr_text)},
        )
        meta = {
            "grounded": False,
            "ocr_text_preview": ocr_text[:500],
            "models": {"ocr": self.ocr_model, "text_llm": self.text_model, "specialist": self.model_label},
        }
        return raw, prov, meta
