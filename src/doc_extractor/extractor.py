"""
Extraction Engine and Evidence Linker for Module C.
Strictly adheres to PRD §5.4 (FR-EXT-001, FR-EXT-002, FR-EXT-003, FR-EXT-004, FR-EXT-005, FR-EXT-007)
and PRD §6.3 Output Contract.
Supports 1-Stage Multimodal extraction with Qwen2.5-VL / Qwen3-VL as well as
grammar-guided LLM and deterministic extraction.
"""

import base64
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union, get_origin
import urllib.request
from pydantic import BaseModel

from doc_extractor.contracts_b import BoundingBox, ModuleBOutputPayload, PageLayoutOutput
from doc_extractor.contracts_c import (
    ExecutionMode,
    ExecutionProvenance,
    ExtractionResult,
    FieldGrounding,
    GenericExtractionResult,
    PageImageRef,
)
from doc_extractor.exceptions import LLMExecutionError
from doc_extractor import vlm_runtime
from doc_extractor.vlm_grounding import (
    PageLeaf,
    aggregate_leaves,
    build_partial_model_from_pages,
    relaxed_schema,
    validate_allowing_missing,
    collect_generic_leaves,
    collect_schema_leaves,
    validate_page_leaves,
)
from doc_extractor.schemas import (
    CASASchema,
    DischargeSummarySchema,
    InsuranceApplicationSchema,
    MedicationItem,
    NomineeDetails,
)

logger = logging.getLogger("doc_extractor.extractor")


# Cap on the pixels of the image sent to a vision model (default 1.3 MP; override with
# DOC_MAX_IMAGE_PIXELS, e.g. 4200000 keeps full 200 DPI on a bigger GPU) and on the smaller image used for
# classification (default 0.5 MP; DOC_CLASSIFY_MAX_PIXELS). The values and the reasoning live in
# vlm_runtime (single source of truth); they are re-exported here because v1.5 code and the Kaggle docs
# refer to them as doc_extractor.extractor.MAX_IMAGE_PIXELS / CLASSIFY_MAX_PIXELS.
MAX_IMAGE_PIXELS = vlm_runtime.DEFAULT_MAX_IMAGE_PIXELS
CLASSIFY_MAX_PIXELS = vlm_runtime.DEFAULT_CLASSIFY_MAX_PIXELS


def parse_json_tolerant(text: str) -> Dict[str, Any]:
    """Parse model JSON. Strips code fences and extra prose. If the answer was cut off (token limit),
    keep the complete fields and close the object, instead of failing the whole page."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", (text or "").strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in model output: {cleaned[:200]!r}")
    body = cleaned[start:]
    try:
        return json.loads(body[: body.rfind("}") + 1], strict=False)
    except json.JSONDecodeError:
        pass
    # Truncated: walk back to each '}' and try to close the object there
    pos = len(body)
    while True:
        pos = body.rfind("}", 0, pos)
        if pos < 0:
            raise ValueError(f"could not recover JSON from model output: {cleaned[:200]!r}")
        try:
            obj = json.loads(body[: pos + 1] + "}", strict=False)
            logger.warning("OCR text step: model output was truncated; kept the complete fields")
            return obj
        except json.JSONDecodeError:
            continue


def _flat_to_nested(raw: Dict[str, Any]) -> Dict[str, Any]:
    """VLM output {"applicant.full_name": {"value": ..., "bbox": ...}} -> nested {"applicant": {"full_name": ...}}
    so it validates against the Pydantic schema. Free-form other.* keys are dropped."""
    nested: Dict[str, Any] = {}
    for key, item in (raw or {}).items():
        if key.startswith("other."):
            continue
        value = item.get("value") if isinstance(item, dict) else item
        node = nested
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                break
        else:
            node[parts[-1]] = value
    return nested


def gpu_dtype():
    """
    Torch dtype for side models (the v1.5 adapters): bfloat16 only on GPUs with native support (compute
    capability >= 8.0, every GPU considered), float16 on older ones such as the T4, float32 without a GPU.
    One rule, shared with the Qwen-VL loader (``vlm_runtime.select_dtype``).
    """
    import torch

    hardware = vlm_runtime.detect_hardware(torch)
    return getattr(torch, vlm_runtime.select_dtype(hardware, "auto", allow_cpu=True))


def load_capped_image(image_input, max_pixels: int = MAX_IMAGE_PIXELS):
    """Open an image (path or bytes) as RGB and downscale it, keeping aspect ratio, to at most max_pixels.
    Boxes are on a 0-1000 normalised grid, so resizing does not change them."""
    import io as _io
    from PIL import Image as _Image

    if isinstance(image_input, (str, Path)):
        img = _Image.open(image_input)
    elif isinstance(image_input, (bytes, bytearray)):
        img = _Image.open(_io.BytesIO(image_input))
    else:
        img = image_input
    img = img.convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / float(w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), _Image.LANCZOS)
    return img


def make_bnb_config(compute_dtype=None):
    """4-bit NF4 with double quantisation (PRD 12.1/12.2): keeps a 4B/7B VLM inside a 16 GB T4."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype or gpu_dtype(),
    )


def free_gpu_memory() -> None:
    """Release cached CUDA blocks after a generation so multi-page sweeps do not fragment VRAM."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # torch missing or CUDA in a bad state must not break extraction
        pass


def token_confidences(model, generated, input_len: int, decode) -> Tuple[str, List[Tuple[int, int, float]]]:
    """From model.generate(..., return_dict_in_generate=True, output_scores=True) output, return the text
    and one (char_start, char_end, probability) per generated token (FR-CNF-001)."""
    import torch

    seq = generated.sequences[0][input_len:]
    scores = model.compute_transition_scores(generated.sequences, generated.scores, normalize_logits=True)[0]
    probs = torch.exp(scores.float()).clamp(0.0, 1.0).tolist()
    spans: List[Tuple[int, int, float]] = []
    pos = 0
    for tok_id, p in zip(seq.tolist(), probs):
        piece = decode([tok_id])
        spans.append((pos, pos + len(piece), float(p)))
        pos += len(piece)
    return "".join(decode([t]) for t in seq.tolist()), spans


def attach_field_confidence(raw: Dict[str, Any], text: str, spans: List[Tuple[int, int, float]]) -> None:
    """Set raw[key]["confidence"] to the geometric-mean token probability of that field's "value" text.
    Fields whose value text cannot be located keep no confidence (None), never a made-up number."""
    import math

    for key, item in raw.items():
        if not isinstance(item, dict):
            continue
        m = re.search(r'"%s"\s*:\s*\{\s*"value"\s*:\s*(.*?)\s*(?:,\s*"(?:bbox|uncertain|confidence)"|\})' % re.escape(key),
                      text, flags=re.DOTALL)
        if not m:
            continue
        lo, hi = m.start(1), m.end(1)
        ps = [p for a, b, p in spans if a < hi and b > lo and p > 0.0]
        covered = [p for a, b, p in spans if a < hi and b > lo]
        if ps and len(ps) == len(covered):
            item["confidence"] = round(math.exp(sum(math.log(p) for p in ps) / len(ps)), 4)
        elif covered:
            item["confidence"] = 0.0

SCHEMA_MAP: Dict[str, Type[BaseModel]] = {
    "DOC-01": CASASchema,
    "DOC-02": InsuranceApplicationSchema,
    "DOC-03": DischargeSummarySchema,
}


def format_indian_currency(n: int) -> str:
    """Formats an integer according to the Indian numbering system (e.g. 500000 -> 5,00,000)."""
    s = str(abs(n))
    if len(s) <= 3:
        res = s
    else:
        last_three = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        res = ",".join(parts) + "," + last_three
    return f"-{res}" if n < 0 else res




def _find_labeled_line_evidence(
    value: Any,
    pages: List[PageLayoutOutput],
    field_name: Optional[str],
    evidence_value: Any = None,
) -> Optional[Dict[str, Any]]:
    """Find field evidence by matching a value and its label across adjacent OCR blocks."""
    if not field_name:
        return None

    leaf = field_name.split(".")[-1].lower()

    aliases = {
        "full_name": ("applicant name", "customer name", "account holder name", "name"),
        "pan_card_number": ("pan card number", "pan number", "pan"),
        "date_of_birth": ("date of birth", "birth date", "dob"),
        "phone_number": ("mobile number", "phone number", "contact number", "mobile", "phone"),
        "account_type": ("account type",),
        "initial_deposit_amount": ("initial deposit amount", "deposit amount", "initial deposit"),
        "currency": ("currency",),
        "cheque_book": ("cheque book", "check book"),
        "debit_card": ("debit card",),
        "internet_banking": ("internet banking", "net banking"),
    }

    field_aliases = aliases.get(
        leaf,
        (leaf.replace("_", " "),),
    )

    def normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    target_values = []

    if isinstance(value, bool):
        target_values.append("yes" if value else "no")
    else:
        target_values.append(value)

    if evidence_value is not None:
        target_values.append(evidence_value)

    targets = set()
    for item in target_values:
        if item is None or isinstance(item, bool):
            continue

        if isinstance(item, (int, float, Decimal)) and float(item).is_integer():
            item = str(int(item))

        normalized = normalize(str(item))
        if normalized:
            targets.add(normalized)

    if not targets:
        return None

    normalized_aliases = [
        normalize(alias) for alias in field_aliases if normalize(alias)
    ]

    best_match = None
    best_score = -1.0

    for page in pages:
        blocks = []

        for block in page.blocks:
            bbox = block.bbox_original or block.bbox_preprocessed
            if bbox is None or not block.text.strip():
                continue

            height = max(1.0, bbox.y1 - bbox.y0)

            blocks.append({
                "block": block,
                "bbox": bbox,
                "cy": (bbox.y0 + bbox.y1) / 2.0,
                "height": height,
            })

        # Group vertically aligned OCR words into rows.
        rows = []

        for item in sorted(blocks, key=lambda x: x["cy"]):
            matching_row = None

            for row in reversed(rows):
                row_cy = sum(x["cy"] for x in row) / len(row)
                tolerance = max(
                    10.0,
                    0.55 * min(item["height"], max(x["height"] for x in row)),
                )

                if abs(item["cy"] - row_cy) <= tolerance:
                    matching_row = row
                    break

            if matching_row is None:
                rows.append([item])
            else:
                matching_row.append(item)

        # Split rows at large horizontal gaps.
        for row in rows:
            row.sort(key=lambda x: x["bbox"].x0)
            segments = []
            current = []

            for item in row:
                if current:
                    previous = current[-1]
                    gap = item["bbox"].x0 - previous["bbox"].x1
                    max_gap = max(
                        45.0,
                        1.75 * max(item["height"], previous["height"]),
                    )

                    if gap > max_gap:
                        segments.append(current)
                        current = []

                current.append(item)

            if current:
                segments.append(current)

            for segment in segments:
                line_text = " ".join(
                    item["block"].text.strip() for item in segment
                )
                normalized_line = normalize(line_text)

                matched_aliases = [
                    alias for alias in normalized_aliases
                    if alias in normalized_line
                ]

                if not matched_aliases:
                    continue

                if not any(target in normalized_line for target in targets):
                    continue

                bbox = {
                    "x0": min(x["bbox"].x0 for x in segment),
                    "y0": min(x["bbox"].y0 for x in segment),
                    "x1": max(x["bbox"].x1 for x in segment),
                    "y1": max(x["bbox"].y1 for x in segment),
                }

                confs = [
                    x["block"].ocr_confidence for x in segment
                    if x["block"].ocr_confidence is not None
                ]
                confidence = (
                    round(sum(confs) / len(confs), 4)
                    if confs
                    else None
                )

                specificity = max(len(alias) for alias in matched_aliases)
                score = specificity + (confidence if confidence is not None else 0.0)

                if score > best_score:
                    best_score = score
                    best_match = {
                        "page": page.page_number,
                        "bbox": (
                            bbox["x0"], bbox["y0"],
                            bbox["x1"], bbox["y1"],
                        ),
                        "text": line_text[:160],
                        "ocr_confidence": confidence,
                    }

    return best_match


def find_evidence_for_value(
    value: Any,
    pages: List[PageLayoutOutput],
    field_name: Optional[str] = None,
    evidence_value: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    FR-EXT-004 & GR-002: Evidence Bounding Boxes.
    Queries LayoutBlocks, CharacterGrids, HTMLTables, and Checkboxes to find
    page, bbox (x0, y0, x1, y1) in 200-DPI pixel space, and supporting text snippet.
    Supports Western (500,000) and Indian (5,00,000) comma numbering systems.
    """
    if value is None:
        return None

    # Checkboxes: booleans must match visual checkbox marks (FR-EXT-007)
    if isinstance(value, bool):
        field_tokens = []
        if field_name:
            leaf = field_name.split(".")[-1].lower()
            field_tokens = [t for t in re.split(r"[^a-z0-9]+", leaf) if len(t) >= 2]

        matched_any_checkbox_for_field = False
        for page in pages:
            for cb in page.checkboxes:
                cb_id_low = cb.mark_id.lower()
                cb_tokens = [t for t in re.split(r"[^a-z0-9]+", cb_id_low) if len(t) >= 2 and t not in ("cb", "chk", "box", "checkbox", "mark")]
                is_field_cb = (
                    bool(field_tokens)
                    and (
                        any(t in field_tokens for t in cb_tokens)
                        or any(t in cb_id_low for t in field_tokens if len(t) >= 3)
                    )
                )
                if is_field_cb:
                    matched_any_checkbox_for_field = True
                    is_checked = (cb.state == "[x]")
                    if is_checked == value:
                        bbox = cb.bbox_original or cb.bbox_preprocessed
                        return {
                            "page": page.page_number,
                            "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                            "text": cb.state,
                            "ocr_confidence": cb.ocr_confidence,
                            "uncertain": cb.is_uncertain,
                        }

        if matched_any_checkbox_for_field:
            return None

        if not field_name:
            for page in pages:
                for cb in page.checkboxes:
                    is_checked = (cb.state == "[x]")
                    if is_checked == value:
                        bbox = cb.bbox_original or cb.bbox_preprocessed
                        return {
                            "page": page.page_number,
                            "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                            "text": cb.state,
                            "ocr_confidence": cb.ocr_confidence,
                            "uncertain": cb.is_uncertain,
                        }

        # Textual YES/NO evidence is acceptable only when explicitly
        # associated with the requested field label.
        if field_name and not matched_any_checkbox_for_field:
            text_evidence = _find_labeled_line_evidence(
                value, pages, field_name, evidence_value
            )
            if text_evidence:
                return text_evidence

        return None

    candidates = [str(value).strip().lower()]
    if evidence_value is not None:
        candidates.append(str(evidence_value).strip().lower())

    clean_numeric_targets: List[str] = []

    if isinstance(value, (int, float, Decimal)):
        num = float(value)
        if num.is_integer():
            int_val = int(num)
            indian_fmt = format_indian_currency(int_val).lower()
            western_fmt = f"{int_val:,}".lower()
            candidates.extend([
                str(int_val),
                western_fmt,
                indian_fmt,
                western_fmt.replace(",", " "),
                indian_fmt.replace(",", " "),
            ])
            clean_numeric_targets.append(str(int_val))
        else:
            candidates.extend([f"{num:,.2f}", f"{num:.2f}", str(num)])
            clean_numeric_targets.append(f"{num:.2f}")

    def matches_text(text: str) -> bool:
        t_low = text.lower()
        if any(c in t_low for c in candidates if c):
            return True
        if clean_numeric_targets:
            t_clean = re.sub(r"[,\s]", "", t_low)
            if any(target in t_clean for target in clean_numeric_targets if target):
                return True
        return False

    for page in pages:
        for grid in page.character_grids:
            grid_text = grid.reconstructed_text.strip()
            if matches_text(grid_text) or any(grid_text.lower() in c for c in candidates if len(grid_text) > 3):
                bbox = grid.bbox_original or grid.bbox_preprocessed
                return {
                    "page": page.page_number,
                    "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                    "text": grid_text,
                    "ocr_confidence": grid.mean_ocr_confidence,
                    "uncertain": grid.has_ambiguity or any(c.is_uncertain for c in grid.character_boxes),
                }

        # 2. Label-aware matching across adjacent OCR word blocks.
        # This handles names, normalized dates/phones and repeated values.
        labeled_evidence = _find_labeled_line_evidence(
            value, [page], field_name, evidence_value
        )
        if labeled_evidence:
            return labeled_evidence

        # 3. Layout blocks (paragraphs, key-value pairs, headers)
        for block in page.blocks:
            block_text = block.text.strip()
            if matches_text(block_text):
                bbox = block.bbox_original or block.bbox_preprocessed
                return {
                    "page": page.page_number,
                    "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                    "text": block_text[:120],
                    "ocr_confidence": block.ocr_confidence,
                }

        for table in page.tables:
            if matches_text(table.html_content):
                bbox = table.bbox_original
                return {
                    "page": page.page_number,
                    "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                    "text": table.html_content[:120],
                    "ocr_confidence": table.ocr_confidence,
                }

    return None


class SchemaGuidedExtractor:
    """
    Module C Schema-Guided Extractor.
    FR-EXT-001, FR-EXT-002: Extracts schema fields using LLM grammar-guided decoding
    (instructor + litellm, or Qwen-VL multimodal) when use_llm=True. Falls back to deterministic
    heuristics for offline testing or when no LLM is configured.
    Supports strict no-fallback mode and execution provenance tracking (SCH-007).
    """

    def __init__(
        self,
        model_name: str = vlm_runtime.DEFAULT_VLM_MODEL_ID,
        use_llm: bool = False,
        allow_fallback: bool = True,
        prompt_version: str = "1.0.0",
        provider_name: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        is_mock: bool = False,
        llm_caller: Optional[Callable[[ModuleBOutputPayload, Type[BaseModel]], Tuple[Any, Optional[str]]]] = None,
        # Single GPU by default (faster than splitting layers across cards). "single" resolves to
        # cuda:0 when a GPU exists, else "auto". Override with DOC_DEVICE_MAP (e.g. "balanced", "auto").
        device: str = os.environ.get("DOC_DEVICE_MAP", "single"),
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        trust_remote_code: bool = False,
        load_in_4bit: bool = True,
        multimodal: Optional[bool] = None,
        precision: str = "auto",
        allow_cpu: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        min_vram_gib: Optional[float] = None,
    ):
        """
        ``multimodal`` selects whether this extractor reads Module A page images directly
        (Qwen-VL, PRD v1.3/v1.5) when page images are supplied. ``None`` infers it from the model:
        VLM model names (e.g. 'Qwen/Qwen3-VL-4B-Instruct') are multimodal; Ollama/text models
        are not and keep extracting from Module B markdown.

        Local Qwen-VL runtime (see ``doc_extractor.vlm_runtime``): ``precision`` is ``"auto"`` (BF16
        only on GPUs with native support, FP16 otherwise, e.g. Tesla T4), or an explicit
        ``"bfloat16"`` / ``"float16"`` / ``"float32"`` that is honoured or rejected, never silently
        changed. ``load_in_4bit`` (default) uses bitsandbytes NF4 with the selected compute dtype and
        fails with an actionable error when unavailable; it never degrades to another precision.
        ``allow_cpu`` opts in to a (very slow) CPU run. ``max_pixels`` defaults to ``MAX_IMAGE_PIXELS``
        (1.3 MP, DOC_MAX_IMAGE_PIXELS) so a page fits a 16 GB T4; ``min_pixels`` defaults to the
        checkpoint's official minimum. ``device`` defaults to DOC_DEVICE_MAP (``single`` = GPU 0).
        ``trust_remote_code`` defaults to False: Qwen-VL is natively supported by Transformers >= 4.57.1
        and requesting it for a non-allowlisted model is refused (see ``vlm_runtime``).
        """
        self.model_name = model_name
        self.multimodal = (
            multimodal
            if multimodal is not None
            else self._infer_multimodal(model_name, provider_name, api_base)
        )
        self.use_llm = use_llm
        self.allow_fallback = allow_fallback
        self.prompt_version = prompt_version
        self.provider_name = provider_name
        self.api_base = api_base
        self.timeout_seconds = timeout_seconds
        self.is_mock = is_mock
        self.llm_caller = llm_caller
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.trust_remote_code = trust_remote_code
        self.load_in_4bit = load_in_4bit
        self.precision = precision
        self.allow_cpu = allow_cpu
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels if max_pixels is not None else MAX_IMAGE_PIXELS
        self.min_vram_gib = min_vram_gib
        self._model = None
        self._processor = None
        self._image_config = None
        self._load_error: Optional[Exception] = None
        self._load_lock = threading.Lock()
        # Versions, loader class, dtype, quantization and image settings of the loaded model
        # (None until a real model is loaded); recorded in the run provenance.
        self.runtime_info: Optional[Dict[str, Any]] = None
        self._local = threading.local()
        self._local.last_provenance = None

    @staticmethod
    def _infer_multimodal(
        model_name: Optional[str],
        provider_name: Optional[str],
        api_base: Optional[str],
    ) -> bool:
        name = (model_name or "").lower()
        if provider_name == "ollama" or name.startswith("ollama/") or (api_base and "11434" in str(api_base)):
            return False
        return "-vl-" in name or name.endswith("-vl")

    @classmethod
    def for_local_ollama(
        cls,
        model_name: str = "ollama/qwen2.5:7b-instruct",
        api_base: str = "http://localhost:11434",
        use_llm: bool = True,
        allow_fallback: bool = False,
        timeout_seconds: float = 30.0,
        prompt_version: str = "1.0.0",
    ) -> "SchemaGuidedExtractor":
        """Factory creating a SchemaGuidedExtractor configured for local Ollama Qwen2.5-7B-Instruct."""
        return cls(
            model_name=model_name,
            use_llm=use_llm,
            allow_fallback=allow_fallback,
            prompt_version=prompt_version,
            provider_name="ollama",
            api_base=api_base,
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def from_settings(cls, settings: Any) -> "SchemaGuidedExtractor":
        """Factory creating a SchemaGuidedExtractor from Settings."""
        return cls(
            model_name=getattr(settings, "llm_model_name", "ollama/qwen2.5:7b-instruct"),
            use_llm=getattr(settings, "llm_use_inference", True),
            allow_fallback=getattr(settings, "llm_allow_fallback", False),
            provider_name=getattr(settings, "llm_provider", "ollama"),
            api_base=getattr(settings, "llm_api_base", "http://localhost:11434"),
            timeout_seconds=getattr(settings, "llm_timeout_seconds", 30.0),
        )

    @property
    def last_provenance(self) -> Optional[ExecutionProvenance]:
        local = getattr(self, "_local", None)
        if local is None:
            return None
        return getattr(local, "last_provenance", None)

    @last_provenance.setter
    def last_provenance(self, value: Optional[ExecutionProvenance]) -> None:
        if not hasattr(self, "_local"):
            self._local = threading.local()
        self._local.last_provenance = value

    def extract(
        self,
        payload: ModuleBOutputPayload,
        schema_cls: Optional[Type[BaseModel]] = None,
        page_images: Optional[Sequence[PageImageRef]] = None,
    ) -> BaseModel:
        """Legacy/default extraction interface returning BaseModel directly."""
        result = self.extract_with_provenance(payload, schema_cls=schema_cls, page_images=page_images)
        return result.payload

    def extract_with_provenance(
        self,
        payload: ModuleBOutputPayload,
        schema_cls: Optional[Type[BaseModel]] = None,
        page_images: Optional[Sequence[PageImageRef]] = None,
    ) -> ExtractionResult:
        """
        Extracts structured data and generates trusted execution provenance (SCH-007).
        Enforces strict no-fallback behavior when allow_fallback=False.

        ``page_images`` are the explicit Module A page-image references. A multimodal
        extractor (Qwen-VL) reads them directly and returns per-field grounding; text-only
        extractors (Ollama/litellm, heuristics) ignore them and use Module B markdown.
        """
        self.last_provenance = None

        doc_type = payload.classification.document_type.value
        target_schema = schema_cls or SCHEMA_MAP.get(doc_type)

        if not target_schema:
            raise ValueError(f"No extraction schema registered for document type: {doc_type}")

        # 1. Mock execution
        if self.is_mock:
            mock_model = self._extract_heuristic(payload, doc_type, target_schema)
            prov = ExecutionProvenance(
                mode=ExecutionMode.MOCK,
                model_name=self.model_name,
                provider="mock",
                prompt_version=self.prompt_version,
                fallback_occurred=False,
            )
            self.last_provenance = prov
            return ExtractionResult(payload=mock_model, provenance=prov)

        # 2. Genuine LLM requested
        if self.use_llm:
            provider = self.provider_name or (
                "ollama"
                if (
                    self.model_name.startswith("ollama/")
                    or (self.api_base and "11434" in str(self.api_base))
                )
                else "litellm"
            )
            try:
                grounding: Dict[str, FieldGrounding] = {}
                request_id: Optional[str] = None
                details: Dict[str, Any] = {}
                if page_images and self.multimodal and self.llm_caller is None:
                    # PRD v1.3/v1.5 1-stage path: Qwen-VL reads every explicit Module A page image.
                    model_result, grounding, provider, details = self._extract_vlm_internal(
                        target_schema, page_images
                    )
                else:
                    model_result, request_id = self._extract_llm_internal(payload, target_schema)
                prov = ExecutionProvenance(
                    mode=ExecutionMode.GENUINE_LLM,
                    model_name=self.model_name,
                    provider=provider,
                    prompt_version=self.prompt_version,
                    fallback_occurred=False,
                    request_id=request_id,
                    details=details,
                )
                self.last_provenance = prov
                return ExtractionResult(payload=model_result, provenance=prov, grounding=grounding)
            except Exception as exc:
                if not self.allow_fallback:
                    prov = ExecutionProvenance(
                        mode=ExecutionMode.GENUINE_LLM,
                        model_name=self.model_name,
                        provider=provider,
                        prompt_version=self.prompt_version,
                        fallback_occurred=False,
                        fallback_reason=str(exc),
                    )
                    self.last_provenance = prov
                    raise LLMExecutionError(
                        message=f"LLM extraction failed under strict no-fallback policy: {exc}",
                        model_name=self.model_name,
                        details={"cause": str(exc), "cause_type": type(exc).__name__},
                    ) from exc

                # Fallback enabled -> heuristic execution
                heuristic_model = self._extract_heuristic(payload, doc_type, target_schema)
                prov = ExecutionProvenance(
                    mode=ExecutionMode.HEURISTIC_FALLBACK,
                    model_name=self.model_name,
                    provider=provider,
                    prompt_version=self.prompt_version,
                    fallback_occurred=True,
                    fallback_reason=str(exc),
                )
                self.last_provenance = prov
                return ExtractionResult(payload=heuristic_model, provenance=prov)

        # 3. Direct heuristic execution (use_llm=False)
        heuristic_model = self._extract_heuristic(payload, doc_type, target_schema)
        prov = ExecutionProvenance(
            mode=ExecutionMode.HEURISTIC,
            model_name="heuristic_regex_v1",
            provider="regex_rules",
            prompt_version=self.prompt_version,
            fallback_occurred=False,
        )
        self.last_provenance = prov
        return ExtractionResult(payload=heuristic_model, provenance=prov)

    def _validate_dict(
        self,
        payload: ModuleBOutputPayload,
        schema_cls: Type[BaseModel],
        data: Dict[str, Any],
    ) -> BaseModel:
        """Types a dict result; an absent required field keeps the rest (see vlm_grounding)."""
        model, _missing = validate_allowing_missing(schema_cls, data)
        return model

    def _extract_llm_internal(
        self,
        payload: ModuleBOutputPayload,
        schema_cls: Type[BaseModel],
    ) -> Tuple[BaseModel, Optional[str]]:
        """Executes LLM extraction via custom caller, vLLM / Qwen-VL, or litellm."""
        if self.llm_caller is not None:
            res = self.llm_caller(payload, schema_cls)
            if isinstance(res, tuple) and len(res) == 2:
                model_inst, req_id = res
            else:
                model_inst, req_id = res, None

            if model_inst is None:
                raise ValueError("LLM caller returned None")

            if not isinstance(model_inst, schema_cls):
                if isinstance(model_inst, dict):
                    model_inst = self._validate_dict(payload, schema_cls, model_inst)
                else:
                    raise ValueError(f"LLM caller returned {type(model_inst).__name__}, expected {schema_cls.__name__}")

            request_id = str(req_id).strip() if (req_id is not None and str(req_id).strip()) else None
            return model_inst, request_id

        # Legacy single-page vision path (PRD v1.5 hand-off): Module B records the preprocessed page
        # image path on each page. It carries NO geometry, so no grounding/evidence can be produced
        # from it; callers that need evidence pass explicit Module A PageImageRefs (the authoritative
        # geometry source) to extract_with_provenance, which takes the _extract_vlm_internal path.
        first_img = getattr(payload.pages[0], "page_image_path", None) if payload.pages else None
        if self.multimodal and first_img and os.path.exists(first_img):
            raw_dict, _ = self.extract_page_image(first_img, schema_cls)
            return schema_cls.model_validate(_flat_to_nested(raw_dict)), None

        # Text LLM path via instructor / litellm (Ollama or other text models). The grounded
        # multimodal path is _extract_vlm_internal, which receives explicit Module A page-image
        # references.
        import instructor
        import litellm

        resolved_model = self.model_name
        is_ollama = (
            self.provider_name == "ollama"
            or resolved_model.startswith("ollama/")
            or (self.api_base and "11434" in str(self.api_base))
        )
        if is_ollama and not resolved_model.startswith("ollama/"):
            if resolved_model in ("Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"):
                resolved_model = "ollama/qwen2.5:7b-instruct"
            else:
                resolved_model = f"ollama/{resolved_model}"

        system_message = (
            "You are a schema-constrained document extraction engine. "
            "Extract only fields defined in the supplied Pydantic schema. "
            "Use only information explicitly present in the source document. "
            "Set absent or illegible scalar fields to null and absent collections to empty lists. "
            "Never infer, guess, calculate, or hallucinate values. "
            "Return strictly valid JSON matching the schema."
        )

        response_mode = (
            instructor.Mode.JSON if is_ollama else instructor.Mode.TOOLS
        )
        client = instructor.from_litellm(
            litellm.completion,
            mode=response_mode,
        )
        call_kwargs: Dict[str, Any] = {
            "model": resolved_model,
            # Requiredness-relaxed twin: a required field the model could not find must not
            # discard everything else it did extract (it is reported as missing downstream).
            "response_model": relaxed_schema(schema_cls),
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {"role": "user", "content": payload.combined_markdown},
            ],
            "temperature": 0.0,
            "_skip_mcp_handler": True,
        }
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.timeout_seconds:
            call_kwargs["timeout"] = self.timeout_seconds
        if is_ollama:
            call_kwargs["custom_llm_provider"] = "ollama"

        completion_res = client.chat.completions.create_with_completion(**call_kwargs)

        if isinstance(completion_res, tuple) and len(completion_res) == 2:
            model_inst, raw_completion = completion_res
        else:
            model_inst, raw_completion = completion_res, None

        if model_inst is None:
            raise ValueError("LLM provider returned empty completion")

        if not isinstance(model_inst, schema_cls):
            if isinstance(model_inst, dict):
                model_inst = self._validate_dict(payload, schema_cls, model_inst)
            else:
                raise ValueError(
                    f"LLM provider returned invalid object of type {type(model_inst).__name__}, expected {schema_cls.__name__}"
                )

        request_id = None
        if raw_completion is not None:
            raw_id = getattr(raw_completion, "id", None)
            if raw_id is not None and str(raw_id).strip():
                request_id = str(raw_id).strip()

        return model_inst, request_id

    def _extract_vlm_internal(
        self,
        schema_cls: Type[BaseModel],
        page_images: Sequence[PageImageRef],
    ) -> Tuple[BaseModel, Dict[str, FieldGrounding], str, Dict[str, Any]]:
        """
        Multimodal extraction over every supplied Module A page image, in document order.

        Each page image is sent to the VLM (``extract_page_image``) on its own; the per-page
        results are aggregated schema-aware (see ``vlm_grounding.aggregate_leaves``) into one
        typed schema instance plus per-field, page-specific grounding.
        Returns (model, grounding, provider, details) where ``details`` records the pages
        attempted / processed / failed / empty for coverage reporting.
        """
        processed, details, provider = self._read_pages(
            page_images,
            schema_cls=schema_cls,
            system_prompt=None,
            collect=lambda raw: collect_schema_leaves(raw, schema_cls),
            validate=lambda leaves: validate_page_leaves(schema_cls, leaves),
        )
        # Required fields absent from every page do not discard what WAS extracted: they are
        # recorded (details + per-field validation + REQUIRED_FIELD_MISSING flag downstream).
        # Malformed values still raise.
        model, grounding, missing = build_partial_model_from_pages(processed, schema_cls)
        details["required_fields_missing"] = missing
        if self.runtime_info:
            details["runtime"] = self.runtime_info
        return model, grounding, provider, details

    def extract_generic_pages(
        self,
        page_images: Sequence[PageImageRef],
        system_prompt: str,
    ) -> GenericExtractionResult:
        """
        Schema-less multimodal extraction (unstructured / handwritten categories) over every
        supplied page image, aggregated with the same page-aware rules as the schema-bound path.
        """
        processed, details, provider = self._read_pages(
            page_images,
            schema_cls=None,
            system_prompt=system_prompt,
            collect=collect_generic_leaves,
            validate=None,
        )
        values, grounding = aggregate_leaves(processed)
        if self.runtime_info:
            details["runtime"] = self.runtime_info
        prov = ExecutionProvenance(
            mode=ExecutionMode.GENUINE_LLM,
            model_name=self.model_name,
            provider=provider,
            prompt_version=self.prompt_version,
            fallback_occurred=False,
            details=details,
        )
        self.last_provenance = prov
        return GenericExtractionResult(values=values, provenance=prov, grounding=grounding)

    def _read_pages(
        self,
        page_images: Sequence[PageImageRef],
        schema_cls: Optional[Type[BaseModel]],
        system_prompt: Optional[str],
        collect: Callable[[Dict[str, Any]], Dict[str, PageLeaf]],
        validate: Optional[Callable[[Dict[str, PageLeaf]], None]],
    ) -> Tuple[List[Tuple[int, Dict[str, PageLeaf]]], Dict[str, Any], str]:
        """
        Runs the VLM once per page image (document order, one image per page number). A page
        failure - missing image file, VLM error, blank/non-JSON output, or values that violate the
        schema - is recorded and does NOT discard the other pages. A page whose processing
        succeeded but yielded no values is *processed (empty)*, not failed. Raises only when no
        page could be processed at all.
        """
        unique: Dict[int, PageImageRef] = {}
        for ref in page_images:
            unique.setdefault(ref.page_number, ref)
        ordered = [unique[n] for n in sorted(unique)]
        multi = len(ordered) > 1

        processed: List[Tuple[int, Dict[str, PageLeaf]]] = []
        failed: List[Dict[str, Any]] = []
        empty: List[int] = []
        provider = "qwen_vl"

        for ref in ordered:
            try:
                image_path = Path(ref.image_path)
                if not image_path.is_file():
                    raise FileNotFoundError(f"page image does not exist: {image_path}")
                raw, vlm_prov = self.extract_page_image(
                    str(image_path),
                    schema_cls,
                    system_prompt=system_prompt,
                    user_text=self._page_user_text(ref.page_number, len(ordered)) if multi else None,
                )
                provider = vlm_prov.provider or provider
                if not isinstance(raw, dict):
                    raise ValueError(f"model output is not a JSON object (got {type(raw).__name__})")
                leaves = collect(raw)
                if validate is not None:
                    validate(leaves)
            except Exception as exc:
                failed.append({"page": ref.page_number, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            processed.append((ref.page_number, leaves))
            if not leaves:
                empty.append(ref.page_number)

        if not processed:
            reasons = "; ".join(f"page {f['page']}: {f['reason']}" for f in failed) or "no page images supplied"
            raise ValueError(f"No page could be processed by the VLM ({len(ordered)} attempted): {reasons}")

        details = {
            "input_modality": "image",
            "pages_attempted": [ref.page_number for ref in ordered],
            "pages_processed": [n for n, _ in processed],
            "pages_failed": failed,
            "pages_empty": empty,
        }
        return processed, details, provider

    @staticmethod
    def _page_user_text(page_number: int, total_pages: int) -> str:
        return (
            f"This image is page {page_number} of a {total_pages}-page document that is read one page "
            "at a time. Transcribe only the values visible on THIS page into the requested JSON "
            "schema; use null for anything that is not on this page."
        )

    def _extract_llm(self, payload: ModuleBOutputPayload, schema_cls: Type[BaseModel]) -> BaseModel:
        """Backward-compatible internal delegate."""
        try:
            model_inst, _ = self._extract_llm_internal(payload, schema_cls)
            return model_inst
        except Exception as exc:
            if not self.allow_fallback:
                raise LLMExecutionError(
                    message=f"LLM extraction failed under strict no-fallback policy: {exc}",
                    model_name=self.model_name,
                    details={"cause": str(exc), "cause_type": type(exc).__name__},
                ) from exc
            return self._extract_heuristic(payload, payload.classification.document_type.value, schema_cls)

    def extract_page_image(
        self,
        image_input: Union[str, Path, bytes],
        schema_cls: Optional[Type[BaseModel]],
        system_prompt: Optional[str] = None,
        user_text: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], ExecutionProvenance]:
        """Direct 1-stage Qwen-VL multimodal extraction from page image.

        Either ``schema_cls`` (schema-bound prompt) or an explicit ``system_prompt``
        (category-specific prompt from doc_extractor.routing) must be supplied.
        """
        if system_prompt is None:
            if schema_cls is None:
                raise ValueError("extract_page_image requires schema_cls or system_prompt")
            system_prompt = self._build_system_prompt(schema_cls)
        user_text = user_text or (
            "Carefully transcribe all real filled values from this document image into the requested JSON schema. "
            "Read every handwritten and printed value directly from the image; use null for anything not visible."
        )

        vllm_api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("VLLM_API_BASE")
        if vllm_api_base:
            return self._extract_via_vllm_api(image_input, schema_cls, vllm_api_base, system_prompt=system_prompt)

        self._local.score_fields = True        # FR-CNF-001: ask _generate_from_image for token probabilities
        try:
            output_text = self._generate_from_image(image_input, system_prompt, user_text, self.max_new_tokens)
        finally:
            self._local.score_fields = False
        raw_dict = self._parse_json_output(output_text)
        _, spans = getattr(self._local, "last_token_spans", ("", []))
        attach_field_confidence(raw_dict, output_text, spans)

        prov = ExecutionProvenance(
            mode=ExecutionMode.GENUINE_LLM,
            model_name=self.model_name,
            provider="transformers_qwen_vl",
            prompt_version=self.prompt_version,
            fallback_occurred=False,
        )
        self.last_provenance = prov
        return raw_dict, prov

    def read_text_crop(self, image) -> str:
        """Transcribe one small cropped region (used by the crop-verify pass); returns plain text."""
        import io

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")   # _generate_from_image takes a path or encoded bytes
        text = self._generate_from_image(
            buf.getvalue(),
            "You are an OCR engine. Output only the text in the image, exactly as written, on one line. "
            "No explanation, no quotes.",
            "Transcribe the text.",
            max_new_tokens=48,
        )
        return text.strip().strip('"').strip()

    def classify_page_image(self, image_input: Union[str, Path, bytes]) -> Dict[str, Any]:
        """Ask the VLM for document category, language and type (FR-CLS-005, FR-CLS-007).

        Returns a dict with keys category, language, document_type, subtype, confidence.
        Raises on model or parse failure so the caller can fall back to heuristics.
        """
        from doc_extractor.routing import CLASSIFIER_SYSTEM_PROMPT

        text = self._generate_from_image(
            image_input,
            CLASSIFIER_SYSTEM_PROMPT,
            "Classify this document image. Return the JSON object only.",
            max_new_tokens=128,
            max_pixels=CLASSIFY_MAX_PIXELS,
        )
        return self._parse_json_output(text)

    @staticmethod
    def _parse_json_output(output_text: str) -> Dict[str, Any]:
        if not (output_text or "").strip():
            # A blank reply is a failed page, not an empty one; `{}` is a valid empty page.
            raise ValueError("empty model response")
        return parse_json_tolerant(output_text)

    def _generate_from_image(
        self,
        image_input: Union[str, Path, bytes],
        system_prompt: str,
        user_text: str,
        max_new_tokens: int,
        max_pixels: Optional[int] = None,
    ) -> str:
        """
        One Qwen-VL generation over one page image, following the official Qwen3-VL flow:
        qwen-vl-utils resizes the image once (aspect ratio kept, rounded to the 16-px-patch factor,
        within the explicit ``min_pixels`` / ``max_pixels``) and the processor is told NOT to resize
        again (``do_resize=False``). Boxes the model returns are relative to the image it saw, so they
        stay valid for the Module A geometry (see ``vlm_grounding``).
        """
        model, processor = self._load_model()
        family = vlm_runtime.resolve_family(self.model_name)
        if max_pixels is not None:
            # per-call cap (classification uses the smaller CLASSIFY_MAX_PIXELS image)
            floor = self.min_pixels if self.min_pixels is not None and self.min_pixels <= max_pixels else None
            image_config = vlm_runtime.build_image_config(family, floor, max_pixels)
        else:
            image_config = self._image_config or vlm_runtime.build_image_config(
                family, self.min_pixels, self.max_pixels
            )

        if isinstance(image_input, (str, Path)):
            image: Any = str(image_input)
        else:
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(image_input))
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    vlm_runtime.image_message_element(image, image_config),
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        try:
            import qwen_vl_utils
        except ImportError as exc:
            raise vlm_runtime.VLMRuntimeError(
                "qwen-vl-utils is not installed. Install the pinned baseline: "
                f"pip install qwen-vl-utils=={vlm_runtime.QWEN_VL_UTILS_BASELINE} (needs torchvision, which "
                "comes with your PyTorch install).",
                code="VLM_DEPENDENCY_MISSING",
            ) from exc
        import torch

        process_vision_info = vlm_runtime.vision_info_function(qwen_vl_utils, family)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=image_config.image_patch_size)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            do_resize=False,          # qwen-vl-utils has already resized: never resize twice
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        # Token probabilities only when extract_page_image asked for them and the model supports them;
        # every other call (classification, fakes in tests) keeps the plain generate path.
        score_fields = bool(getattr(self._local, "score_fields", False)) and hasattr(model, "compute_transition_scores")
        self._local.last_token_spans = ("", [])
        if score_fields:
            with torch.no_grad():
                generated = model.generate(**inputs, **gen_kwargs, return_dict_in_generate=True, output_scores=True)
            decode = lambda ids: processor.batch_decode(  # noqa: E731
                [ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            text_out, spans = token_confidences(model, generated, inputs.input_ids.shape[1], decode)
            self._local.last_token_spans = (text_out, spans)
            del inputs, generated
        else:
            with torch.no_grad():
                generated_ids = model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            text_out = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            del inputs, generated_ids, generated_ids_trimmed
        free_gpu_memory()  # gc + torch.cuda.empty_cache() once after every generation
        return text_out

    def _extract_via_vllm_api(
        self,
        image_input: Union[str, Path, bytes],
        schema_cls: Optional[Type[BaseModel]],
        api_base: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], ExecutionProvenance]:
        """Calls vLLM OpenAI-compatible vision completions endpoint."""
        if isinstance(image_input, (str, Path)):
            with open(image_input, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")
        else:
            b64_img = base64.b64encode(image_input).decode("utf-8")

        prompt = system_prompt or self._build_system_prompt(schema_cls)
        req_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                        {"type": "text", "text": "Extract all fields from this document page."},
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "logprobs": True,
        }

        url = f"{api_base.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(req_body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            lp_content = ((data["choices"][0].get("logprobs") or {}).get("content")) or []

        cleaned = re.sub(r"^```json\s*", "", content.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        if not cleaned:
            raise ValueError("empty model response")
        raw_dict = json.loads(cleaned)

        prov = ExecutionProvenance(
            mode=ExecutionMode.GENUINE_LLM,
            model_name=self.model_name,
            provider="vllm_openai_api",
            prompt_version=self.prompt_version,
            fallback_occurred=False,
        )
        self.last_provenance = prov
        return raw_dict, prov

    def _load_model(self):
        """
        Lazily loads the Qwen-VL model and processor via ``vlm_runtime.load_qwen_vl``: the exact
        Transformers class for the checkpoint, hardware-aware dtype, verified 4-bit settings and an
        actionable ``VLMRuntimeError`` for every incompatibility (no silent fallback to a generic
        text-only loader, another dtype or non-quantized loading). A model/processor injected into
        ``self._model`` / ``self._processor`` (unit tests) is used as is.
        Uses double-checked locking to protect against duplicate concurrent initialization (NFR-006).
        """
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        with self._load_lock:
            if self._model is None or self._processor is None:
                if self._load_error is not None:
                    raise self._load_error
                try:
                    loaded = vlm_runtime.load_qwen_vl(vlm_runtime.VLMLoadConfig(
                        model_id=self.model_name,
                        load_in_4bit=getattr(self, "load_in_4bit", True),
                        precision=self.precision,
                        allow_cpu=self.allow_cpu,
                        device_map=self.device,
                        trust_remote_code=self.trust_remote_code,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                        min_vram_gib=self.min_vram_gib,
                    ))
                except vlm_runtime.VLMRuntimeError as exc:
                    # Deterministic compatibility problems are remembered so a document does not retry an
                    # expensive load once per page; an out-of-memory failure may be retried after freeing memory.
                    if exc.code != "VLM_INSUFFICIENT_VRAM":
                        self._load_error = exc
                    raise
                # Publish supporting processor, image configuration, and runtime info first
                self._processor = loaded.processor
                self._image_config = loaded.image_config
                self.runtime_info = loaded.runtime_info
                # Publish model sentinel reference last
                self._model = loaded.model
        return self._model, self._processor

    def _build_system_prompt(self, schema_cls: Type[BaseModel]) -> str:
        """Constructs schema-bound system prompt with full leaf field paths."""
        def get_leaf_fields(cls: Type[BaseModel], prefix: str = "") -> List[Tuple[str, str]]:
            fields: List[Tuple[str, str]] = []
            for name, info in cls.model_fields.items():
                full_name = f"{prefix}.{name}" if prefix else name
                ann = info.annotation
                args = getattr(ann, "__args__", ())
                target_cls = None
                if isinstance(ann, type) and issubclass(ann, BaseModel):
                    target_cls = ann
                elif args:
                    for a in args:
                        if isinstance(a, type) and issubclass(a, BaseModel):
                            target_cls = a
                            break

                if target_cls:
                    # Repeating groups (List[Model]) are listed with an explicit entry index so
                    # the model can report every entry (nominees.0.name, nominees.1.name, ...).
                    group_prefix = f"{full_name}.0" if get_origin(ann) in (list, List) else full_name
                    fields.extend(get_leaf_fields(target_cls, group_prefix))
                else:
                    desc = info.description or ""
                    fields.append((full_name, desc))
            return fields

        leaf_fields = get_leaf_fields(schema_cls)
        field_lines: List[str] = []
        for field_name, desc in leaf_fields:
            field_lines.append(
                f'  "{field_name}": {{"value": "<extracted_string_or_null>", "bbox": [x1, y1, x2, y2]}}  # {desc}'
            )
        fields_json = ",\n".join(field_lines)
        repeating_rule = ""
        if any(re.search(r"\.0(\.|$)", name) for name, _ in leaf_fields):
            repeating_rule = (
                "6. REPEATING GROUPS: fields written with index 0 (for example nominees.0.name) repeat. "
                "Report every entry visible on this page with consecutive indices (nominees.0.*, "
                "nominees.1.*, ...). Do not merge entries and do not invent entries that are not visible.\n"
            )

        return f"""You are a high-precision document OCR and extraction engine specialized in Indian BFSI forms.

TASK:
Examine this document image with extreme precision. Extract the actual handwritten and printed text corresponding to the target fields.

TARGET SCHEMA FIELDS:
{fields_json}

CRITICAL RULES:
1. TRANSCRIBE ACTUAL INK: Read the real handwritten and printed letters on the page (character boxes, handwriting, checkboxes). Do NOT invent, assume, or hallucinate placeholder names (e.g. NEVER output dummy names like "GOPAL SINGH" or dummy PANs like "ABCD12345C").
2. CHARACTER BOXES: For names, PAN, Aadhaar, read the individual letters written in the character grid boxes carefully (e.g. R-U-P-E-S-H -> RUPESH).
3. ABSENT FIELDS: If a field is not filled or not present on this document, set "value": null and "bbox": null.
4. BOUNDING BOXES: "bbox" must be [x1, y1, x2, y2] (left, top, right, bottom) in normalized 0-1000 scale tightly surrounding the exact handwritten/printed value (not the label) or the ticked box.
5. RETURN EVERY KEY listed above, even when the value is null.
6. NAMES SPLIT INTO ROWS: if the form has separate First / Middle / Last name rows, fill first_name, middle_name and last_name, and set full_name to all three joined with single spaces.
7. CHECKBOXES / TICKS: a ticked or marked option is the value (e.g. account_type "SAVINGS" if the Savings box is ticked); for boolean fields use true if ticked, false if clearly unticked, null if not on the form.
8. OTHER FILLED FIELDS: also add every other filled-in field on the page that is not in the list above, as "other.<snake_case_label>": {{"value": ..., "bbox": [x1, y1, x2, y2]}}.
9. STRICT FORMAT: Return raw valid JSON only. No markdown fences.
{repeating_rule}"""

    def _extract_heuristic(
        self,
        payload: ModuleBOutputPayload,
        doc_type: str,
        schema_cls: Type[BaseModel],
    ) -> BaseModel:
        """Deterministic regex heuristic fast-path (offline / test double)."""
        md = payload.combined_markdown
        extracted_dict: Dict[str, Any] = {}

        if doc_type == "DOC-01":
            applicant: Dict[str, Any] = {}
            account_details: Dict[str, Any] = {}

            name_match = re.search(r"(?:name|applicant name)\s*[:\-]?\s*([A-Z\s]{3,30}?)(?:\n|$)", md, re.IGNORECASE)
            if name_match:
                applicant["full_name"] = name_match.group(1).strip()

            pan_match = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", md)
            if pan_match:
                applicant["pan_card_number"] = pan_match.group(1).strip()

            dob_match = re.search(r"(?:dob|date of birth|birth date)\s*[:\-]?\s*([0-9\/\-\.]+)", md, re.IGNORECASE)
            if dob_match:
                applicant["date_of_birth"] = dob_match.group(1).strip()

            phone_match = re.search(r"(?:phone|mobile|contact)\s*[:\-]?\s*(\+?[0-9\s\-]{10,14})", md, re.IGNORECASE)
            if phone_match:
                applicant["phone_number"] = phone_match.group(1).strip()

            opt_in: Dict[str, Any] = {}
            if "[x] cheque book" in md.lower() or "cheque book: [x]" in md.lower():
                opt_in["cheque_book"] = True
            elif "[ ] cheque book" in md.lower() or "cheque book: [ ]" in md.lower():
                opt_in["cheque_book"] = False

            if "[x] debit card" in md.lower() or "debit card: [x]" in md.lower():
                opt_in["debit_card"] = True
            elif "[ ] debit card" in md.lower() or "debit card: [ ]" in md.lower():
                opt_in["debit_card"] = False

            if "[x] internet banking" in md.lower() or "internet banking: [x]" in md.lower():
                opt_in["internet_banking"] = True
            elif "[ ] internet banking" in md.lower() or "internet banking: [ ]" in md.lower():
                opt_in["internet_banking"] = False

            account_details["opt_in_services"] = opt_in

            deposit_match = re.search(r"(?:initial deposit|deposit amount)\s*[:\-]?\s*(?:inr|rs\.?)?\s*([0-9,\.]+)", md, re.IGNORECASE)
            if deposit_match:
                cleaned_num = deposit_match.group(1).replace(",", "").strip()
                try:
                    account_details["initial_deposit_amount"] = float(cleaned_num)
                except ValueError:
                    pass

            extracted_dict["applicant"] = applicant
            extracted_dict["account_details"] = account_details

        elif doc_type == "DOC-02":
            proposer_match = re.search(r"(?:proposer name|insured name|proposer)\s*[:\-]?\s*([A-Za-z\s]{3,30}?)(?:\n|$)", md, re.IGNORECASE)
            if proposer_match:
                extracted_dict["proposer_name"] = proposer_match.group(1).strip()

            sum_insured_match = re.search(r"(?:sum insured|sum assured)\s*[:\-]?\s*(?:inr|rs\.?)?\s*([0-9,\.]+)", md, re.IGNORECASE)
            if sum_insured_match:
                cleaned_num = sum_insured_match.group(1).replace(",", "").strip()
                try:
                    extracted_dict["sum_insured"] = float(cleaned_num)
                except ValueError:
                    pass

            term_match = re.search(r"(?:policy term|term)\s*[:\-]?\s*([0-9]{1,2})\s*(?:years|yrs)?", md, re.IGNORECASE)
            if term_match:
                try:
                    extracted_dict["policy_term_years"] = int(term_match.group(1))
                except ValueError:
                    pass

            nominees: List[Dict[str, Any]] = []
            nominee_matches = re.finditer(r"nominee\s*[:\-]?\s*([A-Za-z\s]+?)\s*\(([^)]+)\)(?:\s*,?\s*(\d+)%)?", md, re.IGNORECASE)
            for n_match in nominee_matches:
                nominees.append({
                    "name": n_match.group(1).strip(),
                    "relationship": n_match.group(2).strip(),
                    "share_percentage": float(n_match.group(3)) if n_match.group(3) else 100.0,
                })
            extracted_dict["nominees"] = nominees

            med_dec: Dict[str, Any] = {}
            if "[x] diabetes" in md.lower() or "diabetes: [x]" in md.lower():
                med_dec["has_diabetes"] = True
            elif "[ ] diabetes" in md.lower() or "diabetes: [ ]" in md.lower():
                med_dec["has_diabetes"] = False

            if "[x] hypertension" in md.lower() or "hypertension: [x]" in md.lower():
                med_dec["has_hypertension"] = True
            elif "[ ] hypertension" in md.lower() or "hypertension: [ ]" in md.lower():
                med_dec["has_hypertension"] = False

            if "[x] tobacco" in md.lower() or "tobacco: [x]" in md.lower():
                med_dec["has_tobacco_use"] = True
            elif "[ ] tobacco" in md.lower() or "tobacco: [ ]" in md.lower():
                med_dec["has_tobacco_use"] = False

            extracted_dict["medical_declaration"] = med_dec

        elif doc_type == "DOC-03":
            patient_match = re.search(r"(?:patient name|patient)\s*[:\-]?\s*([A-Za-z\s]{3,30}?)(?:\n|$)", md, re.IGNORECASE)
            if patient_match:
                extracted_dict["patient_name"] = patient_match.group(1).strip()

            ip_match = re.search(r"(?:ip no|ip number|admission no|inpatient no)\s*[:\-]?\s*([A-Za-z0-9\/\-]+)", md, re.IGNORECASE)
            if ip_match:
                extracted_dict["ip_number"] = ip_match.group(1).strip()

            adm_match = re.search(r"(?:admission date|date of admission|admitted)\s*[:\-]?\s*([0-9\/\-\.A-Za-z]+)", md, re.IGNORECASE)
            if adm_match:
                extracted_dict["admission_date"] = adm_match.group(1).strip()

            dis_match = re.search(r"(?:discharge date|date of discharge|discharged)\s*[:\-]?\s*([0-9\/\-\.A-Za-z]+)", md, re.IGNORECASE)
            if dis_match:
                extracted_dict["discharge_date"] = dis_match.group(1).strip()

            diag_match = re.search(r"(?:diagnosis|final diagnosis)\s*[:\-]?\s*([A-Za-z0-9\s,\-\.]+?)(?:\n|$)", md, re.IGNORECASE)
            if diag_match:
                extracted_dict["diagnosis"] = diag_match.group(1).strip()

            medications: List[Dict[str, Any]] = []
            med_lines = re.findall(r"(?:rx|medication|tab|cap|inj)\.?\s*([A-Za-z0-9\s]+?)\s+(\d+(?:mg|ml|g))\s+([A-Z]{2,3})\s+(?:for\s+)?(\d+\s*(?:days|weeks))", md, re.IGNORECASE)
            for m in med_lines:
                medications.append({
                    "medicine_name": m[0].strip(),
                    "dosage": m[1].strip(),
                    "frequency": m[2].strip().upper(),
                    "duration": m[3].strip(),
                })
            extracted_dict["medications"] = medications

        if hasattr(schema_cls, "__field_metadata__"):
            from schema_lifecycle.compiler import DynamicSchemaCompiler
            meta_dict = getattr(schema_cls, "__field_metadata__", {})
            for fpath, meta in meta_dict.items():
                if "." not in fpath and "[]" not in fpath and fpath not in extracted_dict:
                    extracted_dict[fpath] = None
            nested = DynamicSchemaCompiler.unflatten(extracted_dict)
            try:
                return schema_cls.model_validate(nested)
            except Exception:
                return schema_cls.model_construct(**nested)

        return schema_cls.model_validate(extracted_dict)
