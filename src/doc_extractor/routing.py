"""
Category-based model routing (PRD v1.4: D7, D8, D10; FR-CLS-005..007, FR-RTE-001..004).

Flow: classify category (+ language) -> look up model in a versioned routing table ->
run that model behind the shared SchemaGuidedExtractor interface -> fall back to the
default model if the routed one fails (FR-RTE-003).

The routing table is configuration. Override it with a JSON file path in the
DOC_ROUTING_TABLE environment variable; shape is the same as DEFAULT_ROUTING_TABLE.
"""

from enum import Enum
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from doc_extractor.contracts_c import ExecutionProvenance

logger = logging.getLogger("doc_extractor.routing")

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class DocumentCategory(str, Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    HANDWRITTEN = "handwritten"


UNSTRUCTURED_SUBTYPES = [
    "brochure", "pamphlet", "menu card", "CASA leaflet", "invitation card", "other",
]
LANGUAGES = {
    "English": "en", "Hindi": "hi", "Telugu": "te", "Tamil": "ta", "Malayalam": "ml",
}

# Until the model evaluation workbook (FR-EVL-001) picks specialists, every route uses the
# default model. Change a row here (or in the JSON override) to route a category elsewhere.
DEFAULT_ROUTING_TABLE: Dict[str, Any] = {
    "version": "1.0.0",
    "default_model": DEFAULT_MODEL,
    "routes": {
        "structured": {"default": DEFAULT_MODEL},
        "unstructured": {"default": DEFAULT_MODEL},
        "handwritten": {"default": DEFAULT_MODEL},
    },
    # A route value is a model name (VLM) or a dict such as
    # {"adapter": "ocr_text", "ocr_model": "...", "text_model": "..."}.
    # alt_engines are the specialists the demo can switch to for comparison (FR-RTE-007).
    "alt_engines": {
        # text step must be a pure text, non-thinking model (no VLM, no thinking mode)
        "ocr_text": {"adapter": "ocr_text", "ocr_model": "zai-org/GLM-OCR", "text_model": "Qwen/Qwen3-4B-Instruct-2507"},
        "ocr_text_7b": {"adapter": "ocr_text", "ocr_model": "zai-org/GLM-OCR", "text_model": "Qwen/Qwen2.5-7B-Instruct"},
    },
}


def load_routing_table() -> Dict[str, Any]:
    path = os.environ.get("DOC_ROUTING_TABLE")
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_ROUTING_TABLE


def select_route(category: DocumentCategory, language: str, table: Optional[Dict[str, Any]] = None) -> Any:
    """FR-RTE-001/002: route keyed by (category, language). Returns a model name or an adapter dict."""
    table = table or load_routing_table()
    cat_routes = table.get("routes", {}).get(category.value, {})
    return cat_routes.get(language) or cat_routes.get("default") or table.get("default_model", DEFAULT_MODEL)


def select_model(category: DocumentCategory, language: str, table: Optional[Dict[str, Any]] = None) -> str:
    """Model name of the route (for an adapter dict, its label)."""
    route = select_route(category, language, table)
    if isinstance(route, dict):
        return route.get("model") or " + ".join(filter(None, [route.get("ocr_model"), route.get("text_model")]))
    return route


# ---------------------------------------------------------------------------
# Classification (FR-CLS-005, FR-CLS-007)
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = """You classify document images. Return raw JSON only, no markdown:
{"category": "structured|unstructured|handwritten",
 "document_type": "DOC-01|DOC-02|DOC-03|DOC-04|DOC-05|DOC-06|DOC-07|DOC-08|OTHER",
 "subtype": "brochure|pamphlet|menu card|CASA leaflet|invitation card|other|null",
 "language": "en|hi|te|ta|ml",
 "confidence": 0.0-1.0,
 "rotation": 0|90|180|270}

Definitions:
- structured: forms with labelled blanks or boxes to fill (KYC, CASA account opening DOC-01, insurance application DOC-02, loan DOC-04, credit card DOC-05, claim DOC-06).
- unstructured: free-layout printed material with no fixed fields (brochures, pamphlets, menu cards, invitation cards, policy wordings DOC-08, discharge summaries DOC-03).
- handwritten: the content is mostly handwriting, e.g. a doctor's prescription DOC-07.
A structured form with some handwritten entries is still structured.
"language" is the dominant language of the text. Use null for subtype unless the category is unstructured.
"rotation" is how many degrees CLOCKWISE the page must be turned so the text reads upright (0 if already upright, 180 if upside down)."""

_FILENAME_HINTS: List[Tuple[Tuple[str, ...], DocumentCategory, Optional[str]]] = [
    (("prescription", "rx"), DocumentCategory.HANDWRITTEN, "DOC-07"),
    (("brochure", "pamphlet", "menu", "invitation", "leaflet"), DocumentCategory.UNSTRUCTURED, None),
    (("discharge", "hospital", "patient", "summary", "medical"), DocumentCategory.UNSTRUCTURED, "DOC-03"),
    (("insurance", "policy", "nominee", "proposal"), DocumentCategory.STRUCTURED, "DOC-02"),
    (("bank", "casa", "account", "saving", "canara", "kyc", "form"), DocumentCategory.STRUCTURED, "DOC-01"),
]


def heuristic_classify(filename: str, rule_type: Optional[str] = None) -> Dict[str, Any]:
    """Offline fallback used when no VLM is available or classification fails."""
    name = filename.lower()
    for keys, cat, doc_type in _FILENAME_HINTS:
        if any(k in name for k in keys):
            return {"category": cat.value, "document_type": doc_type, "subtype": None,
                    "language": "en", "confidence": 0.5, "method": "filename_heuristic"}
    if rule_type in ("DOC-01", "DOC-02"):
        return {"category": "structured", "document_type": rule_type, "subtype": None,
                "language": "en", "confidence": 0.6, "method": "rule_classifier"}
    if rule_type == "DOC-03":
        return {"category": "unstructured", "document_type": rule_type, "subtype": None,
                "language": "en", "confidence": 0.6, "method": "rule_classifier"}
    return {"category": "structured", "document_type": None, "subtype": None,
            "language": "en", "confidence": 0.3, "method": "default"}


# ---------------------------------------------------------------------------
# Category prompts
# ---------------------------------------------------------------------------

_COMMON_RULES = """CRITICAL RULES:
1. TRANSCRIBE ONLY WHAT IS VISIBLE. Never invent or copy placeholder/example values. If a value is absent or illegible set "value": null and "bbox": null.
2. "bbox" is [x1, y1, x2, y2] (left, top, right, bottom) on a 0-1000 normalized scale around the exact text.
3. Return raw valid JSON only. No markdown fences."""


def build_generic_prompt(category: DocumentCategory, subtype: Optional[str] = None, language: str = "en") -> str:
    """Prompt for documents without a bound schema (unstructured, handwritten)."""
    if category == DocumentCategory.HANDWRITTEN:
        task = (
            "This document is handwritten (for example a doctor's prescription). Read the handwriting carefully. "
            "Extract key facts such as patient_name, age, date, doctor_name, diagnosis, and a list of medicines "
            "with dosage and frequency (OD, BD, TDS, SOS). Keep original shorthand. If a word is uncertain "
            "return your best reading and set \"uncertain\": true on that field."
        )
    elif category == DocumentCategory.STRUCTURED:
        task = (
            "This is a form with labelled fields. Extract every filled field as label -> value, using the "
            "printed label as a snake_case field name (character-box entries joined into one string, "
            "ticked checkboxes as true, unticked as false). Do not invent fields that are not on the form."
        )
    else:
        sub = f" Document sub-type: {subtype}." if subtype and subtype != "other" else ""
        task = (
            "This is free-layout printed material (brochure, pamphlet, menu card, invitation card, policy wording)."
            f"{sub} Extract the key facts it states: title, organisation or issuer, dates, contact details, "
            "prices or amounts, items or offers, and any other clearly stated facts."
        )
    return f"""You are a high-precision document reading engine. Document language: {language}.

TASK:
{task}

OUTPUT FORMAT:
{{"<snake_case_field_name>": {{"value": "<text, number, list or null>", "bbox": [x1, y1, x2, y2]}}, ...}}

{_COMMON_RULES}
"""


# ---------------------------------------------------------------------------
# Routed extraction with fallback (FR-RTE-001, FR-RTE-003, FR-RTE-004)
# ---------------------------------------------------------------------------

class RoutedExtractor:
    """Caches one SchemaGuidedExtractor per model and runs the routed one with fallback."""

    def __init__(self, table: Optional[Dict[str, Any]] = None, **extractor_kwargs: Any) -> None:
        self.table = table or load_routing_table()
        self._extractor_kwargs = extractor_kwargs
        self._cache: Dict[str, Any] = {}

    def get_extractor(self, model_name: str, device: Optional[str] = None):
        from doc_extractor.extractor import SchemaGuidedExtractor

        if model_name not in self._cache:
            kwargs = dict(self._extractor_kwargs)
            if device:
                kwargs["device"] = device
            self._cache[model_name] = SchemaGuidedExtractor(model_name=model_name, use_llm=True, **kwargs)
        return self._cache[model_name]

    @property
    def router_model(self) -> str:
        """Model that classifies the document. DOC_ROUTER_MODEL (e.g. Qwen/Qwen3-VL-4B-Instruct) or the
        table's "router_model" picks a smaller, faster router; default is the default extractor."""
        return (os.environ.get("DOC_ROUTER_MODEL") or self.table.get("router_model")
                or self.table.get("default_model", DEFAULT_MODEL))

    def get_router(self):
        default = self.table.get("default_model", DEFAULT_MODEL)
        if self.router_model == default:
            return self.get_extractor(default)
        device = None
        try:  # a separate router goes on the second GPU when there is one
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                device = "cuda:1"
        except Exception:
            pass
        return self.get_extractor(self.router_model, device=device)

    def classify(self, image_path: str, filename: str, rule_type: Optional[str] = None) -> Dict[str, Any]:
        """VLM classification with heuristic fallback. Always returns a dict with a valid category."""
        try:
            result = self.get_router().classify_page_image(image_path)
            cat = str(result.get("category", "")).lower()
            if cat not in {c.value for c in DocumentCategory}:
                raise ValueError(f"unrecognised category {cat!r}")
            result["category"] = cat
            result["method"] = "vlm"
            result["router_model"] = self.router_model
            return result
        except Exception as exc:  # classification must never block the demo
            logger.warning("VLM classification failed, using heuristic: %s", exc)
            result = heuristic_classify(filename, rule_type)
            result["fallback_reason"] = str(exc)
            return result

    def get_adapter(self, route: Any):
        """Build (and cache) the adapter for a route value: model name -> VLM, dict -> by 'adapter'."""
        from doc_extractor.adapters import OCRTextAdapter, VLMAdapter

        if isinstance(route, dict) and route.get("adapter") == "ocr_text":
            key = f"ocr_text::{route.get('ocr_model')}::{route.get('text_model')}"
            if key not in self._cache:
                kw = {k: route[k] for k in ("ocr_model", "text_model") if k in route}
                self._cache[key] = OCRTextAdapter(**kw)
            return self._cache[key]
        model_name = route.get("model") if isinstance(route, dict) else route
        return VLMAdapter(self.get_extractor(model_name))

    def extract(
        self,
        image_path: str,
        category: DocumentCategory,
        language: str = "en",
        schema_cls: Optional[Type[BaseModel]] = None,
        subtype: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], ExecutionProvenance, Dict[str, Any]]:
        """Returns (raw_fields, provenance, route_info).

        engine: None or "vlm" follows the routing table; another key (for example "ocr_text") picks
        that entry of table["alt_engines"] for side-by-side comparison.
        """
        default = self.table.get("default_model", DEFAULT_MODEL)
        if engine and engine != "vlm":
            alt = self.table.get("alt_engines", {}).get(engine)
            if alt is None:
                raise ValueError(f"unknown engine {engine!r}")
            routed = alt
        else:
            routed = select_route(category, language, self.table)

        default_extractor = self.get_extractor(default)
        if schema_cls is not None and category == DocumentCategory.STRUCTURED:
            prompt = default_extractor._build_system_prompt(schema_cls)
        else:
            prompt = build_generic_prompt(category, subtype, language)

        routed_label = select_model(category, language, {"routes": {category.value: {"default": routed}}})
        candidates = [routed] if routed == default else [routed, default]

        last_exc: Optional[Exception] = None
        for i, route in enumerate(candidates):
            try:
                adapter = self.get_adapter(route)
                raw, prov, meta = adapter.extract(image_path, prompt)
                if i > 0:
                    prov.fallback_occurred = True
                    prov.fallback_reason = f"routed model {routed_label} failed: {last_exc}"
                info = {
                    "document_category": category.value,
                    "language": language,
                    "routed_model": routed_label,
                    "model_used": adapter.model_label,
                    "adapter": adapter.kind,
                    "grounded": meta.get("grounded", True),
                    "models": meta.get("models", {}),
                    "fallback_occurred": i > 0,
                    "routing_table_version": self.table.get("version"),
                }
                if "ocr_text_preview" in meta:
                    info["ocr_text_preview"] = meta["ocr_text_preview"]
                return raw, prov, info
            except Exception as exc:
                logger.error("Extraction with %s failed: %s", route, exc)
                last_exc = exc
        raise last_exc  # type: ignore[misc]
