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

# One model routes and extracts. 4B fits a 16 GB T4 with room for the image; set
# DOC_DEFAULT_MODEL=Qwen/Qwen3-VL-8B-Instruct on a bigger GPU for higher accuracy.
DEFAULT_MODEL = os.environ.get("DOC_DEFAULT_MODEL", "Qwen/Qwen3-VL-4B-Instruct")


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
        # Digital printed pages: GLM-OCR reads the text, Qwen3-4B-Instruct-2507 (text only, non-thinking)
        # formats it into JSON. No boxes on this route, so its values are withheld and flagged (GR-002).
        "structured": {"default": {"adapter": "ocr_text", "ocr_model": "zai-org/GLM-OCR",
                                   "text_model": "Qwen/Qwen3-4B-Instruct-2507"}},
        "unstructured": {"default": {"adapter": "ocr_text", "ocr_model": "zai-org/GLM-OCR",
                                     "text_model": "Qwen/Qwen3-4B-Instruct-2507"}},
        # Handwriting: the vision model reads the page image directly.
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
{"filled_by": "hand|machine|empty",
 "category": "structured|unstructured|handwritten",
 "document_type": "DOC-01|DOC-02|DOC-03|DOC-04|DOC-05|DOC-06|DOC-07|DOC-08|<short descriptive name>",
 "subtype": "brochure|pamphlet|menu card|CASA leaflet|invitation card|other|null",
 "language": "en|hi|te|ta|ml",
 "confidence": 0.0-1.0,
 "rotation": 0|90|180|270}

First decide "filled_by": look at the values written INTO the document (names, dates, numbers, addresses), not the printed labels.
- "hand": the values are written by hand with a pen (ink strokes, uneven letters), even on a printed form.
- "machine": the values are typed or printed.
- "empty": nothing is filled in.

Definitions:
- structured: forms with labelled blanks or boxes to fill (KYC, CASA account opening DOC-01, insurance application DOC-02, loan DOC-04, credit card DOC-05, claim DOC-06).
- unstructured: free-layout printed material with no fixed fields (brochures, pamphlets, menu cards, invitation cards, policy wordings DOC-08, discharge summaries DOC-03).
- handwritten: the content is mostly handwriting, e.g. a doctor's prescription DOC-07.
If the filled-in entries on a form are handwritten, the category is HANDWRITTEN even though the layout is a form. Use STRUCTURED only when the filled-in entries are typed or printed. Handwritten routes to the vision model; structured digital pages route to the OCR text route.
If the document is none of the listed DOC types, set "document_type" to a short descriptive name in lower case (for example "rent agreement", "utility bill"); do not force it into a DOC code (FR-CLS-008).
"language" is the dominant language of the text. Use null for subtype unless the category is unstructured.
"rotation" is how many degrees CLOCKWISE the page must be turned so the text reads upright (0 if already upright, 180 if upside down)."""

# A filename is not evidence of document identity: it may hint at the broad *category* (which
# only selects a prompt/model) but never binds a document-specific schema.
_FILENAME_HINTS: List[Tuple[Tuple[str, ...], DocumentCategory]] = [
    (("prescription", "rx"), DocumentCategory.HANDWRITTEN),
    (("brochure", "pamphlet", "menu", "invitation", "leaflet"), DocumentCategory.UNSTRUCTURED),
    (("discharge", "hospital", "patient", "summary", "medical"), DocumentCategory.UNSTRUCTURED),
    (("insurance", "policy", "nominee", "proposal"), DocumentCategory.STRUCTURED),
    (("bank", "casa", "account", "saving", "canara", "kyc", "form"), DocumentCategory.STRUCTURED),
]

# Categories of the schema-bound types Module B can identify from page content.
_RULE_TYPE_CATEGORY: Dict[str, DocumentCategory] = {
    "DOC-01": DocumentCategory.STRUCTURED,
    "DOC-02": DocumentCategory.STRUCTURED,
    "DOC-03": DocumentCategory.UNSTRUCTURED,
}


def heuristic_classify(
    filename: str,
    rule_type: Optional[str] = None,
    rule_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Offline fallback used when no VLM is available or classification fails.

    ``document_type`` is only ever set from ``rule_type`` - Module B's content-based
    classification, already subject to its confidence threshold - and carries Module B's real
    ``rule_confidence``. Filename hints and the default produce no document type and no
    confidence (None), so an unidentified document stays unidentified (never DOC-01).
    """
    if rule_type in _RULE_TYPE_CATEGORY:
        return {"category": _RULE_TYPE_CATEGORY[rule_type].value, "document_type": rule_type,
                "subtype": None, "language": "en", "confidence": rule_confidence,
                "method": "rule_classifier"}
    name = filename.lower()
    for keys, cat in _FILENAME_HINTS:
        if any(k in name for k in keys):
            return {"category": cat.value, "document_type": None, "subtype": None,
                    "language": "en", "confidence": None, "method": "filename_hint"}
    return {"category": "structured", "document_type": None, "subtype": None,
            "language": "en", "confidence": None, "method": "default"}


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
            "This page is handwritten. It is either a filled-in form (application, KYC, account opening, claim) "
            "or a doctor's prescription. Read EVERY filled-in value on the page, not just a few. "
            "FORM: one entry per labelled field that has a value, keyed by its printed label in snake_case "
            "(for example applicant_name, date_of_birth, fathers_name, gender, marital_status, nationality, "
            "address, city, state, pin_code, mobile, email, pan_card_number, aadhaar_number, profession, "
            "employer, annual_income, account_type, nominee_name, relationship). Give each tick box as "
            "true (ticked) or false (empty box) when the box is part of the form. "
            "PRESCRIPTION: patient_name, age, date, doctor_name, diagnosis, and medicines with dosage and "
            "frequency (OD, BD, TDS, SOS). Keep the original shorthand. "
            "Use null only for a field that is empty or not on the page. If a word is uncertain, give your best "
            "reading and set \"uncertain\": true on that field. "
            "READING RULES (handwriting): "
            "(a) Read each value from its own box or line, next to its own label. Never copy a value from one "
            "field into another; two different fields (for example PAN and Aadhaar) almost never hold the same "
            "value, so if they seem to, read both boxes again. "
            "(b) Transcribe the complete entry exactly as written: every word, initial and digit (all parts of "
            "a name, the whole address). Do not shorten, reorder or correct it. "
            "(c) Printed labels, section headings and rubber stamps are not filled-in values. Never turn printed "
            "text or a stamp into true/false. "
            "(d) If an entry is struck out, overwritten or corrected, give the final reading and set "
            "\"uncertain\": true. "
            "(e) Repeating entries (several nominees, medicines, dependents) go in a list of objects, in page "
            "order, each item with its own bbox. "
            "(f) Leave out fields that are empty. Never write the word \"null\", \"N/A\" or \"-\" as a value. "
            "(g) Give exact box coordinates around the ink, not numbers rounded to 50 or 100."
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


PAGE_TEXT_ENGINE = "page_text"
PAGE_TEXT_MAX_NEW_TOKENS = 4096      # a whole page of text is much longer than a list of fields


def build_page_text_prompt(category: DocumentCategory, subtype: Optional[str] = None, language: str = "en") -> str:
    """Prompt for whole-page OCR: every piece of text in reading order, no field picking."""
    kind = category.value + (f" ({subtype})" if subtype and subtype != "other" else "")
    return f"""You are a high-precision OCR engine. Document language: {language}. Page category: {kind}.

TASK:
Transcribe ALL the text on this page, printed and handwritten, from top to bottom in reading order. Do not pick
fields, summarise, translate or correct anything. Split the page into blocks: a heading, a paragraph, one
label-with-its-value line, one list item, one table row (cells separated by " | "), one tick-box line
(write "[x] label" when ticked and "[ ] label" when empty), a stamp, or a signature.

OUTPUT FORMAT (the key is the block type and its number in reading order):
{{"<heading|text|field|list_item|table_row|checkbox|stamp|signature>_<001>": {{"value": "<the text exactly as written>", "bbox": [x1, y1, x2, y2]}}, ...}}

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

    def classify(
        self,
        image_path: str,
        filename: str,
        rule_type: Optional[str] = None,
        rule_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """VLM classification with heuristic fallback. Always returns a dict with a valid category."""
        try:
            result = self.get_router().classify_page_image(image_path)
            cat = str(result.get("category", "")).lower()
            if cat not in {c.value for c in DocumentCategory}:
                raise ValueError(f"unrecognised category {cat!r}")
            # Hard rule (PRD 3.1): anything filled in by hand is handwritten, whatever the layout.
            # Decided in code from the explicit "filled_by" answer, so it does not depend on the
            # model applying the rule itself.
            filled_by = str(result.get("filled_by", "")).lower()
            if filled_by == "hand" and cat != DocumentCategory.HANDWRITTEN.value:
                result["category_before_rule"] = cat
                cat = DocumentCategory.HANDWRITTEN.value
            result["category"] = cat
            result["method"] = "vlm"
            result["router_model"] = self.router_model
            return result
        except Exception as exc:  # classification must never block the demo
            logger.warning("VLM classification failed, using heuristic: %s", exc)
            result = heuristic_classify(filename, rule_type, rule_confidence)
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
        page_text = engine == PAGE_TEXT_ENGINE
        if page_text:
            routed = default            # one vision model reads every category, so each block has a box
        elif engine and engine != "vlm":
            alt = self.table.get("alt_engines", {}).get(engine)
            if alt is None:
                raise ValueError(f"unknown engine {engine!r}")
            routed = alt
        else:
            routed = select_route(category, language, self.table)

        default_extractor = self.get_extractor(default)
        if page_text:
            prompt = build_page_text_prompt(category, subtype, language)
        elif schema_cls is not None and category == DocumentCategory.STRUCTURED:
            prompt = default_extractor._build_system_prompt(schema_cls)
        else:
            prompt = build_generic_prompt(category, subtype, language)

        routed_label = select_model(category, language, {"routes": {category.value: {"default": routed}}})
        candidates = [routed] if routed == default else [routed, default]

        last_exc: Optional[Exception] = None
        for i, route in enumerate(candidates):
            try:
                adapter = self.get_adapter(route)
                usual_tokens = getattr(default_extractor, "max_new_tokens", None)
                if page_text and usual_tokens:
                    default_extractor.max_new_tokens = max(usual_tokens, PAGE_TEXT_MAX_NEW_TOKENS)
                try:
                    raw, prov, meta = adapter.extract(image_path, prompt)
                finally:
                    if page_text and usual_tokens:
                        default_extractor.max_new_tokens = usual_tokens
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
