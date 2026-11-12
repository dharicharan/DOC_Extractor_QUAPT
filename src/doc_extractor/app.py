"""
Interactive Gradio Testing UI for Document Extraction Engine.
Ideal for Kaggle notebooks (with share=True) and local testing.
Uploads PDF/Image -> Ingests & Preprocesses -> Extracts with Qwen-VL ->
Draws grounded bounding boxes on the image -> Displays structured JSON.
"""

import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Ensure src/ is on sys.path and clean out doc_extractor directory from sys.path
# to prevent local files (logging.py, etc.) from shadowing stdlib and sibling packages.
_src_dir = str(Path(__file__).resolve().parent.parent)
_current_dir = str(Path(__file__).resolve().parent)

while "" in sys.path:
    sys.path.remove("")
while _current_dir in sys.path:
    sys.path.remove(_current_dir)
while "." in sys.path:
    sys.path.remove(".")

if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from PIL import Image, ImageDraw, ImageFont

from doc_extractor.config import Settings
from doc_extractor.contracts_b import MVPDocumentType
from doc_extractor.extractor import SchemaGuidedExtractor, SCHEMA_MAP
from doc_extractor.handoff import CoordinateTransform
from doc_extractor.pipeline import ModuleAPipeline
from doc_extractor.pipeline_b import ModuleBPipeline
from doc_extractor.pipeline_c import ModuleCPipeline
from doc_extractor.validation_rules import normalize_and_format
from doc_extractor.routing import (
    DEFAULT_MODEL,
    LANGUAGES,
    UNSTRUCTURED_SUBTYPES,
    DocumentCategory,
    RoutedExtractor,
    heuristic_classify,
)
from doc_extractor.storage import LocalDiskStorage

AUTO = "Auto-Detect"
ENGINE_CHOICES = {
    "Follow routing table": None,
    "GLM-OCR + Qwen3-4B-Instruct text (experimental)": "ocr_text",
    "GLM-OCR + Qwen2.5-7B text (experimental, comparison)": "ocr_text_7b",
}

try:
    import gradio as gr
except ImportError:
    gr = None


def get_pipeline():
    """Initializes and caches pipeline instances."""
    storage = LocalDiskStorage(Path("./data/storage"), tenant_id="test_tenant")
    pipe_a = ModuleAPipeline(storage=storage, tenant_id="test_tenant")
    pipe_b = ModuleBPipeline(storage=storage, tenant_id="test_tenant")
    pipe_c = ModuleCPipeline()
    return pipe_a, pipe_b, pipe_c


def draw_bounding_boxes(
    pil_image: Image.Image,
    fields: Dict[str, Any],
) -> Image.Image:
    """Draws color-coded bounding boxes and field labels on the page image."""
    img_draw = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img_draw)
    img_w, img_h = img_draw.size

    # Color palette for bounding boxes
    colors = [
        "#00E676", "#2979FF", "#FF9100", "#FF1744",
        "#651FFF", "#00E5FF", "#F50057", "#76FF03",
    ]

    for idx, (field_name, field_data) in enumerate(fields.items()):
        val = getattr(field_data, "value", None)
        if isinstance(field_data, dict):
            val = field_data.get("value")

        if val is None:
            continue

        evidence = getattr(field_data, "evidence", None)
        if isinstance(field_data, dict):
            evidence = field_data.get("evidence")

        if not evidence:
            continue

        bbox = getattr(evidence, "bbox", None)
        if isinstance(evidence, dict):
            bbox = evidence.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x0, y0, x1, y1 = bbox
        color = colors[idx % len(colors)]

        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        # Draw label header
        label = f"{field_name.split('.')[-1]}: {str(val)[:20]}"
        text_bbox = draw.textbbox((x0, max(0, y0 - 18)), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x0 + 2, max(0, y0 - 18)), label, fill="#000000")

    return img_draw


_ROUTED: Optional[RoutedExtractor] = None


def get_routed_extractor() -> RoutedExtractor:
    """One RoutedExtractor per process so model weights load once."""
    global _ROUTED
    if _ROUTED is None:
        _ROUTED = RoutedExtractor()
    return _ROUTED


def _bbox_to_pixels(bbox, w: int, h: int):
    """Qwen3-VL bbox [x1, y1, x2, y2] on 0-1000 (its native format) -> (x0, y0, x1, y1) pixels.
    Scaled once, here only. Corners are sorted and clipped so a reversed or out-of-range box still draws."""
    x1, y1, x2, y2 = (float(c) for c in bbox)
    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    return (int(x1 * w / 1000.0), int(y1 * h / 1000.0), int(x2 * w / 1000.0), int(y2 * h / 1000.0))


MAX_PAGES = int(os.environ.get("DOC_MAX_PAGES", "5"))  # demo cap; every page up to this is read


def _classification_markdown(pages_info: List[Dict[str, Any]]) -> str:
    """One row per page: what it was classified as and which model read it."""
    lines = [
        "| Page | Category | Type | Sub-type | Language | Confidence | Model | Source |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i in pages_info:
        lines.append(
            f"| {i.get('page', 1)} | {i.get('category')} | {i.get('document_type')} | {i.get('subtype') or '-'} | "
            f"{i.get('language')} | {float(i.get('confidence') or 0):.2f} | {i.get('model_used', '-')}"
            f"{' (fallback)' if i.get('fallback_occurred') else ''} | {i.get('category_source')}/{i.get('method', '-')} |"
        )
    return "\n".join(lines)


def _write_json_download(document_id: str, text: str) -> str:
    out = Path(tempfile.gettempdir()) / f"{document_id}.json"
    out.write_text(text, encoding="utf-8")
    return str(out)


def _resolve_page(
    img_path: Path,
    filename: str,
    rule_type: Optional[str],
    choices: Dict[str, str],
    page_no: int,
) -> Dict[str, Any]:
    """Category, type, sub-type and language for one page: user choice wins, else auto-detect (FR-CLS-005)."""
    user_cat = bool(choices["category"]) and choices["category"] != AUTO
    user_type = bool(choices["doc_type"]) and choices["doc_type"] != AUTO
    user_lang = bool(choices["language"]) and choices["language"] != AUTO
    user_sub = bool(choices["subtype"]) and choices["subtype"] != AUTO

    auto: Dict[str, Any] = {}
    if not (user_cat and user_type and user_lang):
        if img_path and img_path.exists():
            auto = get_routed_extractor().classify(str(img_path), filename, rule_type if page_no == 1 else None)
        else:
            auto = heuristic_classify(filename, rule_type if page_no == 1 else None)

    category = DocumentCategory(choices["category"].lower() if user_cat else auto.get("category", "structured"))
    doc_type = choices["doc_type"] if user_type else (auto.get("document_type") or (rule_type if page_no == 1 else None))
    if category != DocumentCategory.STRUCTURED and not user_type:
        doc_type = None  # unstructured/handwritten use generic extraction unless the user picks a schema
    if doc_type not in SCHEMA_MAP:
        doc_type = None  # no schema yet (e.g. DOC-04..08): generic extraction, not a wrong schema
    language = LANGUAGES[choices["language"]] if user_lang else str(auto.get("language") or "en")
    subtype = choices["subtype"] if user_sub else auto.get("subtype")
    if category != DocumentCategory.UNSTRUCTURED or subtype in (None, "null"):
        subtype = None

    return {
        "page": page_no,
        "category": category.value,
        "category_source": "user" if user_cat else "auto",
        "document_type": doc_type or auto.get("document_type") or "generic",
        "schema_id": doc_type,
        "type_source": "user" if user_type else "auto",
        "subtype": subtype,
        "language": language,
        "language_source": "user" if user_lang else "auto",
        "confidence": 1.0 if user_cat else auto.get("confidence", 0.0),
        "method": "user" if user_cat else auto.get("method", "-"),
    }


def process_document(
    file_obj,
    category_choice: str,
    doc_type_choice: str,
    subtype_choice: str,
    language_choice: str,
    engine_choice: str = "Follow routing table",
):
    """Reads every page (up to MAX_PAGES): classify each page, route it to the model for its category,
    merge fields with page numbers (FR-CLS-005, FR-RTE-001, FR-ING-004).

    Yields (gallery_of_annotated_pages, status_text, json_output, classification_markdown, json_file_path).
    """
    if file_obj is None:
        yield None, "No file uploaded", "{}", "", None
        return

    pipe_a, pipe_b, _pipe_c = get_pipeline()
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    claimed_name = Path(file_path).name
    choices = {"category": category_choice, "doc_type": doc_type_choice,
               "subtype": subtype_choice, "language": language_choice}

    try:
        # 1. Ingestion & preprocessing (Module A): every page rendered at 200 DPI
        payload_a = pipe_a.process_document(file_path, claimed_filename=claimed_name)
        if payload_a.status == "rejected":
            err = json.dumps(payload_a.to_dict(), indent=2)
            yield None, f"Rejected: {payload_a.error}", err, "", None
            return

        page_items: List[Tuple[int, Path, Optional[Path]]] = []
        for n, rec in enumerate(payload_a.pages, start=1):
            prep = pipe_a.storage.resolve_uri_to_path(rec.preprocessed_storage_uri)
            orig = pipe_a.storage.resolve_uri_to_path(rec.original_storage_uri)
            page_items.append((n, prep if prep.exists() else orig, orig))
        if not page_items:
            page_items = [(1, Path(file_path), None)]
        total_pages = len(page_items)
        page_items = page_items[:MAX_PAGES]

        # 2. Rule-based hint from Module B (page-1 classification fallback)
        payload_b = pipe_b.process_handoff(payload_a)
        rule_type = payload_b.classification.document_type.value
        rule_type = rule_type if rule_type != "UNKNOWN" else None

        engine = ENGINE_CHOICES.get(engine_choice)
        routed = get_routed_extractor()

        fields_dict: Dict[str, Any] = {}
        flags: List[Dict[str, Any]] = []
        no_evidence: List[str] = []
        failed: List[str] = []
        pages_info: List[Dict[str, Any]] = []
        gallery: List[Tuple[Image.Image, str]] = []
        provs: List[Any] = []
        routes: List[Dict[str, Any]] = []

        for page_no, img_path, orig_path in page_items:
            info = _resolve_page(img_path, claimed_name, rule_type, choices, page_no)
            category = DocumentCategory(info["category"])
            schema_cls = SCHEMA_MAP.get(info["schema_id"]) if info["schema_id"] else None
            pages_info.append(info)
            yield (gallery or None,
                   f"Page {page_no}/{len(page_items)}: classified as {category.value} ({info['language']}). Extracting...",
                   "", _classification_markdown(pages_info), None)

            if not (img_path and img_path.exists()):
                raise FileNotFoundError(f"Page {page_no} image not found: {img_path}")

            # 3. Extraction with the model routed for this page's category
            raw_dict, prov, route = routed.extract(
                str(img_path), category, info["language"], schema_cls, info["subtype"], engine=engine
            )
            info["model_used"] = route["model_used"]
            info["fallback_occurred"] = route["fallback_occurred"]
            provs.append(prov)
            routes.append({"page": page_no, **route})

            with Image.open(orig_path if orig_path and orig_path.exists() else img_path) as pil_img:
                w, h = pil_img.size
                base_img = pil_img.copy()

            grounded = route.get("grounded", True)
            page_fields: Dict[str, Any] = {}
            for fpath, item in raw_dict.items():
                if not isinstance(item, dict):
                    continue
                v = item.get("value")
                bbox = item.get("bbox")
                key = fpath if fpath not in fields_dict else f"{fpath}@p{page_no}"
                null_reason = None
                validation = "not_applicable"
                if v is None:
                    null_reason = "absent"
                elif schema_cls is not None and not fpath.startswith("other."):
                    # FR-VAL-001/002 (schema fields only; free-form other.* labels are not normalised): deterministic normalisation and format checks (schema'd fields only)
                    v, err = normalize_and_format(fpath, v)
                    validation = "failed" if err else "passed"
                    if err:
                        failed.append(key)
                ev = None
                if v is not None and grounded and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    ev = {"page": page_no, "bbox": _bbox_to_pixels(bbox, w, h), "text": str(v)[:120]}
                elif v is not None:
                    no_evidence.append(key)  # GR-002: value without a box
                entry = {
                    "value": v,
                    # confidence is not calibrated yet (FR-CNF-001, M6), so no number is claimed
                    "confidence": None,
                    "null_reason": null_reason,
                    "review_required": (
                        v is not None and (
                            category == DocumentCategory.HANDWRITTEN or bool(item.get("uncertain"))
                            or validation == "failed" or ev is None
                        )
                    ),
                    "validation": validation,
                    "evidence": ev,
                    "page": page_no,
                }
                fields_dict[key] = entry
                page_fields[key] = entry

            gallery.append((draw_bounding_boxes(base_img, page_fields), f"Page {page_no}: {category.value}"))

        if no_evidence:
            # PRD SCH-013 says null such values. The demo keeps the value but flags it for review.
            flags.append({"code": "EVIDENCE_MISSING", "fields": no_evidence})
        if failed:
            flags.append({"code": "VALIDATION_FAILED", "fields": failed})
        if total_pages > len(page_items):
            flags.append({"code": "PAGES_NOT_PROCESSED", "fields": [],
                          "detail": f"Read pages 1-{len(page_items)} of {total_pages} (demo cap DOC_MAX_PAGES={MAX_PAGES})."})

        first = pages_info[0]
        cats = {p["category"] for p in pages_info}
        specialists = sorted({p["model_used"] for p in pages_info})
        # SCH-012 gating. Confidence is uncalibrated, so the demo never auto-approves (GR-027).
        doc_status = "needs_review"
        envelope = {
            "document_id": payload_a.document_id,
            "document_metadata": {
                "document_category": first["category"] if len(cats) == 1 else "mixed",
                "document_type": first["document_type"],
                "document_subtype": first["subtype"],
                "language": first["language"],
                "schema_id": first["schema_id"],
                "schema_version": "1.0.0" if first["schema_id"] else None,
                "classification_confidence": first["confidence"],
                "status": doc_status,
                "page_count": total_pages,
                "pages_processed": len(page_items),
                "pages": [
                    {k: p.get(k) for k in ("page", "category", "document_type", "subtype", "language", "model_used")}
                    for p in pages_info
                ],
            },
            "fields": fields_dict,
            "flags": flags,
            "run": {
                "run_id": payload_a.run_id,
                "model_versions": {
                    "router": DEFAULT_MODEL if first.get("method") == "vlm" else first.get("method"),
                    "specialist": ", ".join(specialists),
                    "normalizer": "deterministic-v1",
                },
                "confidence_calibrated": False,
                "routing": routes,
                "provenance": [p.model_dump() if hasattr(p, "model_dump") else p for p in provs],
            },
        }
        json_text = json.dumps(envelope, indent=2, default=str)
        status_msg = (
            f"Status: {doc_status.upper()} | Pages read: {len(page_items)}/{total_pages} | "
            f"Fields: {len(fields_dict)} | Flags: {len(flags)} | Model(s): {', '.join(specialists)}"
        )
        yield gallery, status_msg, json_text, _classification_markdown(pages_info), _write_json_download(payload_a.document_id, json_text)
        return

    except Exception as e:
        yield None, f"Error: {e}", json.dumps({"error": str(e)}, indent=2), "", None
        return


def create_demo() -> "gr.Blocks":
    """Constructs Gradio UI layout (FR-DEMO-001..006)."""
    with gr.Blocks(title="Document Extraction Engine") as demo:
        gr.Markdown(
            "# 📄 Document Extraction Engine\n"
            "Upload a document. The engine first classifies it (structured, unstructured or handwritten), "
            "then routes it to the model for that category and returns grounded JSON."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(label="Upload Document (PDF, JPG, PNG, TIFF)", file_types=[".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"])
                with gr.Accordion("Manual override (optional, leave on Auto-Detect to classify automatically)", open=False):
                    category_dd = gr.Dropdown(
                        choices=[AUTO, "Structured", "Unstructured", "Handwritten"], value=AUTO,
                        label="Document Category",
                    )
                    doc_type_dd = gr.Dropdown(
                        choices=[AUTO, "DOC-01", "DOC-02", "DOC-03"], value=AUTO,
                        label="Document Type (structured schemas)",
                    )
                    subtype_dd = gr.Dropdown(
                        choices=[AUTO] + UNSTRUCTURED_SUBTYPES, value=AUTO,
                        label="Unstructured Sub-type (brochure, pamphlet, menu card, ...)",
                    )
                    language_dd = gr.Dropdown(
                        choices=[AUTO] + list(LANGUAGES), value=AUTO, label="Language",
                    )
                    engine_dd = gr.Dropdown(
                        choices=list(ENGINE_CHOICES), value="Follow routing table",
                        label="Engine (compare the default VLM with GLM-OCR + text LLM; the OCR route has no boxes)",
                    )
                submit_btn = gr.Button("Classify & Extract", variant="primary")

            with gr.Column(scale=2):
                status_box = gr.Textbox(label="Status & Quality Gate", interactive=False)
                classification_md = gr.Markdown(label="Classification Result")
                with gr.Tab("Visual Grounding"):
                    output_image = gr.Gallery(label="Annotated pages (grounded bounding boxes)", columns=1, height=700)
                with gr.Tab("Canonical JSON Output"):
                    output_json = gr.Code(label="Extracted JSON Envelope", language="json")
                    json_file = gr.File(label="Download JSON")

        submit_btn.click(
            fn=process_document,
            inputs=[input_file, category_dd, doc_type_dd, subtype_dd, language_dd, engine_dd],
            outputs=[output_image, status_box, output_json, classification_md, json_file],
        )

    return demo


if __name__ == "__main__":
    if gr is None:
        print("Gradio is not installed. Run: pip install gradio")
    else:
        demo = create_demo()
        # In Kaggle or remote server, share=True generates a public gradio.live URL
        share_mode = os.environ.get("GRADIO_SHARE", "True").lower() == "true"
        port = int(os.environ.get("GRADIO_SERVER_PORT", 0)) or None
        demo.launch(share=share_mode, server_port=port)

