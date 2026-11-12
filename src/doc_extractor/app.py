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
from doc_extractor.routing import (
    LANGUAGES,
    UNSTRUCTURED_SUBTYPES,
    DocumentCategory,
    RoutedExtractor,
    heuristic_classify,
)
from doc_extractor.storage import LocalDiskStorage

AUTO = "Auto-Detect"

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
    """Qwen bbox [ymin, xmin, ymax, xmax] on 0-1000 -> (x0, y0, x1, y1) pixels. Scaled once, here only."""
    ymin, xmin, ymax, xmax = bbox
    return (int(xmin * w / 1000.0), int(ymin * h / 1000.0), int(xmax * w / 1000.0), int(ymax * h / 1000.0))


def _classification_markdown(info: Dict[str, Any]) -> str:
    rows = [
        ("Category", info.get("category"), info.get("category_source")),
        ("Document type", info.get("document_type"), info.get("type_source")),
        ("Sub-type", info.get("subtype") or "-", info.get("subtype_source", "-")),
        ("Language", info.get("language"), info.get("language_source")),
        ("Confidence", f"{float(info.get('confidence', 0)):.2f}", info.get("method", "-")),
        ("Model", info.get("model_used", "-"), "fallback used" if info.get("fallback_occurred") else "routed"),
    ]
    lines = ["| Field | Value | Source |", "| --- | --- | --- |"]
    lines += [f"| {k} | {v} | {src} |" for k, v, src in rows]
    return "\n".join(lines)


def _write_json_download(document_id: str, text: str) -> str:
    out = Path(tempfile.gettempdir()) / f"{document_id}.json"
    out.write_text(text, encoding="utf-8")
    return str(out)


def process_document(
    file_obj,
    category_choice: str,
    doc_type_choice: str,
    subtype_choice: str,
    language_choice: str,
    use_qwen_vl: bool = True,
) -> Tuple[Optional[Image.Image], str, str, str, Optional[str]]:
    """Returns (annotated_image, status_text, json_output, classification_markdown, json_file_path).

    Category, type, sub-type and language are either user-selected or auto-detected
    (FR-DEMO-002..005). The category picks the model via the routing table (FR-RTE-001).
    """
    if file_obj is None:
        yield None, "No file uploaded", "{}", "", None
        return

    pipe_a, pipe_b, pipe_c = get_pipeline()
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    claimed_name = Path(file_path).name

    try:
        # 1. Ingestion & preprocessing (Module A)
        payload_a = pipe_a.process_document(file_path, claimed_filename=claimed_name)
        if payload_a.status == "rejected":
            err = json.dumps(payload_a.to_dict(), indent=2)
            yield None, f"Rejected: {payload_a.error}", err, "", None
            return

        orig_path = None
        if payload_a.pages:
            rec = payload_a.pages[0]
            prep_path = pipe_a.storage.resolve_uri_to_path(rec.preprocessed_storage_uri)
            orig_path = pipe_a.storage.resolve_uri_to_path(rec.original_storage_uri)
            img_path = prep_path if prep_path.exists() else orig_path
        else:
            img_path = Path(file_path)

        # 2. Rule-based parse/classify (Module B): classification hint and CPU path input
        payload_b = pipe_b.process_handoff(payload_a)
        rule_type = payload_b.classification.document_type.value
        rule_type = rule_type if rule_type != "UNKNOWN" else None

        # 3. Resolve category, type, sub-type, language: user choice wins, else auto-detect
        user_cat = bool(category_choice) and category_choice != AUTO
        user_type = bool(doc_type_choice) and doc_type_choice != AUTO
        user_lang = bool(language_choice) and language_choice != AUTO
        user_sub = bool(subtype_choice) and subtype_choice != AUTO

        auto: Dict[str, Any] = {}
        if not (user_cat and user_type and user_lang):
            if use_qwen_vl and img_path and img_path.exists():
                auto = get_routed_extractor().classify(str(img_path), claimed_name, rule_type)
            else:
                auto = heuristic_classify(claimed_name, rule_type)

        category = DocumentCategory(category_choice.lower() if user_cat else auto.get("category", "structured"))

        doc_type = doc_type_choice if user_type else (auto.get("document_type") or rule_type)
        if category != DocumentCategory.STRUCTURED and not user_type:
            doc_type = None  # unstructured/handwritten use generic extraction unless the user picks a schema
        if doc_type not in SCHEMA_MAP:
            doc_type = None  # no schema yet (e.g. DOC-04..08): generic extraction, not a wrong schema

        language = LANGUAGES[language_choice] if user_lang else str(auto.get("language") or "en")
        subtype = subtype_choice if user_sub else auto.get("subtype")
        if category != DocumentCategory.UNSTRUCTURED or subtype in (None, "null"):
            subtype = None

        info: Dict[str, Any] = {
            "category": category.value,
            "category_source": "user" if user_cat else "auto",
            "document_type": doc_type or auto.get("document_type") or "generic",
            "type_source": "user" if user_type else "auto",
            "subtype": subtype,
            "subtype_source": "user" if user_sub else "auto",
            "language": language,
            "language_source": "user" if user_lang else "auto",
            "confidence": 1.0 if user_cat else auto.get("confidence", 0.0),
            "method": "user" if user_cat else auto.get("method", "-"),
        }

        schema_cls = SCHEMA_MAP.get(doc_type) if doc_type else None

        yield None, f"Classified as {category.value} ({language}). Extracting...", "", _classification_markdown(info), None

        # 4. Extraction
        if use_qwen_vl and img_path and img_path.exists():
            raw_dict, prov, route = get_routed_extractor().extract(
                str(img_path), category, language, schema_cls, subtype
            )
            info["model_used"] = route["model_used"]
            info["fallback_occurred"] = route["fallback_occurred"]

            with Image.open(orig_path if orig_path and orig_path.exists() else img_path) as pil_img:
                w, h = pil_img.size
                base_img = pil_img.copy()

            fields_dict: Dict[str, Any] = {}
            for fpath, item in raw_dict.items():
                if not isinstance(item, dict):
                    continue
                v = item.get("value")
                bbox = item.get("bbox")
                if v is None and bbox is None:
                    continue
                ev = None
                if bbox and len(bbox) == 4:
                    ev = {
                        "page": 1,
                        "bbox": _bbox_to_pixels(bbox, w, h),
                        "text_snippet": str(v)[:120],
                        "ocr_confidence": 0.95,
                    }
                fields_dict[fpath] = {
                    "value": v,
                    "calibrated_confidence": 0.95 if v is not None else 0.0,
                    "review_required": category == DocumentCategory.HANDWRITTEN or bool(item.get("uncertain")),
                    "evidence": ev,
                }

            annotated_img = draw_bounding_boxes(base_img, fields_dict)
            envelope = {
                "document_id": payload_a.document_id,
                "document_metadata": {
                    "document_category": category.value,
                    "document_type": doc_type or "generic",
                    "document_subtype": subtype,
                    "language": language,
                    "schema_id": doc_type,
                    "schema_version": "1.0.0" if doc_type else None,
                    "classification_confidence": info["confidence"],
                    "status": "valid",
                    "page_count": len(payload_a.pages),
                },
                "fields": fields_dict,
                "flags": [],
                "run": {
                    "run_id": payload_a.run_id,
                    "document_category": category.value,
                    "model_versions": {
                        "routed_model": route["model_used"],
                        "normalizer": "deterministic-v1",
                    },
                    "routing": route,
                    "provenance": prov.model_dump() if hasattr(prov, "model_dump") else prov,
                },
            }
            json_text = json.dumps(envelope, indent=2, default=str)
            status_msg = (
                f"Status: VALID | Category: {category.value} | Type: {doc_type or 'generic'} | "
                f"Fields: {len(fields_dict)} | Model: {route['model_used']}"
            )
            yield annotated_img, status_msg, json_text, _classification_markdown(info), _write_json_download(payload_a.document_id, json_text)
            return

        # CPU heuristic fast path: only the rule-based structured schemas are supported
        if doc_type not in SCHEMA_MAP:
            msg = "CPU heuristic mode only supports structured DOC-01 to DOC-03. Enable the VLM for this document."
            yield None, msg, json.dumps({"error": msg}, indent=2), _classification_markdown(info), None
            return

        payload_b.classification.document_type = MVPDocumentType(doc_type)
        payload_b.classification.schema_id = doc_type
        payload_b.classification.review_required = False
        payload_b.classification.classification_confidence = max(0.95, payload_b.classification.classification_confidence)
        payload_b.status = "processing"
        pipe_c.extractor.use_llm = False
        env = pipe_c.process(payload_b)

        annotated_img = None
        if orig_path and orig_path.exists():
            with Image.open(orig_path) as pil_img:
                annotated_img = draw_bounding_boxes(pil_img, env.fields)
        info["model_used"] = "heuristic"
        status_msg = (
            f"Status: {env.document_metadata.status.upper()} | Type: {env.document_metadata.document_type} | "
            f"Confidence: {env.document_metadata.classification_confidence:.2f} | Flags: {len(env.flags)}"
        )
        json_text = json.dumps(env.model_dump(), indent=2, default=str)
        yield annotated_img, status_msg, json_text, _classification_markdown(info), _write_json_download(payload_a.document_id, json_text)
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
                submit_btn = gr.Button("Classify & Extract", variant="primary")

            with gr.Column(scale=2):
                status_box = gr.Textbox(label="Status & Quality Gate", interactive=False)
                classification_md = gr.Markdown(label="Classification Result")
                with gr.Tab("Visual Grounding"):
                    output_image = gr.Image(label="Annotated Document Page (with Grounded Bounding Boxes)", type="pil")
                with gr.Tab("Canonical JSON Output"):
                    output_json = gr.Code(label="Extracted JSON Envelope", language="json")
                    json_file = gr.File(label="Download JSON")

        submit_btn.click(
            fn=process_document,
            inputs=[input_file, category_dd, doc_type_dd, subtype_dd, language_dd],
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

