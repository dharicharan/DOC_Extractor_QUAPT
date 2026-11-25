"""
Interactive Gradio Testing UI for Document Extraction Engine (PRD v1.5).
Ideal for Kaggle notebooks (with share=True) and local testing.

Uploads PDF/Image -> Module A (200 DPI pages) -> Module B (classification hand-off) -> per page:
classify once, route to the specialist for the category (VLM, or OCR + text LLM), read the page ->
ModuleCPipeline (the canonical processing and envelope layer) -> grounded boxes on every page + JSON.

The app does the per-page routing and model calls (so it can stream live progress and turn pages
upright); ModuleCPipeline.process_routed_pages then normalizes, grounds, validates, gates and assembles
the envelope from those results. No model is run twice and no envelope is built here.
"""

import io
import json
import os

# Reduce CUDA fragmentation on small GPUs (must be set before torch initialises CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import sys
import tempfile
import time
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

from doc_extractor.config import Settings, load_settings
from doc_extractor.contracts_b import MVPDocumentType
from doc_extractor.contracts_c import CanonicalExtractionEnvelope, FlagItem
from doc_extractor.extractor import SchemaGuidedExtractor, SCHEMA_MAP
from doc_extractor.pipeline import ModuleAPipeline
from doc_extractor.pipeline_b import ModuleBPipeline, ReviewerOverrideRequest
from doc_extractor.pipeline_c import ModuleCPipeline
from doc_extractor.processing_service import (
    DocumentProcessingRequest,
    DocumentProcessingResult,
    DocumentProcessingService,
    ProcessingProgressEvent,
    RequestTemporaryWorkspace,
    _rotate_page as service_rotate_page,
)
from doc_extractor.routed_pages import RoutedPageResult
from doc_extractor.routing import (
    DEFAULT_MODEL,
    LANGUAGES,
    UNSTRUCTURED_SUBTYPES,
    DocumentCategory,
    RoutedExtractor,
    build_generic_prompt,
    heuristic_classify,
)
from doc_extractor.storage import LocalDiskStorage
from doc_extractor.vlm_grounding import flatten_vlm_output

AUTO = "Auto-Detect"
TENANT_ID = "test_tenant"
OVERRIDE_REVIEWER_ID = "extraction_ui_user"
ENGINE_CHOICES = {
    "Follow routing table": None,
    "GLM-OCR + Qwen3-4B-Instruct text (experimental)": "ocr_text",
    "GLM-OCR + Qwen2.5-7B text (experimental, comparison)": "ocr_text_7b",
    "Full page text (OCR as JSON, no field picking)": "page_text",
}

try:
    import gradio as gr
except ImportError:
    gr = None


_SETTINGS: Optional[Settings] = None
_PIPELINES: Dict[str, Tuple[ModuleAPipeline, ModuleBPipeline]] = {}
_MODULE_C: Optional[ModuleCPipeline] = None


def get_settings() -> Settings:
    """Application settings; validated once and shared by every module the app builds."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def configure_settings(settings: Settings) -> None:
    """Sets the application settings (e.g. a non-default OCR backend) before processing."""
    global _SETTINGS
    _SETTINGS = settings


def get_pipelines(settings: Optional[Settings] = None) -> Tuple[ModuleAPipeline, ModuleBPipeline]:
    """
    Builds (and caches per settings) Module A and Module B with the *same* Settings, so the configured
    storage path and OCR backend are used instead of silent defaults. The default backend is the explicit
    no-OCR pass-through, so this needs no OCR runtime; a configured backend whose runtime is missing
    raises here with an actionable error (there is no fallback to another engine).
    """
    settings = settings if settings is not None else get_settings()
    key = settings.model_dump_json()
    if key not in _PIPELINES:
        storage = LocalDiskStorage(settings.storage_base_dir, tenant_id=TENANT_ID)
        pipe_a = ModuleAPipeline(settings=settings, storage=storage, tenant_id=TENANT_ID)
        pipe_b = ModuleBPipeline(storage=storage, tenant_id=TENANT_ID, settings=settings)
        _PIPELINES[key] = (pipe_a, pipe_b)
    return _PIPELINES[key]


def get_module_c() -> ModuleCPipeline:
    """The canonical Module C orchestrator. It reads nothing itself here: routed page results go in."""
    global _MODULE_C
    if _MODULE_C is None:
        _MODULE_C = ModuleCPipeline(extractor=SchemaGuidedExtractor(use_llm=False))
    return _MODULE_C


def get_pipeline():
    """Backward-compatible (pipe_a, pipe_b, pipe_c) accessor; pipe_c is the canonical ModuleCPipeline."""
    pipe_a, pipe_b = get_pipelines()
    return pipe_a, pipe_b, get_module_c()


def draw_bounding_boxes(
    pil_image: Image.Image,
    fields: Dict[str, Any],
    page: Optional[int] = None,
) -> Image.Image:
    """
    Draws color-coded boxes and labels from each field's EVIDENCE: ``(x0, y0, x1, y1)`` pixels of the
    200-DPI original page (PRD FR-EXT-011), exactly as ModuleCPipeline mapped them - nothing is rescaled
    or reordered here. With ``page`` only the evidence located on that page is drawn (the primary
    evidence and any additional evidence of the same value on other pages).
    """
    img_draw = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img_draw)

    # Color palette for bounding boxes
    colors = [
        "#00E676", "#2979FF", "#FF9100", "#FF1744",
        "#651FFF", "#00E5FF", "#F50057", "#76FF03",
    ]

    for idx, (field_name, field_data) in enumerate(fields.items()):
        val = field_data.get("value") if isinstance(field_data, dict) else getattr(field_data, "value", None)
        if val is None:
            continue

        evidences = []
        primary = field_data.get("evidence") if isinstance(field_data, dict) else getattr(field_data, "evidence", None)
        if primary:
            evidences.append(primary)
        extra = (field_data.get("additional_evidence") if isinstance(field_data, dict)
                 else getattr(field_data, "additional_evidence", None)) or []
        evidences.extend(extra)

        color = colors[idx % len(colors)]
        for evidence in evidences:
            ev_page = evidence.get("page") if isinstance(evidence, dict) else getattr(evidence, "page", None)
            if page is not None and ev_page != page:
                continue
            bbox = evidence.get("bbox") if isinstance(evidence, dict) else getattr(evidence, "bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = bbox

            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
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
        # allow_fallback=False: a failed model run must surface (or use the routing fallback), never
        # silently degrade to regex heuristics.
        _ROUTED = RoutedExtractor(allow_fallback=False)
    return _ROUTED


# Every page is read by default. DOC_MAX_PAGES=N is only an optional limit for quick test runs (0 = no limit).
MAX_PAGES = int(os.environ.get("DOC_MAX_PAGES", "0"))
# FR-CNF-001: p**alpha recalibration of token probabilities; fit with validation_rules.fit_calibration_alpha
CONFIDENCE_ALPHA = float(os.environ.get("DOC_CONFIDENCE_ALPHA", "1.0"))
REVIEW_THRESHOLD = float(os.environ.get("DOC_REVIEW_THRESHOLD", "0.75"))  # FR-CNF-002


def _fmt_confidence(value: Any) -> str:
    """Confidence as measured, or n/a when there is none (nothing is invented for display either)."""
    return f"{float(value):.2f}" if value is not None else "n/a"


def _classification_markdown(pages_info: List[Dict[str, Any]]) -> str:
    """One row per page: what it was classified as and which model read it."""
    lines = [
        "| Page | Category | Type | Sub-type | Language | Confidence | Model | Source |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i in pages_info:
        lines.append(
            f"| {i.get('page', 1)} | {i.get('category')} | {i.get('document_type')}"
            f"{' (suggested ' + str(i['suggested_document_type']) + ')' if i.get('suggested_document_type') else ''} | {i.get('subtype') or '-'} | "
            f"{i.get('language')} | {_fmt_confidence(i.get('confidence'))} | {i.get('model_used', '-')}"
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
    rule_confidence: Optional[float],
    choices: Dict[str, str],
    page_no: int,
) -> Dict[str, Any]:
    """
    Category, type, sub-type and language for one page: user choice wins, else auto-detect (FR-CLS-005).

    A document-specific schema is bound ONLY by an explicit user choice of a supported type (v1.5 reads
    dynamic fields by default). A filename never binds a type or supplies a confidence, and the VLM
    router's self-reported type is advisory only (``suggested_document_type``).
    """
    service = DocumentProcessingService(routed_extractor=get_routed_extractor())
    return service.resolve_page(img_path, filename, rule_type, rule_confidence, choices, page_no)


_STEP_ICONS = {"pending": "⬜", "running": "⏳", "done": "✅", "failed": "❌"}


def _step(steps: List[Dict[str, Any]], key: str, label: Optional[str] = None,
          state: Optional[str] = None, detail: Optional[str] = None) -> None:
    """Add or update one row of the pipeline-steps panel."""
    row = next((r for r in steps if r["key"] == key), None)
    if row is None:
        row = {"key": key, "label": label or key, "state": "pending", "detail": "", "t0": None, "secs": None}
        steps.append(row)
    if label:
        row["label"] = label
    if state:
        if state == "running":
            row["t0"] = time.perf_counter()
        elif state in ("done", "failed") and row.get("t0"):
            row["secs"] = time.perf_counter() - row["t0"]
        row["state"] = state
    if detail is not None:
        row["detail"] = detail


def _render_steps(steps: List[Dict[str, Any]]) -> str:
    if not steps:
        return "Upload a document and click **Classify & Extract**."
    lines = ["| | Step | Result | Time |", "| --- | --- | --- | --- |"]
    for r in steps:
        t = f"{r['secs']:.1f}s" if r.get("secs") is not None else ("..." if r["state"] == "running" else "")
        lines.append(f"| {_STEP_ICONS[r['state']]} | {r['label']} | {r['detail']} | {t} |")
    return "\n".join(lines)


def _safe_rotation(value: Any) -> int:
    try:
        r = int(value) % 360
    except (TypeError, ValueError):
        return 0
    return r if r in (90, 180, 270) else 0


def _rotate_page(
    path: Path,
    clockwise_deg: int,
    workspace: RequestTemporaryWorkspace,
    page_no: int = 1,
) -> Path:
    """
    Rotation helper delegating to the processing service's secure RequestTemporaryWorkspace.
    Requires an explicit RequestTemporaryWorkspace to prevent unmanaged temporary file creation.
    Maintained for backward compatibility and test verification.
    """
    if not isinstance(workspace, RequestTemporaryWorkspace):
        raise TypeError("A valid RequestTemporaryWorkspace instance is required to rotate pages.")
    return service_rotate_page(path, clockwise_deg, workspace=workspace, page_no=page_no)


def _page_text_json(result) -> Dict[str, Any]:
    """The whole page as OCR text blocks: category, sub-type, then every block in reading order.

    Built from each page's own model output, so text is neither merged across pages nor withheld.
    Boxes are on the 0-1000 page grid.
    """
    meta = result.envelope.document_metadata
    pages = []
    for page in result.page_results:
        blocks = []
        for key, leaf in flatten_vlm_output(page.raw or {}).items():
            value = leaf.get("value")
            text = "" if value is None else str(value).strip()
            if not text or text.lower() == "null":
                continue
            kind = key.split(".")[-1].rstrip("0123456789").rstrip("_") or "text"
            blocks.append({"type": kind, "text": text, "bbox": leaf.get("bbox")})
        entry = {"page": page.page_number, "blocks": blocks, "full_text": "\n".join(b["text"] for b in blocks)}
        if page.error:
            entry["error"] = page.error
        pages.append(entry)
    return {
        "document_category": meta.document_category,
        "document_subtype": meta.document_subtype,
        "language": meta.language,
        "bbox_scale": "0-1000",
        "pages": pages,
    }


def _thumb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def process_document(
    file_obj,
    category_choice: str,
    doc_type_choice: str,
    subtype_choice: str,
    language_choice: str,
    engine_choice: str = "Follow routing table",
    settings: Optional[Settings] = None,
):
    """UI entry point: runs the pipeline and adds the live pipeline-steps panel to every update."""
    steps: List[Dict[str, Any]] = []
    for out in _process_core(file_obj, category_choice, doc_type_choice, subtype_choice,
                             language_choice, engine_choice, steps, settings):
        yield (*out, _render_steps(steps))


def _process_core(
    file_obj,
    category_choice: str,
    doc_type_choice: str,
    subtype_choice: str,
    language_choice: str,
    engine_choice: str,
    steps: List[Dict[str, Any]],
    settings: Optional[Settings] = None,
):
    """Reads every page of the document: classify once, route each page to the model for its category,
    then hand the page results to ModuleCPipeline for the canonical envelope (FR-CLS-005, FR-RTE-001,
    FR-ING-004).

    Yields (gallery_of_annotated_pages, status_text, json_output, classification_markdown, json_file_path).
    """
    if file_obj is None:
        yield None, "No file uploaded", "{}", "", None
        return

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    claimed_name = Path(file_path).name

    split_gallery: List[Tuple[Image.Image, str]] = []

    try:
        settings = settings if settings is not None else get_settings()
        pipe_a, pipe_b = get_pipelines(settings)
        pipe_c = get_module_c()
        routed = get_routed_extractor()

        # Shared processing service coordinates pipelines and invokes ModuleCPipeline.process_routed_pages
        service = DocumentProcessingService(
            module_a=pipe_a,
            module_b=pipe_b,
            module_c=pipe_c,
            routed_extractor=routed,
            tenant_id=TENANT_ID,
            actor_id=OVERRIDE_REVIEWER_ID,
            settings=settings,
        )

        request = DocumentProcessingRequest(
            file_path_or_stream=file_path,
            claimed_filename=claimed_name,
            tenant_id=TENANT_ID,
            actor_id=OVERRIDE_REVIEWER_ID,
            category_choice=category_choice,
            doc_type_choice=doc_type_choice,
            subtype_choice=subtype_choice,
            language_choice=language_choice,
            engine_choice=engine_choice,
            max_pages=MAX_PAGES,
        )

        gradio_diagnostics: Dict[str, str] = {}

        def _gradio_diagnostic_sink(stage: str, msg: str) -> None:
            gradio_diagnostics[stage] = msg

        for event in service.process_document_stream(request, diagnostic_sink=_gradio_diagnostic_sink):
            step_detail = event.detail
            if event.step_key.startswith("ext") and event.state == "failed":
                diag = gradio_diagnostics.get(event.step_key)
                if diag:
                    step_detail = str(diag)[:120]
            _step(steps, event.step_key, event.label, event.state, step_detail)

            if event.step_key == "upload":
                pass
            elif event.step_key == "split":
                if event.state == "running":
                    yield None, "Splitting into page images...", "", "", None
                elif event.state == "failed":
                    result = event.result
                    if result and result.is_rejected:
                        err_payload = result.payload_a.to_dict() if result.payload_a else (result.error or {})
                        rejected_detail = result.payload_a.error if result.payload_a else event.detail
                        _step(steps, "split", state="failed", detail=str(rejected_detail)[:120])
                        yield None, f"Rejected: {rejected_detail}", json.dumps(err_payload, indent=2), "", None
                        return
                    else:
                        diag = gradio_diagnostics.get("split") or event.detail
                        _step(steps, "split", state="failed", detail=str(diag)[:120])
                        yield None, f"Error: {diag}", json.dumps({"error": str(diag)}, indent=2), "", None
                        return
                elif event.state == "done":
                    page_items = event.data.get("page_items", []) if event.data else []
                    total_pages = event.data.get("total_pages", len(page_items)) if event.data else len(page_items)
                    split_gallery = [(_thumb(p), f"Page {n} (split)") for n, p, _ in page_items if p and Path(p).exists()]
                    if split_gallery:
                        pw, ph = split_gallery[0][0].size
                        _step(steps, "split", detail=steps[1]["detail"] + f" · page 1 is {pw}×{ph} px ({pw * ph / 1e6:.1f} MP)")
                    _step(steps, "cls", "Classify document from page 1 (router model, once)")
                    for n, _, _ in page_items:
                        _step(steps, f"ext{n}", f"Page {n}: extract fields")
                    _step(steps, "merge", "Validate and build JSON")
                    yield split_gallery or None, f"Split into {total_pages} page image(s).", "", "", None
            elif event.step_key == "cls":
                if event.state == "running":
                    yield (split_gallery or None, "Classifying the document from page 1...", "", "", None)
                elif event.state == "failed":
                    result = event.result
                    diag = gradio_diagnostics.get("cls") or event.detail
                    if result and result.payload_b and result.payload_b.status == "failed":
                        msg = str(diag)
                        err_out = result.payload_b.error or (result.error or {})
                        _step(steps, "cls", state="failed", detail=msg[:120])
                        yield split_gallery or None, msg, json.dumps({"error": err_out}, indent=2, default=str), "", None
                        return
                    else:
                        _step(steps, "cls", state="failed", detail=str(diag)[:120])
                        yield split_gallery or None, f"Error: {diag}", json.dumps({"error": str(diag)}, indent=2), "", None
                        return
            elif event.step_key.startswith("ext"):
                if event.state == "running":
                    pages_info = event.data.get("pages_info", []) if event.data else []
                    category_val = event.data.get("category", "") if event.data else ""
                    lang_val = event.data.get("language", "") if event.data else ""
                    page_no = event.data.get("page_no", 1) if event.data else 1
                    read_count = event.data.get("read_count", 1) if event.data else 1
                    yield (
                        split_gallery or None,
                        f"Page {page_no}/{read_count}: classified as {category_val} ({lang_val}). Extracting...",
                        "",
                        _classification_markdown(pages_info),
                        None,
                    )
                elif event.state == "failed":
                    diag = gradio_diagnostics.get(event.step_key) or event.detail
                    _step(steps, event.step_key, state="failed", detail=str(diag)[:120])
            elif event.step_key == "merge":
                if event.state == "failed":
                    result = event.result
                    diag = gradio_diagnostics.get("merge") or event.detail
                    _step(steps, "merge", state="failed", detail=str(diag)[:120])
                    yield split_gallery or None, f"Error: {diag}", json.dumps({"error": str(diag)}, indent=2), "", None
                    return
                elif event.state == "done" and event.result is not None:
                    result = event.result
                    env = result.envelope
                    page_items = result.page_items
                    rotation = event.data.get("rotation", 0) if event.data else 0
                    category = DocumentCategory(env.document_metadata.document_category)

                    gallery: List[Tuple[Image.Image, str]] = []
                    for page_no, img_path, orig_path in page_items:
                        canvas_path = orig_path if orig_path and orig_path.exists() else img_path
                        with Image.open(canvas_path) as pil_img:
                            annotated = draw_bounding_boxes(pil_img, env.fields, page=page_no)
                        if rotation:
                            annotated = annotated.rotate(-rotation, expand=True)
                        gallery.append((annotated, f"Page {page_no}: {category.value}"))

                    json_text = json.dumps(env.model_dump(mode="json"), indent=2, default=str)
                    withheld = any(not r.route.get("grounded", True) for r in result.page_results)
                    specialists = sorted(
                        {p.get("model_used") for p in result.pages_info if p.get("model_used") not in (None, "-")}
                    )
                    total_pages = event.data.get("total_pages", len(page_items)) if event.data else len(page_items)
                    if ENGINE_CHOICES.get(engine_choice) == "page_text":
                        text_doc = _page_text_json(result)
                        json_text = json.dumps(text_doc, indent=2, ensure_ascii=False)
                        yield (
                            gallery,
                            f"Full page text | Category: {text_doc['document_category']} | "
                            f"Sub-type: {text_doc['document_subtype'] or '-'} | Pages read: {len(page_items)}/{total_pages} | "
                            f"Text blocks: {sum(len(p['blocks']) for p in text_doc['pages'])} | "
                            f"Model(s): {', '.join(specialists) or '-'}",
                            json_text,
                            _classification_markdown(result.pages_info),
                            _write_json_download(result.document_id, json_text),
                        )
                        return
                    status_msg = (
                        f"Status: {env.document_metadata.status.upper()} | Pages read: {len(page_items)}/{total_pages} | "
                        f"Fields: {len(env.fields)} | Flags: {len(env.flags)} | Model(s): {', '.join(specialists) or '-'}"
                        + (" | No boxes on this route: values withheld (EVIDENCE_MISSING); unverified readings are in "
                           "run.provenance.details" if withheld else "")
                    )
                    yield (
                        gallery,
                        status_msg,
                        json_text,
                        _classification_markdown(result.pages_info),
                        _write_json_download(result.document_id, json_text),
                    )
                    return

    except Exception as e:
        for r in steps:
            if r["state"] == "running":
                _step(steps, r["key"], state="failed", detail=str(e)[:120])
        yield split_gallery or None, f"Error: {e}", json.dumps({"error": str(e)}, indent=2), "", None
        return


def create_demo(settings: Optional[Settings] = None) -> "gr.Blocks":
    """Constructs Gradio UI layout (FR-DEMO-001..006). ``settings`` configures every module it runs."""
    if settings is not None:
        configure_settings(settings)
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
                        label="Document Type (pick one to force its fixed schema; Auto = dynamic fields)",
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
                        label="Output / engine (fields by default; or the full page text as JSON; or compare GLM-OCR + text LLM)",
                    )
                submit_btn = gr.Button("Classify & Extract", variant="primary")

            with gr.Column(scale=2):
                status_box = gr.Textbox(label="Status & Quality Gate", interactive=False)
                gr.Markdown("### Pipeline steps")
                steps_md = gr.Markdown(_render_steps([]))
                classification_md = gr.Markdown(label="Classification Result")
                with gr.Tab("Visual Grounding"):
                    output_image = gr.Gallery(label="Annotated pages (grounded bounding boxes)", columns=1, height=700)
                with gr.Tab("Canonical JSON Output"):
                    output_json = gr.Code(label="Extracted JSON Envelope", language="json")
                    json_file = gr.File(label="Download JSON")

        submit_btn.click(
            fn=process_document,
            inputs=[input_file, category_dd, doc_type_dd, subtype_dd, language_dd, engine_dd],
            outputs=[output_image, status_box, output_json, classification_md, json_file, steps_md],
        )

    return demo


if __name__ == "__main__":
    if gr is None:
        print("Gradio is not installed. Run: pip install gradio")
    else:
        if os.environ.get("DOC_PRELOAD", "1") == "1":
            # Load the models before the UI starts so the first click is not slowed by model loading
            try:
                _r = get_routed_extractor()
                print("Preloading models (set DOC_PRELOAD=0 to skip)...")
                _r.get_extractor(_r.table.get("default_model", DEFAULT_MODEL))._load_model()
                _r.get_router()._load_model()
                print("Models loaded.")
            except Exception as _e:  # the app still starts; models load on first use
                print(f"Preload skipped: {_e}")
        demo = create_demo()
        # In Kaggle or remote server, share=True generates a public gradio.live URL
        share_mode = os.environ.get("GRADIO_SHARE", "True").lower() == "true"
        port = int(os.environ.get("GRADIO_SERVER_PORT", 0)) or None
        demo.launch(share=share_mode, server_port=port)

