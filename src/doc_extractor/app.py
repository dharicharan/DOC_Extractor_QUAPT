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
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from doc_extractor.config import Settings
from doc_extractor.contracts_b import MVPDocumentType
from doc_extractor.extractor import SchemaGuidedExtractor, SCHEMA_MAP
from doc_extractor.handoff import CoordinateTransform
from doc_extractor.pipeline import ModuleAPipeline
from doc_extractor.pipeline_b import ModuleBPipeline
from doc_extractor.pipeline_c import ModuleCPipeline
from doc_extractor.storage import LocalDiskStorage

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

    # Color palette for bounding boxes
    colors = [
        "#00E676", "#2979FF", "#FF9100", "#FF1744",
        "#651FFF", "#00E5FF", "#F50057", "#76FF03",
    ]

    for idx, (field_name, field_data) in enumerate(fields.items()):
        # ExtractedField instance or dict
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


def process_document(
    file_obj,
    doc_type_override: str,
    use_qwen_vl: bool,
) -> Tuple[Optional[Image.Image], str, str]:
    """Processes document and returns (annotated_page_image, status_text, json_output)."""
    if file_obj is None:
        return None, "No file uploaded", "{}"

    pipe_a, pipe_b, pipe_c = get_pipeline()

    # Toggle Qwen-VL model usage
    pipe_c.extractor.use_llm = use_qwen_vl

    # Handle file path
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    claimed_name = Path(file_path).name

    try:
        # 1. Ingestion & Preprocessing (Module A)
        payload_a = pipe_a.process_document(file_path, claimed_filename=claimed_name)
        if payload_a.status == "rejected":
            return None, f"Rejected: {payload_a.error}", json.dumps(payload_a.to_dict(), indent=2)

        # 2. Classification & Parsing (Module B)
        payload_b = pipe_b.process_handoff(payload_a)

        # Optional manual doc type override
        if doc_type_override and doc_type_override != "Auto-Detect":
            target_type = MVPDocumentType(doc_type_override)
            payload_b.classification.document_type = target_type
            payload_b.classification.schema_id = doc_type_override
            payload_b.status = "processing"

        # 3. Extraction & Grounding (Module C)
        envelope = pipe_c.process(payload_b)

        # 4. Render first page with bounding boxes
        annotated_img = None
        if payload_a.pages:
            first_page_rec = payload_a.pages[0]
            orig_path = pipe_a.storage.resolve_uri_to_path(first_page_rec.original_storage_uri)
            if orig_path.exists():
                with Image.open(orig_path) as pil_img:
                    annotated_img = draw_bounding_boxes(pil_img, envelope.fields)

        status_msg = (
            f"Status: {envelope.document_metadata.status.upper()} | "
            f"Type: {envelope.document_metadata.document_type} | "
            f"Confidence: {envelope.document_metadata.classification_confidence:.2f} | "
            f"Flags: {len(envelope.flags)}"
        )

        return annotated_img, status_msg, json.dumps(envelope.model_dump(), indent=2, default=str)

    except Exception as e:
        return None, f"Error: {e}", json.dumps({"error": str(e)}, indent=2)


def create_demo() -> "gr.Blocks":
    """Constructs Gradio UI layout."""
    with gr.Blocks(title="Document Extraction Engine — Qwen-VL Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 📄 Document Extraction Engine\n"
            "**Single-Stage Multimodal Extraction (Qwen3-VL / Qwen2.5-VL)** with deterministic financial normalization and pixel evidence grounding."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(label="Upload Document (PDF, JPG, PNG, TIFF)", file_types=[".pdf", ".jpg", ".png", ".tiff"])
                doc_type_dropdown = gr.Dropdown(
                    choices=["Auto-Detect", "DOC-01", "DOC-02", "DOC-03"],
                    value="Auto-Detect",
                    label="Document Type Hint",
                )
                use_vlm_checkbox = gr.Checkbox(
                    value=False,
                    label="Enable Live Qwen-VL Model (Checked = GPU/vLLM, Unchecked = Heuristic fast-path)",
                )
                submit_btn = gr.Button("Extract Document", variant="primary")

            with gr.Column(scale=2):
                status_box = gr.Textbox(label="Status & Quality Gate", interactive=False)
                with gr.Tab("Visual Grounding"):
                    output_image = gr.Image(label="Annotated Document Page (with Grounded Bounding Boxes)", type="pil")
                with gr.Tab("Canonical JSON Output"):
                    output_json = gr.Code(label="Extracted §6.3 JSON Envelope", language="json")

        submit_btn.click(
            fn=process_document,
            inputs=[input_file, doc_type_dropdown, use_vlm_checkbox],
            outputs=[output_image, status_box, output_json],
        )

    return demo


if __name__ == "__main__":
    if gr is None:
        print("Gradio is not installed. Run: pip install gradio")
    else:
        demo = create_demo()
        # In Kaggle or remote server, share=True generates a public gradio.live URL
        share_mode = os.environ.get("GRADIO_SHARE", "True").lower() == "true"
        demo.launch(share=share_mode, server_port=7860)

