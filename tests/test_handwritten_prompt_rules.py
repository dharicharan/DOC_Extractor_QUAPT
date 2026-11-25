"""The handwritten prompt carries the reading rules behind PRD FR-EXT-003, FR-EXT-005 and FR-PRS-006."""
from doc_extractor.routing import DocumentCategory, build_generic_prompt


def test_handwritten_prompt_has_the_reading_rules():
    p = build_generic_prompt(DocumentCategory.HANDWRITTEN)
    assert "Never copy a value from one field into another" in p          # PAN copied into Aadhaar
    assert "every word, initial and digit" in p                           # dropped surname
    assert "Never turn printed text or a stamp into true/false" in p      # FR-EXT-003: no inferred values
    assert "struck out, overwritten or corrected" in p                    # FR-PRS-006
    assert "list of objects, in page order" in p                          # FR-EXT-005


def test_digital_prompts_are_unchanged_by_the_handwriting_rules():
    for c in (DocumentCategory.STRUCTURED, DocumentCategory.UNSTRUCTURED):
        assert "READING RULES (handwriting)" not in build_generic_prompt(c)


def test_the_word_null_from_the_model_is_treated_as_no_value():
    """FR-EXT-003: a field the model fills with the string "null" / "N/A" / "-" is absent, not a value."""
    from doc_extractor.vlm_grounding import collect_generic_leaves

    raw = {
        "name": {"value": "AJITSINH PARMAR", "bbox": [50, 170, 300, 195]},
        "staff_no": {"value": "null", "bbox": [50, 500, 150, 520]},
        "passport": {"value": "N/A", "bbox": [50, 880, 150, 900]},
        "jointly": {"value": " None ", "bbox": [250, 716, 350, 742]},
        "dash": {"value": "-", "bbox": [1, 1, 2, 2]},
    }
    leaves = collect_generic_leaves(raw)
    assert set(leaves) == {"name"}


def test_prompt_forbids_null_strings_and_rounded_boxes():
    p = build_generic_prompt(DocumentCategory.HANDWRITTEN)
    assert 'Never write the word \\"null\\"' in p or "Never write the word \"null\"" in p
    assert "not numbers rounded to 50 or 100" in p


def test_full_page_text_option_maps_to_the_page_text_engine():
    from doc_extractor import app
    from doc_extractor.processing_service import resolve_engine_choice

    label = "Full page text (OCR as JSON, no field picking)"
    assert resolve_engine_choice(label) == "page_text" and app.ENGINE_CHOICES[label] == "page_text"


def test_page_text_prompt_asks_for_all_text_and_names_the_category_and_subtype():
    from doc_extractor.routing import build_page_text_prompt

    p = build_page_text_prompt(DocumentCategory.UNSTRUCTURED, "menu card", "en")
    assert "Transcribe ALL the text" in p and "unstructured (menu card)" in p and "reading order" in p


def test_router_reads_page_text_with_the_vision_model_and_a_longer_output(monkeypatch):
    from doc_extractor.routing import RoutedExtractor

    seen = {}

    class _Extractor:
        max_new_tokens = 2048
        model_name = "fake-vlm"

        def extract_page_image(self, image_path, schema, system_prompt=None):
            from doc_extractor.contracts_c import ExecutionMode, ExecutionProvenance

            seen.update(prompt=system_prompt, tokens=self.max_new_tokens)
            prov = ExecutionProvenance(mode=ExecutionMode.GENUINE_LLM, model_name="fake-vlm", provider="x",
                                       prompt_version="1.0.0", fallback_occurred=False)
            return {"heading_001": {"value": "MENU", "bbox": [1, 1, 9, 9]}}, prov

    ex = _Extractor()
    routed = RoutedExtractor()
    monkeypatch.setattr(routed, "get_extractor", lambda *a, **k: ex)
    raw, _, info = routed.extract("page.png", DocumentCategory.UNSTRUCTURED, subtype="menu card", engine="page_text")
    assert "Transcribe ALL the text" in seen["prompt"] and seen["tokens"] == 4096 and ex.max_new_tokens == 2048
    assert info["adapter"] == "vlm" and raw["heading_001"]["value"] == "MENU"


def test_page_text_json_keeps_every_block_per_page_in_order():
    from types import SimpleNamespace

    from doc_extractor.app import _page_text_json

    def page(n, raw):
        return SimpleNamespace(page_number=n, raw=raw, error=None)

    result = SimpleNamespace(
        envelope=SimpleNamespace(document_metadata=SimpleNamespace(
            document_category="unstructured", document_subtype="menu card", language="en")),
        page_results=[
            page(1, {"heading_001": {"value": "MENU", "bbox": [1, 1, 9, 9]},
                     "list_item_002": {"value": "Idli 40", "bbox": [1, 10, 9, 19]},
                     "text_003": {"value": None, "bbox": None}}),
            page(2, {"heading_001": {"value": "DRINKS", "bbox": [1, 1, 9, 9]}}),
        ],
    )
    doc = _page_text_json(result)
    assert doc["document_subtype"] == "menu card"
    assert [b["type"] for b in doc["pages"][0]["blocks"]] == ["heading", "list_item"]
    assert doc["pages"][0]["full_text"] == "MENU\nIdli 40" and doc["pages"][1]["full_text"] == "DRINKS"
