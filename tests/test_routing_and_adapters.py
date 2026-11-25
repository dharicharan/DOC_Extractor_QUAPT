"""
PRD v1.5 routing and specialist adapters (D14, FR-RTE-001..004, FR-CLS-005..007) and the v1.5 symbols
that this integration must keep. The OCR + text adapter is exercised with stubbed models only: it is
UNTESTED against the real GLM-OCR / Qwen3-4B (see adapters.py), and nothing here claims otherwise.
"""

import inspect
import json
from types import SimpleNamespace as NS
from typing import Any, Dict, List, Optional

from PIL import Image
import pytest

from doc_extractor import adapters, app
from doc_extractor import extractor as ex
from doc_extractor import routing
from doc_extractor.adapters import DEFAULT_OCR_MODEL, DEFAULT_TEXT_MODEL, ModelAdapter, OCRTextAdapter, VLMAdapter
from doc_extractor.contracts_c import ExecutionMode, ExecutionProvenance
from doc_extractor.routing import DocumentCategory, RoutedExtractor, heuristic_classify

TABLE = routing.DEFAULT_ROUTING_TABLE


def _prov(model="stub", provider="stub", **kw) -> ExecutionProvenance:
    return ExecutionProvenance(mode=ExecutionMode.GENUINE_LLM, model_name=model, provider=provider,
                               fallback_occurred=False, **kw)


class StubAdapter(ModelAdapter):
    def __init__(self, kind, label, grounded, fail=False):
        self.kind, self._label, self.grounded, self.fail = kind, label, grounded, fail
        self.calls: List[str] = []

    @property
    def model_label(self):
        return self._label

    def extract(self, image_path, system_prompt):
        self.calls.append(image_path)
        if self.fail:
            raise RuntimeError(f"{self._label} failed")
        return {"title": {"value": "T", "bbox": [1, 1, 9, 9] if self.grounded else None}}, _prov(self._label), {
            "grounded": self.grounded, "models": {"specialist": self._label}}


# -- the routing table -----------------------------------------------------------------------------

def test_digital_pages_route_to_ocr_plus_text_and_handwriting_to_the_vision_model():
    structured = routing.select_route(DocumentCategory.STRUCTURED, "en", TABLE)
    assert structured == {"adapter": "ocr_text", "ocr_model": DEFAULT_OCR_MODEL, "text_model": DEFAULT_TEXT_MODEL}
    assert routing.select_route(DocumentCategory.UNSTRUCTURED, "hi", TABLE)["adapter"] == "ocr_text"
    assert routing.select_route(DocumentCategory.HANDWRITTEN, "en", TABLE) == routing.DEFAULT_MODEL
    assert routing.DEFAULT_MODEL == "Qwen/Qwen3-VL-4B-Instruct" and TABLE["default_model"] == routing.DEFAULT_MODEL
    assert DEFAULT_OCR_MODEL == "zai-org/GLM-OCR" and DEFAULT_TEXT_MODEL == "Qwen/Qwen3-4B-Instruct-2507"


def test_alt_engines_offer_the_comparison_specialists():
    assert set(TABLE["alt_engines"]) == {"ocr_text", "ocr_text_7b"}
    assert TABLE["alt_engines"]["ocr_text_7b"]["text_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert app.ENGINE_CHOICES["Follow routing table"] is None
    # "page_text" is the whole-page OCR output option, not a comparison specialist
    assert set(filter(None, app.ENGINE_CHOICES.values())) == set(TABLE["alt_engines"]) | {routing.PAGE_TEXT_ENGINE}


def test_get_adapter_builds_and_caches_the_right_kind():
    routed = RoutedExtractor(table=TABLE, allow_fallback=False)
    ocr_route = TABLE["routes"]["structured"]["default"]
    a1, a2 = routed.get_adapter(ocr_route), routed.get_adapter(dict(ocr_route))
    assert isinstance(a1, OCRTextAdapter) and a1 is a2                       # cached per (ocr, text) pair
    assert a1.model_label == f"{DEFAULT_OCR_MODEL} + {DEFAULT_TEXT_MODEL}" and a1.kind == "ocr_text"
    vlm = routed.get_adapter(routing.DEFAULT_MODEL)
    assert isinstance(vlm, VLMAdapter) and vlm.kind == "vlm" and vlm.model_label == routing.DEFAULT_MODEL


def test_the_ocr_text_adapter_reads_then_maps_and_reports_that_it_has_no_boxes(monkeypatch):
    adapter = OCRTextAdapter()
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(adapter, "read_text", lambda path: "Gold Shield Plan")

    def text_to_fields(text, prompt):
        seen["text"], seen["prompt"] = text, prompt
        return {"title": {"value": "Gold Shield Plan", "bbox": None}}

    monkeypatch.setattr(adapter, "text_to_fields", text_to_fields)
    raw, prov, meta = adapter.extract("page.png", "PROMPT")

    assert seen == {"text": "Gold Shield Plan", "prompt": "PROMPT"} and raw["title"]["value"] == "Gold Shield Plan"
    assert meta["grounded"] is False and meta["models"]["ocr"] == DEFAULT_OCR_MODEL
    assert prov.provider == "transformers_ocr_text" and prov.details["ocr_chars"] == 16


# -- routed extraction, engine selection and fallback -----------------------------------------------

def test_extract_follows_the_table_per_category_and_reports_the_route(monkeypatch):
    routed = RoutedExtractor(table=TABLE, allow_fallback=False)
    ocr, vlm = StubAdapter("ocr_text", "ocr+text", False), StubAdapter("vlm", "vlm-model", True)
    monkeypatch.setattr(RoutedExtractor, "get_adapter",
                        lambda self, route: ocr if isinstance(route, dict) else vlm)

    _, _, info = routed.extract("p.png", DocumentCategory.STRUCTURED, "en")
    assert info["adapter"] == "ocr_text" and info["grounded"] is False and info["fallback_occurred"] is False
    assert info["routing_table_version"] == TABLE["version"] and info["model_used"] == "ocr+text"
    _, _, info = routed.extract("p.png", DocumentCategory.HANDWRITTEN, "en")
    assert info["adapter"] == "vlm" and info["grounded"] is True


def test_the_engine_choice_overrides_the_table_and_an_unknown_engine_is_an_error(monkeypatch):
    routed = RoutedExtractor(table=TABLE, allow_fallback=False)
    chosen: List[Any] = []
    monkeypatch.setattr(RoutedExtractor, "get_adapter",
                        lambda self, route: chosen.append(route) or StubAdapter("ocr_text", "x", False))
    routed.extract("p.png", DocumentCategory.HANDWRITTEN, "en", engine="ocr_text_7b")
    assert chosen[0] == TABLE["alt_engines"]["ocr_text_7b"]
    with pytest.raises(ValueError, match="unknown engine"):
        routed.extract("p.png", DocumentCategory.HANDWRITTEN, "en", engine="nope")


def test_a_failing_specialist_falls_back_to_the_default_model_and_says_so(monkeypatch):
    routed = RoutedExtractor(table=TABLE, allow_fallback=False)
    failing, default = StubAdapter("ocr_text", "ocr+text", False, fail=True), StubAdapter("vlm", "default-vlm", True)
    monkeypatch.setattr(RoutedExtractor, "get_adapter",
                        lambda self, route: failing if isinstance(route, dict) else default)
    _, prov, info = routed.extract("p.png", DocumentCategory.STRUCTURED, "en")
    assert info["fallback_occurred"] is True and info["model_used"] == "default-vlm"
    assert prov.fallback_occurred is True and "ocr+text failed" in prov.fallback_reason


def test_if_the_default_model_also_fails_the_error_surfaces(monkeypatch):
    routed = RoutedExtractor(table=TABLE, allow_fallback=False)
    monkeypatch.setattr(RoutedExtractor, "get_adapter", lambda self, route: StubAdapter("vlm", "m", True, fail=True))
    with pytest.raises(RuntimeError, match="failed"):
        routed.extract("p.png", DocumentCategory.HANDWRITTEN, "en")


# -- router, classification, rotation, filled_by ---------------------------------------------------

def test_get_router_uses_the_default_extractor_unless_a_router_model_is_configured(monkeypatch):
    monkeypatch.delenv("DOC_ROUTER_MODEL", raising=False)
    routed = RoutedExtractor(table=TABLE, allow_fallback=False)
    assert routed.router_model == routing.DEFAULT_MODEL
    assert routed.get_router() is routed.get_extractor(routing.DEFAULT_MODEL)
    monkeypatch.setenv("DOC_ROUTER_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    assert routed.router_model == "Qwen/Qwen3-VL-2B-Instruct"
    assert routed.get_router().model_name == "Qwen/Qwen3-VL-2B-Instruct"


def _router(monkeypatch, result: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None):
    def classify_page_image(image):
        if error:
            raise error
        return dict(result)

    monkeypatch.setattr(RoutedExtractor, "get_router", lambda self: NS(classify_page_image=classify_page_image))


def test_hand_filled_pages_are_forced_to_handwritten_in_code_and_rotation_is_returned(monkeypatch):
    _router(monkeypatch, {"filled_by": "hand", "category": "structured", "language": "en", "rotation": 90,
                          "document_type": "DOC-01", "confidence": 0.9})
    result = RoutedExtractor(table=TABLE, allow_fallback=False).classify("p.png", "scan.png")
    assert result["category"] == "handwritten" and result["category_before_rule"] == "structured"
    assert result["rotation"] == 90 and result["filled_by"] == "hand" and result["method"] == "vlm"


def test_an_unrecognised_category_falls_back_to_the_heuristic_without_inventing_a_type(monkeypatch):
    _router(monkeypatch, {"category": "banana"})
    result = RoutedExtractor(table=TABLE, allow_fallback=False).classify("p.png", "bank_account_form.pdf")
    assert result["document_type"] is None and result["confidence"] is None       # Phase 1: filenames never bind
    assert "fallback_reason" in result


def test_the_fallback_keeps_module_bs_type_and_its_real_confidence(monkeypatch):
    _router(monkeypatch, error=RuntimeError("model unavailable"))
    result = RoutedExtractor(table=TABLE, allow_fallback=False).classify("p.png", "x.png", "DOC-02", 0.83)
    assert result["document_type"] == "DOC-02" and result["confidence"] == 0.83 and result["method"] == "rule_classifier"


def test_the_classifier_prompt_asks_for_filled_by_and_rotation():
    prompt = routing.CLASSIFIER_SYSTEM_PROMPT
    assert '"filled_by"' in prompt and '"rotation"' in prompt and "CLOCKWISE" in prompt


# -- symbols and behaviour the v1.5 integration must preserve --------------------------------------

def test_v15_extractor_helpers_are_present_and_behave():
    assert ex.MAX_IMAGE_PIXELS == 1_300_000 and ex.CLASSIFY_MAX_PIXELS == 500_000
    assert ex.parse_json_tolerant('```json\n{"a": 1}\n```') == {"a": 1}
    assert ex.parse_json_tolerant('{"a": {"value": 1}, "b": {"value": 2}, "c": {"va') == {
        "a": {"value": 1}, "b": {"value": 2}}                                    # truncated output keeps complete fields
    assert ex._flat_to_nested({"applicant.full_name": {"value": "A"}, "other.x": {"value": 1}}) == {
        "applicant": {"full_name": "A"}}                                          # other.* keys are dropped
    assert callable(ex.gpu_dtype)
    big = Image.new("RGB", (2000, 2000), "white")                                 # 4 MP
    capped = ex.load_capped_image(big)
    assert capped.size[0] * capped.size[1] <= ex.MAX_IMAGE_PIXELS
    assert abs(capped.size[0] / capped.size[1] - 1.0) < 0.01
    small = ex.load_capped_image(Image.new("RGB", (100, 80)))
    assert small.size == (100, 80)                                                # below the cap: untouched


def test_adapters_module_keeps_both_adapters_and_the_shared_helpers():
    assert issubclass(VLMAdapter, ModelAdapter) and issubclass(OCRTextAdapter, ModelAdapter)
    source = inspect.getsource(adapters)
    for name in ("gpu_dtype", "load_capped_image", "parse_json_tolerant", "resolve_trust_remote_code"):
        assert name in source


def test_the_app_keeps_the_v15_ui_features():
    source = inspect.getsource(app)
    for token in ("gr.Gallery", "_render_steps", "_rotate_page", "ENGINE_CHOICES", "MAX_PAGES", "DOC_PRELOAD",
                  "get_routed_extractor", "ModuleCPipeline", "process_routed_pages"):
        assert token in source, token
    assert app.MAX_PAGES == 0                                                     # every page is read by default


def test_rotate_page_turns_a_page_clockwise(tmp_path):
    from doc_extractor.processing_service import RequestTemporaryWorkspace

    path = tmp_path / "wide.png"
    img = Image.new("RGB", (60, 20), "white")
    img.putpixel((0, 0), (255, 0, 0))                                             # top-left marker
    img.save(path)
    with RequestTemporaryWorkspace() as workspace:
        turned_path = app._rotate_page(path, 90, workspace=workspace)
        assert turned_path.exists()
        turned = Image.open(turned_path)
        assert turned.size == (20, 60)
        assert turned.getpixel((19, 0)) == (255, 0, 0)                            # top-left -> top-right when clockwise
    assert not turned_path.exists()


def test_the_streaming_steps_panel_tracks_state_and_timing():
    steps: List[Dict[str, Any]] = []
    app._step(steps, "a", "Step A", "running")
    app._step(steps, "a", state="done", detail="ok")
    app._step(steps, "b", "Step B")
    table = app._render_steps(steps)
    assert "Step A" in table and "ok" in table and "✅" in table and "⬜" in table
    assert app._render_steps([]).startswith("Upload a document")


def test_v15_schemas_arrive_intact_with_the_integration():
    from doc_extractor.schemas import ApplicantDetails, BankDetails, CASASchema

    assert {"first_name", "middle_name", "last_name", "aadhaar_number", "gross_annual_income"} <= set(
        ApplicantDetails.model_fields)
    assert {"account_number", "ifsc_code", "micr_code", "bank_name", "pin_code"} <= set(BankDetails.model_fields)
    assert "bank_details" in CASASchema.model_fields


def test_heuristic_classify_never_lets_a_filename_force_a_type_or_fabricate_confidence():
    for name in ("bank_account_form.pdf", "insurance_proposal.pdf", "casa_kyc.png", "prescription.png"):
        result = heuristic_classify(name)
        assert result["document_type"] is None and result["confidence"] is None
