"""Crop-and-verify pass: OpenCV box snapping and crop re-read agreement (no model, no GPU)."""
from PIL import Image, ImageDraw, ImageFont

from doc_extractor import crop_verify as cv


def _page():
    img = Image.new("RGB", (1000, 1400), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=30)
    d.line([(60, 200), (940, 200)], fill="black", width=2)          # a form rule: must not become text
    d.text((300, 300), "Rajesh Kumar Sharma", fill="black", font=font)
    d.text((300, 420), "Vikram Sharma", fill="black", font=font)
    return img


def test_detect_finds_text_lines_and_ignores_form_rules():
    boxes = cv.detect_text_boxes(_page())
    assert boxes, "no text found"
    assert all(not (b[3] - b[1] < 5 and b[2] - b[0] > 500) for b in boxes), "form rule detected as text"


def test_snap_moves_a_drifted_box_onto_the_text_line():
    detected = [(300.0, 214.0, 420.0, 226.0)]
    drifted = (290.0, 205.0, 430.0, 222.0)   # one row high
    snapped = cv.snap_bbox(drifted, detected)
    assert abs(snapped[1] - 214.0) < 1.0 and abs(snapped[3] - 226.0) < 1.0


def test_snap_keeps_the_box_when_nothing_is_near():
    box = (10.0, 10.0, 50.0, 30.0)
    assert cv.snap_bbox(box, [(600.0, 600.0, 700.0, 620.0)]) == box


def test_texts_agree_ignores_case_and_punctuation_but_not_missing_words():
    assert cv.texts_agree("ABCDE 1234-F", "abcde1234f")
    assert cv.texts_agree("Rajesh Kumar Sharma", "Rajesh Kumar Sharma")
    assert not cv.texts_agree("Mrs. Priya - Wife", "Mrs. Priya Sharma - Wife")
    assert not cv.texts_agree("ABCDE1234F", "1254 5678 9101")


def test_refine_halves_confidence_on_disagreement_and_keeps_it_on_agreement():
    raw = {
        "applicant_name": {"value": "Rajesh Kumar Sharma", "bbox": [290, 205, 430, 222], "confidence": 0.9},
        "fathers_name": {"value": "Vikram Sharma", "bbox": [290, 300, 430, 312], "confidence": 0.9},
        "missing": {"value": None, "bbox": None},
    }
    reads = iter(["Rajesh Kumar Sharma", "Suresh Patel"])
    stats = cv.refine_raw(raw, _page(), lambda crop: next(reads))
    assert stats["verified"] == 2 and stats["agreed"] == 1 and stats["disagreed"] == 1
    assert raw["applicant_name"]["confidence"] == 0.9 and raw["applicant_name"]["crop_verified"] is True
    assert raw["fathers_name"]["confidence"] == 0.45 and raw["fathers_name"]["uncertain"] is True
    assert raw["fathers_name"]["crop_text"] == "Suresh Patel"


def test_refine_without_reader_only_snaps_and_never_raises():
    raw = {"a": {"value": "x", "bbox": [1, 1, 2, 2], "confidence": 0.9}}
    stats = cv.refine_raw(raw, _page(), None)
    assert stats["verified"] == 0 and raw["a"]["confidence"] == 0.9


def test_disabled_unless_flag_set(monkeypatch):
    monkeypatch.delenv("DOC_CROP_VERIFY", raising=False)
    assert not cv.enabled()
    monkeypatch.setenv("DOC_CROP_VERIFY", "1")
    assert cv.enabled()


def test_two_close_lines_are_not_merged_into_one_region():
    img = Image.new("RGB", (1000, 1400), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=30)
    d.text((300, 300), "Rajesh Kumar Sharma", fill="black", font=font)
    d.text((300, 338), "12/03/1985", fill="black", font=font)
    column = [b for b in cv.detect_text_boxes(img) if b[0] < 450 < b[2]]
    assert len(column) >= 2, column


def test_crop_table_shows_each_crop_its_page_reading_and_crop_text():
    cv.begin_run()
    items = []
    cv.refine_raw({"a": {"value": "Rajesh Kumar Sharma", "bbox": [290, 205, 430, 222], "confidence": 0.9}},
                  _page(), lambda crop: "Rajesh Sharma", items_out=items)
    cv.record(items)
    md = cv.render_markdown(cv.collect())
    assert "Crop check" in md and "data:image/png;base64," in md
    assert "Rajesh Kumar Sharma" in md and "Rajesh Sharma" in md and "review" in md
    cv.begin_run()
    assert cv.render_markdown(cv.collect()) == ""


def test_a_snap_that_blows_the_box_up_is_rejected():
    vlm = (100.0, 500.0, 300.0, 520.0)
    strip = (0.0, 495.0, 700.0, 525.0)            # page-wide ruled strip
    assert cv.snap_bbox(vlm, [strip]) == vlm


def test_two_fields_never_share_one_snapped_line():
    raw = {"a": {"value": "x", "bbox": [290, 206, 430, 222]}, "b": {"value": "y", "bbox": [290, 207, 430, 223]}}
    cv.refine_raw(raw, _page(), None)
    assert raw["a"]["bbox"] != raw["b"]["bbox"] or raw["a"]["bbox"] == [290, 206, 430, 222]


def test_failed_crop_reads_are_counted_and_reported():
    raw = {"a": {"value": "Rajesh Kumar Sharma", "bbox": [290, 205, 430, 222], "confidence": 0.9}}

    def boom(crop):
        raise RuntimeError("model said no")

    stats = cv.refine_raw(raw, _page(), boom)
    assert stats["errors"] == 1 and "model said no" in stats["first_error"] and stats["verified"] == 0
