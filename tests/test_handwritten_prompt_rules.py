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
