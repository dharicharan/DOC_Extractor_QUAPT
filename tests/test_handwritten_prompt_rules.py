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
