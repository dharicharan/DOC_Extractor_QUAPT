"""Tests for the DOC-01 CASA field rules (src/doc_extractor/casa_form.py)."""

import json

from doc_extractor.casa_form import (
    FIELD_SPEC,
    normalise_casa_fields,
    parse_model_json,
)

# Values checked by eye against the form images (Priya Sengupta form).
FORM_PRIYA = {
    "applicant_name": "PRIYA SENGUPTA",
    "date_of_birth": "28-11-1992",
    "pan": "BPWPS8491K",
    "mobile": "9830154672",
    "account_type": "SAVINGS",
    "initial_deposit_amount": "10,000.00",
    "opt_in_cheque_book": True,
    "opt_in_debit_card": True,
    "opt_in_internet_banking": True,
}


def test_complete_form_is_auto_approved_with_normalised_values():
    res = normalise_casa_fields(FORM_PRIYA)
    assert res.status == "auto_approved"
    assert res.flags == []
    assert res.fields["date_of_birth"].value == "1992-11-28"
    assert res.fields["pan"].value == "BPWPS8491K"
    assert res.fields["pan"].validation == "passed"
    assert res.fields["mobile"].value == "+919830154672"  # E.164 (FR-VAL-001)
    assert res.fields["initial_deposit_amount"].value == 10000.0
    assert res.fields["account_type"].value == "SAVINGS"


def test_missing_account_type_is_null_and_routed_to_review():
    data = dict(FORM_PRIYA, account_type=None)
    res = normalise_casa_fields(data)
    assert res.status == "needs_review"
    assert res.fields["account_type"].value is None
    assert res.fields["account_type"].null_reason == "absent"
    assert "account_type" in res.flags[0]["fields"]


def test_invalid_pan_is_not_guessed():
    data = dict(FORM_PRIYA, pan="BPWPS8491")  # 9 characters
    res = normalise_casa_fields(data)
    pan = res.fields["pan"]
    assert pan.value is None
    assert pan.validation == "failed"
    assert pan.review_required is True
    assert res.status == "needs_review"


def test_name_with_digits_is_routed_to_review():
    res = normalise_casa_fields(dict(FORM_PRIYA, applicant_name="PR1YA"))
    assert res.fields["applicant_name"].value is None
    assert res.fields["applicant_name"].review_required is True


def test_checkbox_strings_are_parsed():
    data = dict(FORM_PRIYA, opt_in_cheque_book="false", opt_in_debit_card="TRUE")
    res = normalise_casa_fields(data)
    assert res.fields["opt_in_cheque_book"].value is False
    assert res.fields["opt_in_debit_card"].value is True


def test_json_wrapped_in_prose_is_parsed():
    text = "Here is the result:\n```json\n" + json.dumps(FORM_PRIYA) + "\n```\nDone."
    res = normalise_casa_fields(text)
    assert res.status == "auto_approved"
    assert res.fields["pan"].value == "BPWPS8491K"


def test_unparseable_output_sends_every_field_to_review():
    res = normalise_casa_fields("I could not read this form.")
    assert res.status == "needs_review"
    assert res.flags[0]["code"] == "MODEL_OUTPUT_UNPARSEABLE"
    assert all(f.review_required for f in res.fields.values())
    assert set(res.fields) == set(FIELD_SPEC)


def test_parse_model_json_rejects_non_object():
    assert parse_model_json("[1, 2, 3]") is None
    assert parse_model_json("") is None


from doc_extractor.casa_form import ground_fields  # noqa: E402

PRINTED_TOKENS = [
    {"text": "PRIYA SENGUPTA", "bbox": [100, 200, 500, 240], "page": 1},
    {"text": "28-11-1992", "bbox": [100, 300, 300, 330], "page": 1},
    {"text": "BPWPS8491K", "bbox": [100, 400, 300, 430], "page": 1},
    {"text": "9830154672", "bbox": [100, 500, 300, 530], "page": 1},
    {"text": "10,000.00", "bbox": [100, 600, 300, 630], "page": 1},
]


def test_grounded_values_get_evidence_and_keep_their_value():
    res = ground_fields(normalise_casa_fields(FORM_PRIYA), PRINTED_TOKENS)
    assert res.fields["date_of_birth"].value == "1992-11-28"
    assert res.fields["date_of_birth"].evidence["bbox"] == [100, 300, 300, 330]
    assert res.fields["mobile"].evidence is not None
    assert res.fields["initial_deposit_amount"].evidence is not None


def test_value_not_printed_on_page_is_removed_and_routed_to_review():
    tokens = [t for t in PRINTED_TOKENS if t["text"] != "BPWPS8491K"]
    res = ground_fields(normalise_casa_fields(FORM_PRIYA), tokens)
    pan = res.fields["pan"]
    assert pan.value is None
    assert pan.error == "EVIDENCE_MISSING"
    assert pan.review_required is True
    assert res.status == "needs_review"
    assert any(f["code"] == "EVIDENCE_MISSING" and "pan" in f["fields"] for f in res.flags)


def test_checkbox_values_are_not_auto_approved_without_checkbox_evidence():
    res = ground_fields(normalise_casa_fields(FORM_PRIYA), PRINTED_TOKENS)
    assert res.fields["opt_in_cheque_book"].review_required is True
    assert res.status == "needs_review"


def test_underage_applicant_is_routed_to_review():
    res = normalise_casa_fields(dict(FORM_PRIYA, date_of_birth="15-08-2015"))
    dob = res.fields["date_of_birth"]
    assert dob.review_required is True
    assert dob.error == "UNDERAGE_APPLICANT"
    assert res.status == "needs_review"


from doc_extractor.casa_form import fields_from_ocr_text  # noqa: E402

RAHUL_PAGE_TEXT = """Name
Mr.
Cor.
Applicant
Name
RAHUL
SHARMA
DATE OF BIRTH
15 - 08- 1995
PA
Validation Field
PAN NO.
GBLPS8743A
(Validation Field)
MOBILE NO.
9876543210
Mobilee No.
SALARIED
Amount:
5000.00
Currency:
INR
Signature of Applicant:
Rahul Sharma"""

PRIYA_PAGE_TEXT = """NAME:
☑ Mr.
☐ v
☐ P R I Y A
S E N G U P T A
DATE OF BIRTH:
2 8 - 1 1 - 1 9 9 2
PAN NO.
PA
B P W P S 8 4 9 1 K
MOBILE NO.
9 8 3 0 1 5 4 6 7 2
Amount: 10,000.00 Currency: INR Cash dep: ☑ Payment: INR"""


def test_page_text_from_rahul_form_is_read_correctly():
    f = fields_from_ocr_text(RAHUL_PAGE_TEXT)
    assert f["applicant_name"] == "RAHUL SHARMA"
    assert f["date_of_birth"] == "15-08-1995"
    assert f["pan"] == "GBLPS8743A"
    assert f["mobile"] == "9876543210"
    assert f["initial_deposit_amount"] == "5000.00"
    assert f["account_type"] is None  # not in the text: review


def test_spaced_letters_and_checkbox_glyphs_are_handled():
    f = fields_from_ocr_text(PRIYA_PAGE_TEXT)
    assert f["applicant_name"] == "PRIYA SENGUPTA"
    assert f["date_of_birth"] == "28-11-1992"
    assert f["pan"] == "BPWPS8491K"
    assert f["mobile"] == "9830154672"
    assert f["initial_deposit_amount"] == "10,000.00"


def test_page_text_goes_through_rules_and_review():
    res = normalise_casa_fields(fields_from_ocr_text(RAHUL_PAGE_TEXT))
    assert res.fields["pan"].value == "GBLPS8743A"
    assert res.fields["account_type"].review_required is True
    assert res.status == "needs_review"


VIKRAM_PAGE_TEXT = """Name: ✓ Mr. □ Mr. ✓ □ I ✓ K ✓ A □ M ✓ R □ A ✓ T ✓ H ✓ O ✓ R ✓ E
Date of Birth: 2 4 - 0 9 - 1 9 9 1
PAN No. ✓ PA □ PA □ FN K G T P D 3 2 9 8 F Validation Field
MOBILE No. 9 8 4 0 1 2 3 4 5 6 Mobile No. Mobie No. SALARIED
ACCOUNT DETAILS: ✓ SAVINGS □ Safafilat Date of points
Address: □ Otter: ___ Occupation: SOFTWARE ENGINEER
INITIAL DEPOSIT: Amount: 12,000.00 Currency: INR Cash dep: ✓ Payment: INR
OPT-IN SERVICES: ✓ Cheque book ✓ Debit card ✓ Internet banking"""


def test_page_text_with_label_values_and_tick_glyphs():
    f = fields_from_ocr_text(VIKRAM_PAGE_TEXT)
    assert f["pan"] == "KGTPD3298F"
    assert f["mobile"] == "9840123456"
    assert f["initial_deposit_amount"] == "12,000.00"
    assert f["account_type"] == "SAVINGS"
    assert f["opt_in_cheque_book"] is True
    assert f["opt_in_debit_card"] is True
    assert f["opt_in_internet_banking"] is True
    assert "RATHORE" in f["applicant_name"]
    assert "MR" not in f["applicant_name"].split()


def test_empty_box_glyph_is_false_and_missing_tick_is_null():
    f = fields_from_ocr_text("☐ Cheque book\nDebit card\n")
    assert f["opt_in_cheque_book"] is False
    assert f["opt_in_debit_card"] is None
