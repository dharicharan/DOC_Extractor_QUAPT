"""DOC-02 insurance application rules (src/doc_extractor/insurance_form.py). Synthetic inputs only."""

from doc_extractor.insurance_form import normalise_insurance_fields

BASE = {
    "proposer_name": "Anita Rao",
    "policy_term_years": 20,
    "sum_insured": "5,00,000",
    "nominees": [
        {"name": "Ravi Rao", "relationship": "Spouse", "share_percentage": 60},
        {"name": "Meena Rao", "relationship": "Daughter", "share_percentage": 40},
    ],
    "has_diabetes": False,
    "has_hypertension": False,
    "has_tobacco_use": False,
}


def test_complete_form_parses_and_routes_checkboxes_to_review():
    res = normalise_insurance_fields(BASE)
    assert res.fields["proposer_name"].value == "ANITA RAO"
    assert res.fields["policy_term_years"].value == 20
    assert res.fields["sum_insured"].value == 500000.0
    assert [n["name"] for n in res.nominees] == ["RAVI RAO", "MEENA RAO"]
    # Checkbox evidence is not implemented yet, so the document is not auto-approved.
    assert res.fields["has_diabetes"].review_required is True
    assert res.status == "needs_review"


def test_missing_proposer_name_is_null_and_routed_to_review():
    res = normalise_insurance_fields(dict(BASE, proposer_name=None))
    assert res.fields["proposer_name"].value is None
    assert res.fields["proposer_name"].review_required is True


def test_nominee_shares_not_totalling_100_are_flagged():
    data = dict(BASE, nominees=[{"name": "Ravi Rao", "share_percentage": 60},
                                {"name": "Meena Rao", "share_percentage": 30}])
    res = normalise_insurance_fields(data)
    assert any(f["code"] == "NOMINEE_SHARE_MISMATCH" and f["total"] == 90.0 for f in res.flags)
    assert res.status == "needs_review"


def test_nominee_without_name_is_flagged():
    res = normalise_insurance_fields(dict(BASE, nominees=[{"name": None, "share_percentage": 100}]))
    assert any(f["code"] == "NOMINEE_NAME_MISSING" for f in res.flags)


def test_out_of_range_policy_term_is_not_guessed():
    res = normalise_insurance_fields(dict(BASE, policy_term_years=250))
    assert res.fields["policy_term_years"].value is None
    assert res.fields["policy_term_years"].review_required is True


def test_unparseable_model_output_sends_fields_to_review():
    res = normalise_insurance_fields("no json here")
    assert res.status == "needs_review"
    assert res.flags[0]["code"] == "MODEL_OUTPUT_UNPARSEABLE"
