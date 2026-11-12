"""
DOC-02 Insurance Application Form: field rules (PRD FR-EXT-003/005/007, FR-VAL-001/003, FR-CNF-002).

Field list follows src/doc_extractor/schemas.py (InsuranceApplicationSchema). Same rules as DOC-01:
  - null with a reason instead of guessing (FR-EXT-003),
  - repeating groups (nominees) as an ordered list (FR-EXT-005),
  - checkbox medical declarations as booleans, with review until checkbox evidence exists (FR-EXT-007),
  - arithmetic check: nominee shares must total 100 % (FR-VAL-003), otherwise review.

Status: rules and text parsing are tested with synthetic text only. No real DOC-02 form has been
tested yet, so accuracy on real forms is unknown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from doc_extractor.casa_form import FieldResult, parse_model_json

DOC_TYPE = "DOC-02"

REQUIRED = {"proposer_name"}
CHECKBOX_FIELDS = {"has_diabetes", "has_hypertension", "has_tobacco_use"}


@dataclass
class InsuranceResult:
    fields: Dict[str, FieldResult] = field(default_factory=dict)
    nominees: List[Dict[str, Any]] = field(default_factory=list)
    flags: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "needs_review"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": DOC_TYPE,
            "status": self.status,
            "flags": self.flags,
            "fields": {
                k: {"value": v.value, "null_reason": v.null_reason, "review_required": v.review_required,
                    "validation": v.validation, "error": v.error}
                for k, v in self.fields.items()
            },
            "nominees": self.nominees,
        }


def _text(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s or None


def _money(v: Any) -> Optional[float]:
    s = _text(v)
    if s is None:
        return None
    cleaned = re.sub(r"[^0-9.]", "", s)
    try:
        return round(float(cleaned), 2) if cleaned else None
    except ValueError:
        return None


def normalise_insurance_fields(model_output: Any) -> InsuranceResult:
    """Takes the model's JSON (dict or raw text) and returns the DOC-02 result with review routing."""
    if isinstance(model_output, str):
        parsed = parse_model_json(model_output)
        if parsed is None:
            res = InsuranceResult(flags=[{"code": "MODEL_OUTPUT_UNPARSEABLE", "fields": ["all"]}])
            for name in ("proposer_name", "policy_term_years", "sum_insured"):
                res.fields[name] = FieldResult(null_reason="illegible", review_required=True)
            return res
        model_output = parsed

    res = InsuranceResult()

    # Proposer name: letters only.
    name = _text(model_output.get("proposer_name"))
    if name and re.fullmatch(r"[A-Za-z .'\-]{3,}", name):
        res.fields["proposer_name"] = FieldResult(value=name.upper())
    else:
        res.fields["proposer_name"] = FieldResult(null_reason="absent" if name is None else "illegible",
                                                  review_required=True, error=None if name is None else "NAME_FORMAT")

    # Policy term: whole years, 1 to 99.
    term = _money(model_output.get("policy_term_years"))
    if term is not None and float(term).is_integer() and 1 <= term <= 99:
        res.fields["policy_term_years"] = FieldResult(value=int(term), validation="passed")
    else:
        res.fields["policy_term_years"] = FieldResult(
            null_reason="absent" if model_output.get("policy_term_years") is None else "illegible",
            review_required=model_output.get("policy_term_years") is not None, error="TERM_FORMAT" if model_output.get("policy_term_years") is not None else None)

    # Sum insured: amount in rupees.
    amount = _money(model_output.get("sum_insured"))
    if amount is not None and amount > 0:
        res.fields["sum_insured"] = FieldResult(value=amount, validation="passed")
    else:
        res.fields["sum_insured"] = FieldResult(
            null_reason="absent" if model_output.get("sum_insured") is None else "illegible",
            review_required=model_output.get("sum_insured") is not None, error="AMOUNT_FORMAT" if model_output.get("sum_insured") is not None else None)

    # Medical declaration: booleans from checkboxes. Review until checkbox evidence exists.
    for key in CHECKBOX_FIELDS:
        raw = model_output.get(key)
        if isinstance(raw, bool):
            res.fields[key] = FieldResult(value=raw, review_required=True, error="CHECKBOX_EVIDENCE_NOT_AVAILABLE")
        else:
            res.fields[key] = FieldResult(null_reason="absent", review_required=True)

    # Nominees: ordered list, name required, shares must total 100 when present.
    nominees = []
    for item in model_output.get("nominees") or []:
        if not isinstance(item, dict):
            continue
        nm = _text(item.get("name"))
        share = _money(item.get("share_percentage"))
        nominees.append({
            "name": nm.upper() if nm else None,
            "relationship": _text(item.get("relationship")),
            "share_percentage": share,
        })
    res.nominees = nominees
    if nominees:
        if any(n["name"] is None for n in nominees):
            res.flags.append({"code": "NOMINEE_NAME_MISSING", "fields": ["nominees"]})
        shares = [n["share_percentage"] for n in nominees if n["share_percentage"] is not None]
        if shares and abs(sum(shares) - 100.0) > 0.01:
            res.flags.append({"code": "NOMINEE_SHARE_MISMATCH", "fields": ["nominees"], "total": round(sum(shares), 2)})

    flagged = [k for k, f in res.fields.items() if f.review_required]
    if flagged or any(f["code"] in ("NOMINEE_NAME_MISSING", "NOMINEE_SHARE_MISMATCH") for f in res.flags):
        res.flags.append({"code": "FIELDS_REQUIRE_REVIEW", "fields": flagged})
        res.status = "needs_review"
    else:
        res.status = "auto_approved"
    return res
