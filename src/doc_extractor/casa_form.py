"""
DOC-01 CASA Account Opening Form: field rules (PRD FR-EXT-001/003/007, FR-VAL-001/002, FR-CNF-002).

Template-independent. A model (for example PaddleOCR-VL) reads the page and returns JSON
for these fields. This module:
  - parses that JSON defensively (the model may add prose or code fences),
  - normalises and validates every value with the shared validators,
  - returns null with a reason instead of guessing (FR-EXT-003),
  - routes any value that is missing, malformed or uncertain to review (FR-CNF-002).

No values are hardcoded here. Nothing in this module knows about a particular sample form.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from datetime import date

from doc_extractor.validation_rules import (
    normalize_date_to_iso,
    normalize_phone_number,
    validate_pan,
)


def _adult_check(dob_iso: str, min_age_years: int = 18) -> Optional[str]:
    """Rule R1 (schema): returns an error code when the applicant is under min_age_years."""
    try:
        y, m, d = (int(x) for x in dob_iso.split("-"))
        dob = date(y, m, d)
    except ValueError:
        return None  # not a parseable date: handled by the date validator
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return "UNDERAGE_APPLICANT" if age < min_age_years else None

DOC_TYPE = "DOC-01"

# Field name -> (kind, description given to the model)
FIELD_SPEC: Dict[str, tuple] = {
    "applicant_name": ("text", "Applicant full name, as printed in the boxes, uppercase, words separated by single spaces"),
    "date_of_birth": ("text", "Date of birth exactly as written (for example 28-11-1992 or 07/03/1991)"),
    "pan": ("text", "PAN card number, 10 characters: 5 letters, 4 digits, 1 letter"),
    "mobile": ("text", "Mobile number, digits only"),
    "account_type": ("enum:SAVINGS|CURRENT", "Account type that is ticked: SAVINGS or CURRENT"),
    "initial_deposit_amount": ("text", "Initial deposit amount as written, digits and decimal point only"),
    "opt_in_cheque_book": ("bool", "true if the Cheque book box is ticked, false if it is empty"),
    "opt_in_debit_card": ("bool", "true if the Debit card box is ticked, false if it is empty"),
    "opt_in_internet_banking": ("bool", "true if the Internet banking box is ticked, false if it is empty"),
}

PROMPT = (
    "This is a CASA account opening form. Read it and return only a JSON object with these keys: "
    + ", ".join(FIELD_SPEC)
    + ". For each key use the value written on the form. Use null if the field is blank or unreadable. "
    "Do not guess. Checkboxes are true when ticked and false when empty. Return only the JSON."
)


@dataclass
class FieldResult:
    value: Any = None
    null_reason: Optional[str] = None      # "absent" | "illegible" | None
    review_required: bool = False
    validation: str = "not_applicable"     # "passed" | "failed" | "not_applicable"
    error: Optional[str] = None
    raw: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None  # {"page", "bbox", "text"} (PRD FR-EXT-004)


@dataclass
class CasaFormResult:
    fields: Dict[str, FieldResult] = field(default_factory=dict)
    flags: list = field(default_factory=list)
    status: str = "needs_review"           # "auto_approved" only when nothing needs review

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": DOC_TYPE,
            "status": self.status,
            "flags": self.flags,
            "fields": {
                k: {
                    "value": v.value,
                    "null_reason": v.null_reason,
                    "review_required": v.review_required,
                    "validation": v.validation,
                    "error": v.error,
                    "evidence": v.evidence,
                }
                for k, v in self.fields.items()
            },
        }


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_model_json(text: str) -> Optional[Dict[str, Any]]:
    """Extracts the first JSON object from model output. Returns None if there is none."""
    if not text:
        return None
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = re.sub(r"\s+", " ", str(value)).strip()
    return s or None


# Required fields: if empty, the document goes to review (FR-CNF-003).
REQUIRED_FIELDS = {"applicant_name", "date_of_birth", "account_type"}


def _normalise_field(name: str, raw: Any) -> FieldResult:
    kind = FIELD_SPEC[name][0]
    res = FieldResult(raw=None if raw is None else str(raw))

    if kind == "bool":
        if isinstance(raw, bool):
            res.value = raw
        elif isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
            res.value = raw.strip().lower() == "true"
        else:
            res.null_reason = "absent" if raw is None else "illegible"
            res.review_required = True
        return res

    text = _clean_text(raw)
    if text is None:
        res.null_reason = "absent"
        res.review_required = name in REQUIRED_FIELDS
        return res

    if kind.startswith("enum:"):
        allowed = kind.split(":", 1)[1].split("|")
        upper = text.upper()
        if upper in allowed:
            res.value = upper
        else:
            res.null_reason = "illegible"
            res.review_required = True
            res.error = "NOT_IN_ALLOWED_VALUES"
        return res

    if name == "applicant_name":
        upper = text.upper()
        if not re.fullmatch(r"[A-Z .'\-]{3,}", upper):
            res.null_reason = "illegible"
            res.review_required = True
            res.error = "NAME_FORMAT"
        else:
            res.value = upper
        return res

    if name == "pan":
        compact = re.sub(r"\s", "", text).upper()
        norm, err = validate_pan(compact)
        if err is not None or norm is None:
            res.null_reason = "illegible"
            res.review_required = True
            res.validation = "failed"
            res.error = err or "PAN_FORMAT"
        else:
            res.value = norm
            res.validation = "passed"
        return res

    if name == "date_of_birth":
        norm, err = normalize_date_to_iso(text)
        if err is not None or norm is None:
            res.null_reason = "illegible"
            res.review_required = True
            res.validation = "failed"
            res.error = err or "DATE_FORMAT"
        else:
            res.value = norm
            res.validation = "passed"
        return res

    if name == "mobile":
        digits = re.sub(r"\D", "", text)
        norm, err = normalize_phone_number(digits)
        if err is not None or norm is None:
            res.null_reason = "illegible"
            res.review_required = True
            res.validation = "failed"
            res.error = err or "PHONE_FORMAT"
        else:
            res.value = norm
            res.validation = "passed"
        return res

    if name == "initial_deposit_amount":
        cleaned = re.sub(r"[^0-9.]", "", text)
        try:
            res.value = round(float(cleaned), 2)
            res.validation = "passed"
        except ValueError:
            res.null_reason = "illegible"
            res.review_required = True
            res.validation = "failed"
            res.error = "AMOUNT_FORMAT"
        return res

    res.value = text
    return res


def normalise_casa_fields(model_output: Any) -> CasaFormResult:
    """
    Takes the model's JSON (as a dict, or the raw text it returned) and produces the
    DOC-01 result. Missing or invalid values stay null and are routed to review.
    """
    if isinstance(model_output, str):
        parsed = parse_model_json(model_output)
        if parsed is None:
            result = CasaFormResult()
            result.flags.append({"code": "MODEL_OUTPUT_UNPARSEABLE", "fields": list(FIELD_SPEC)})
            for name in FIELD_SPEC:
                result.fields[name] = FieldResult(null_reason="illegible", review_required=True)
            return result
        model_output = parsed

    result = CasaFormResult()
    for name in FIELD_SPEC:
        result.fields[name] = _normalise_field(name, model_output.get(name))

    # Rule R1 (schema): applicant must be an adult. Failure routes to review, not to rejection.
    dob = result.fields["date_of_birth"].value
    if dob:
        code = _adult_check(dob)
        if code:
            result.fields["date_of_birth"].review_required = True
            result.fields["date_of_birth"].error = code
            result.flags.append({"code": code, "fields": ["date_of_birth"]})

    flagged = [n for n, f in result.fields.items() if f.review_required]
    if flagged:
        result.flags.append({"code": "FIELDS_REQUIRE_REVIEW", "fields": flagged})
        result.status = "needs_review"
    else:
        result.status = "auto_approved"
    return result


# ---------------------------------------------------------------------------
# Evidence grounding (PRD FR-EXT-004, GR-002): every value must be found in the page's text.
# ---------------------------------------------------------------------------

# Booleans come from checkbox marks, not text. Their evidence needs a checkbox detector,
# which is not part of this module, so they are routed to review until it is.
_CHECKBOX_FIELDS = {"opt_in_cheque_book", "opt_in_debit_card", "opt_in_internet_banking"}


def _alnum_upper(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())



def _printed_forms(name: str, value: Any) -> list:
    """Ways the value can appear on the printed form, as alphanumeric strings."""
    if name == "date_of_birth" and isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        y, m, d = value.split("-")
        return [_alnum_upper(f"{d}{m}{y}"), _alnum_upper(f"{d}-{m}-{y}")]
    if name == "mobile" and isinstance(value, str):
        digits = re.sub(r"\D", "", value)
        return [digits[-10:], digits]
    if isinstance(value, float):
        return [_alnum_upper(f"{value:,.2f}"), _alnum_upper(f"{value:.2f}"), _alnum_upper(f"{value:,.0f}")]
    return [_alnum_upper(str(value))]


def _find_printed(haystack: str, name: str, value: Any):
    """Returns (position, matched_needle) for the first printed form found, or (-1, '')."""
    for needle in _printed_forms(name, value):
        if needle:
            pos = haystack.find(needle)
            if pos >= 0:
                return pos, needle
    return -1, ""

def ground_fields(result: CasaFormResult, tokens: list) -> CasaFormResult:
    """
    tokens: list of dicts {"text": str, "bbox": [x0, y0, x1, y1], "page": int}, from OCR.
    A non-null value keeps its value only if its characters appear, in order, in the page text.
    Otherwise the value is removed, marked EVIDENCE_MISSING, and sent to review.
    """
    # Concatenate normalised token text and remember which token each character came from.
    chars: list = []
    owners: list = []
    for idx, tok in enumerate(tokens):
        norm = _alnum_upper(tok.get("text", ""))
        chars.extend(norm)
        owners.extend([idx] * len(norm))
    haystack = "".join(chars)

    missing = []
    for name, f in result.fields.items():
        if name in _CHECKBOX_FIELDS:
            if f.value is not None:
                f.review_required = True
                f.error = f.error or "CHECKBOX_EVIDENCE_NOT_AVAILABLE"
            continue
        if f.value is None:
            continue
        pos, needle = _find_printed(haystack, name, f.value)
        if pos < 0:
            f.value = None
            f.null_reason = "absent"
            f.review_required = True
            f.error = "EVIDENCE_MISSING"
            missing.append(name)
            continue
        hit = sorted(set(owners[pos:pos + len(needle)]))
        boxes = [tokens[i]["bbox"] for i in hit if tokens[i].get("bbox")]
        if boxes:
            x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
            x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
            f.evidence = {
                "page": tokens[hit[0]].get("page", 1),
                "bbox": [x0, y0, x1, y1],
                "text": " ".join(tokens[i]["text"] for i in hit),
            }

    if missing:
        result.flags.append({"code": "EVIDENCE_MISSING", "fields": missing})
    if any(f.review_required for f in result.fields.values()):
        result.status = "needs_review"
    return result


# ---------------------------------------------------------------------------
# Reading values from the model's plain-text page output (PaddleOCR-VL "OCR:" task).
# Each value is taken from the text next to its printed label. Checkboxes and account
# type are not in the text, so they stay null and go to review.
# ---------------------------------------------------------------------------

_HONORIFICS = {"MR", "MRS", "MS", "MISS", "DR", "COR"}
_LABEL_WORDS = {"APPLICANT", "NAME", "MR", "MR.", "COR", "COR.", "PA", "VALIDATION", "FIELD", "APPLICANT NAME"}


def _lines(text: str) -> list:
    return [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]


def _label_tail(line: str, label_re: str) -> str:
    """Text after the label keyword on the same line (the value is often printed there)."""
    m = re.search(label_re, line, re.IGNORECASE)
    return line[m.end():].lstrip(" :") if m else ""


def _after(lines: list, label_re: str, span: int = 1) -> list:
    """The label's own line after the keyword, then the following lines."""
    for i, ln in enumerate(lines):
        if re.search(label_re, ln, re.IGNORECASE):
            tail = _label_tail(ln, label_re)
            return ([tail] if tail else []) + lines[i + 1:i + 1 + span]
    return []


def _name_from_lines(name_lines: list) -> Optional[str]:
    """Joins the printed name from its lines: checkbox glyphs, honorifics and label words are dropped;
    spaced single letters ("P R I Y A") are joined into one word."""
    words: list = []
    for ln in name_lines:
        cleaned = re.sub(r"\s+", " ", re.sub(r"[^A-Za-z .]", " ", ln)).strip()
        if not cleaned or re.search(r"VALIDATION|FIELD", cleaned, re.IGNORECASE):
            continue
        tokens = [t.replace(".", "") for t in cleaned.split(" ") if t.replace(".", "")]
        tokens = [t for t in tokens if t.upper() not in _HONORIFICS and t.upper() not in _LABEL_WORDS]
        if len(tokens) == 1 and tokens[0].islower():  # a stray checkbox letter such as "v"
            continue
        run: list = []
        for t in tokens:
            if len(t) == 1 and t.isalpha():
                run.append(t.upper())
                continue
            if run:
                words.append("".join(run))
                run = []
            words.append(t.upper())
        if run:
            words.append("".join(run))
    name = " ".join(words)
    return name or None


def fields_from_ocr_text(text: str) -> Dict[str, Any]:
    """Maps the model's plain-text page output to DOC-01 field values (or None)."""
    lines = _lines(text)
    joined = " ".join(lines)
    out: Dict[str, Any] = {k: None for k in FIELD_SPEC}

    # Name: the line(s) between "Applicant" and the date-of-birth label, letters only.
    if "DATE OF BIRTH" in joined.upper():
        idx_dob = next(i for i, ln in enumerate(lines) if "DATE OF BIRTH" in ln.upper())
        start = next((i for i, ln in enumerate(lines[:idx_dob]) if "APPLICANT" in ln.upper() or re.search(r"\bNAME\b", ln, re.IGNORECASE)), None)
        if start is not None:
            out["applicant_name"] = _name_from_lines(
                [_label_tail(lines[start], r"NAME|APPLICANT")] + lines[start + 1:idx_dob]
            )

    # Date of birth: a dd-mm-yyyy shape after the label, spaces allowed inside.
    dob_text = re.sub(r"\s", "", " ".join(_after(lines, r"DATE OF BIRTH", span=1)))
    m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", dob_text)
    if m:
        out["date_of_birth"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # PAN: letters and digits, spaces removed, over the label and the next lines.
    pan_text = "".join(re.sub(r"[^A-Za-z0-9]", "", s) for s in _after(lines, r"\bPAN\b", span=3)).upper()
    m = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", pan_text)
    if m:
        out["pan"] = m.group(0)

    # Mobile: 10 digits after the first mobile label.
    mob_text = "".join(re.sub(r"[^0-9]", "", s) for s in _after(lines, r"MOBILE NO", span=1))
    m = re.search(r"\d{10}", mob_text)
    if m:
        out["mobile"] = m.group(0)

    # Account type and opt-ins: a tick glyph before the label means ticked; an empty box glyph means not ticked.
    tick = r"(?:✓|☑|✔|√)"
    empty = r"(?:☐|□)"
    if re.search(tick + r"\s*SAVINGS", joined, re.IGNORECASE):
        out["account_type"] = "SAVINGS"
    elif re.search(tick + r"\s*CURRENT", joined, re.IGNORECASE):
        out["account_type"] = "CURRENT"
    for field, label in (("opt_in_cheque_book", "Cheque\s*book"), ("opt_in_debit_card", "Debit\s*card"), ("opt_in_internet_banking", "Internet\s*banking")):
        if re.search(tick + r"\s*" + label, joined, re.IGNORECASE):
            out[field] = True
        elif re.search(empty + r"\s*" + label, joined, re.IGNORECASE):
            out[field] = False

    # Initial deposit: number after "Amount:".
    dep = " ".join(_after(lines, r"^Amount\s*:?$|Amount:", span=1))
    m = re.search(r"\d[\d,]*\.?\d*", dep)
    if m:
        out["initial_deposit_amount"] = m.group(0)

    return out
