"""
Dynamic Rule Evaluator for Schema Metamodel Rules (PRD §5.5, §6.2, FR-VAL-003, FR-VAL-004, FR-VAL-006).
Evaluates declared chronology, arithmetic, regex, and business rules dynamically
using existing deterministic validation functions without inventing expressions.
"""

from decimal import Decimal
import re
from typing import Any, Dict, List, Optional

from doc_extractor.contracts_c import ExtractedField, FlagItem
from doc_extractor.validation_rules import (
    normalize_currency_amount,
    normalize_date_to_iso,
    validate_arithmetic,
    validate_chronology,
)
from schema_lifecycle.enums import RuleOnFail
from schema_lifecycle.metamodel import RuleDefinition




def evaluate_dynamic_rules(
    rules: List[RuleDefinition],
    flat_dict: Dict[str, Any],
    fields_envelope: Dict[str, ExtractedField],
    flags: List[FlagItem],
) -> None:
    """
    Evaluates schema rules against extracted flat fields.
    Updates fields_envelope with review_required / validation status and
    appends FlagItem entries to flags.
    """
    if not rules:
        return

    for rule in rules:
        expr = rule.expression.strip()
        expr_low = expr.lower()
        err_code = rule.error_code or "RULE_VIOLATION"
        targets = list(rule.target_fields)

        # ---------------------------------------------------------------------
        # 1. Chronology Rules (FR-VAL-004)
        # ---------------------------------------------------------------------
        is_chronology = (
            rule.error_code == "CHRONOLOGY_MISMATCH"
            or "chronology" in rule.id.lower()
            or "chronology" in (rule.description or "").lower()
            or (
                len(targets) == 2
                and any(op in expr for op in ["<=", "<", ">=", ">"])
                and any("date" in t.lower() for t in targets)
            )
        )
        if is_chronology:
            _evaluate_chronology_rule(rule, expr, targets, flat_dict, fields_envelope, flags)
            continue

        # ---------------------------------------------------------------------
        # 2. Arithmetic Rules (FR-VAL-003)
        # ---------------------------------------------------------------------
        is_arithmetic = (
            rule.error_code == "ARITHMETIC_MISMATCH"
            or "arithmetic" in rule.id.lower()
            or "arithmetic" in (rule.description or "").lower()
            or "sum(" in expr_low
        )
        if is_arithmetic:
            _evaluate_arithmetic_rule(rule, expr, targets, flat_dict, fields_envelope, flags)
            continue

        # ---------------------------------------------------------------------
        # 3. Regex / Format Rules (FR-VAL-002)
        # ---------------------------------------------------------------------
        is_regex = (
            rule.error_code == "INVALID_FORMAT"
            or expr.startswith("regex:")
            or (expr.startswith("^") and expr.endswith("$"))
        )
        if is_regex:
            _evaluate_regex_rule(rule, expr, targets, flat_dict, fields_envelope, flags)
            continue

        # ---------------------------------------------------------------------
        # 4. Numeric / Field Comparison Rules (e.g. 'initial_deposit >= 1000')
        # ---------------------------------------------------------------------
        _evaluate_comparison_rule(rule, expr, targets, flat_dict, fields_envelope, flags)


def _evaluate_chronology_rule(
    rule: RuleDefinition,
    expr: str,
    targets: List[str],
    flat_dict: Dict[str, Any],
    fields_envelope: Dict[str, ExtractedField],
    flags: List[FlagItem],
) -> None:
    """Evaluates chronology rule: start_date <= end_date."""
    # Determine start and end field paths
    if len(targets) >= 2:
        field_start, field_end = targets[0], targets[1]
    else:
        # Fallback: extract identifiers from expression
        tokens = re.findall(r"[A-Za-z0-9_\.]+", expr)
        if len(tokens) >= 2:
            field_start, field_end = tokens[0], tokens[1]
        else:
            return

    # If expression is 'discharge >= admission', invert order
    if ">" in expr and not ("<" in expr):
        field_start, field_end = field_end, field_start

    raw_start = flat_dict.get(field_start)
    raw_end = flat_dict.get(field_end)

    if raw_start is None or raw_end is None:
        return

    iso_start, _ = normalize_date_to_iso(str(raw_start))
    iso_end, _ = normalize_date_to_iso(str(raw_end))

    chron_flag = validate_chronology(iso_start, iso_end)
    if chron_flag:
        flag_code = rule.error_code or "CHRONOLOGY_MISMATCH"
        msg = rule.description or chron_flag["message"]
        affected_fields = targets if targets else [field_start, field_end]
        flags.append(FlagItem(code=flag_code, message=msg, fields=affected_fields))

        for f in affected_fields:
            if f in fields_envelope:
                fields_envelope[f].review_required = True


def _evaluate_arithmetic_rule(
    rule: RuleDefinition,
    expr: str,
    targets: List[str],
    flat_dict: Dict[str, Any],
    fields_envelope: Dict[str, ExtractedField],
    flags: List[FlagItem],
) -> None:
    """Evaluates arithmetic rule: sum(items) == total within tolerance."""
    total_val: Optional[float] = None
    item_amounts: List[float] = []

    # Identify total field vs item fields
    total_field = None
    item_fields = []
    for t in targets:
        if "total" in t.lower() or "sum" in t.lower() or "declared" in t.lower():
            total_field = t
        else:
            item_fields.append(t)

    if not total_field and targets:
        total_field = targets[-1]
        item_fields = targets[:-1]

    if total_field:
        raw_total = flat_dict.get(total_field)
        if raw_total is not None:
            num, _, _ = normalize_currency_amount(raw_total)
            total_val = num

    # Collect item amounts, supporting repeating groups (e.g. 'items[].amount' or 'line_items')
    for item_spec in item_fields:
        if "[]" in item_spec:
            rx_pat = "^" + re.escape(item_spec).replace(r"\[\]", r"\.\d+") + "$"
            rx = re.compile(rx_pat)
            for k, v in flat_dict.items():
                if rx.match(k):
                    if isinstance(v, (int, float, Decimal)):
                        item_amounts.append(float(v))
                    elif isinstance(v, str):
                        num, _, _ = normalize_currency_amount(v)
                        if num is not None:
                            item_amounts.append(num)
        else:
            clean_spec = item_spec.rstrip(".")
            for k, v in flat_dict.items():
                if k == clean_spec or k.startswith(f"{clean_spec}."):
                    if isinstance(v, (int, float, Decimal)):
                        item_amounts.append(float(v))
                    elif isinstance(v, str):
                        num, _, _ = normalize_currency_amount(v)
                        if num is not None:
                            item_amounts.append(num)

    arith_flag = validate_arithmetic(line_item_amounts=item_amounts, total_amount=total_val, tolerance=0.01)
    if arith_flag:
        flag_code = rule.error_code or "ARITHMETIC_MISMATCH"
        msg = rule.description or arith_flag["message"]
        flags.append(FlagItem(code=flag_code, message=msg, fields=targets))

        for f in targets:
            clean_f = f.replace("[]", "")
            if f in fields_envelope:
                fields_envelope[f].review_required = True
            for env_k, env_v in fields_envelope.items():
                if env_k == f or env_k == clean_f or env_k.startswith(f"{clean_f}."):
                    env_v.review_required = True


def _evaluate_regex_rule(
    rule: RuleDefinition,
    expr: str,
    targets: List[str],
    flat_dict: Dict[str, Any],
    fields_envelope: Dict[str, ExtractedField],
    flags: List[FlagItem],
) -> None:
    """Evaluates regex pattern matching against target fields."""
    pattern = expr[6:].strip() if expr.startswith("regex:") else expr.strip()
    try:
        rx = re.compile(pattern)
    except re.error:
        return

    for t in targets:
        val = flat_dict.get(t)
        if val is not None and not rx.search(str(val)):
            flag_code = rule.error_code or "INVALID_FORMAT"
            msg = rule.description or f"Field '{t}' failed regex format validation: {pattern}"
            flags.append(FlagItem(code=flag_code, message=msg, fields=[t]))
            if t in fields_envelope:
                fields_envelope[t].validation = "failed"
                fields_envelope[t].review_required = True


def _evaluate_comparison_rule(
    rule: RuleDefinition,
    expr: str,
    targets: List[str],
    flat_dict: Dict[str, Any],
    fields_envelope: Dict[str, ExtractedField],
    flags: List[FlagItem],
) -> None:
    """Evaluates simple comparison expression (e.g. 'deposit >= 1000' or 'amount > 0')."""
    match = re.match(r"^\s*([A-Za-z0-9_\.]+)\s*(>=|<=|>|<|==|!=)\s*(.+?)\s*$", expr)
    if not match:
        return

    field_name, op, raw_operand = match.groups()
    target_field = targets[0] if targets else field_name

    val = flat_dict.get(target_field)
    if val is None:
        # Null check
        if raw_operand.strip().lower() in ("null", "none"):
            violated = (op == "!=")
            if violated:
                _flag_violation(rule, target_field, fields_envelope, flags, expr)
        return

    # Try numeric comparison
    try:
        val_num = float(val)
        operand_num = float(raw_operand)
        violated = False
        if op == ">=":
            violated = not (val_num >= operand_num)
        elif op == "<=":
            violated = not (val_num <= operand_num)
        elif op == ">":
            violated = not (val_num > operand_num)
        elif op == "<":
            violated = not (val_num < operand_num)
        elif op == "==":
            violated = not (abs(val_num - operand_num) < 1e-6)
        elif op == "!=":
            violated = not (abs(val_num - operand_num) >= 1e-6)

        if violated:
            _flag_violation(rule, target_field, fields_envelope, flags, expr)
    except (ValueError, TypeError):
        pass


def _flag_violation(
    rule: RuleDefinition,
    target_field: str,
    fields_envelope: Dict[str, ExtractedField],
    flags: List[FlagItem],
    expr: str,
) -> None:
    flag_code = rule.error_code or "RULE_VIOLATION"
    msg = rule.description or f"Rule '{rule.id}' failed: {expr}"
    affected = rule.target_fields if rule.target_fields else [target_field]
    flags.append(FlagItem(code=flag_code, message=msg, fields=affected))

    on_fail_action = str(rule.on_fail).lower()
    if on_fail_action in (RuleOnFail.REVIEW.value, "review", RuleOnFail.REJECT.value, "reject"):
        for f in affected:
            if f in fields_envelope:
                fields_envelope[f].review_required = True
