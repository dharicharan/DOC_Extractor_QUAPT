"""
Multimodal (Qwen-VL) output handling for Module C (PRD v1.5: ARC-002, GR-002, GR-003, SCH-013).

Two deterministic stages, both reusing existing canonical components:

1. ``build_model_from_vlm``: raw model JSON (``{"applicant.full_name": {"value", "bbox"}}``)
   -> typed schema instance + per-field ``FieldGrounding``. Every value passes through
   ``validation_rules.normalize_and_format`` before it is typed into the schema.
2. ``grounding_to_evidence_hints``: normalized 0-1000 model boxes -> 200-DPI original page
   pixels via ``CoordinateTransform.map_bbox_preprocessed_to_original`` (inverting the
   deskew/rotation applied by Module A). The hints feed ``field_assembly.assemble_field_entry``,
   which keeps owning validation status, confidence and review gating.

Canonical box convention (PRD v1.5 FR-EXT-004 / GR-002 / FR-EXT-011): model boxes are
``[x1, y1, x2, y2]`` = (left, top, right, bottom) on a 0-1000 grid relative to the image the model
saw; evidence boxes are ``(x0, y0, x1, y1)`` pixels of the 200-DPI ORIGINAL page (xyxy, top-left
origin, X first). Nothing in this module reorders axes: x is always scaled by the width and y by the
height.

No confidence values are produced here: the model returns none, and none is invented.
"""

import json
import types
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError, create_model
from pydantic.fields import FieldInfo

from doc_extractor.contracts_c import FieldGrounding, FlagItem, PageImageRef
from doc_extractor.handoff import CoordinateTransform
from doc_extractor.validation_rules import normalize_and_format, normalize_currency_amount

_TRUE_TOKENS = {"true", "yes", "y", "1", "checked", "[x]", "x", "on", "✓", "✔"}
_FALSE_TOKENS = {"false", "no", "n", "0", "unchecked", "[ ]", "off", ""}


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union or origin is getattr(types, "UnionType", None):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def flatten_vlm_output(raw: Any) -> Dict[str, Dict[str, Any]]:
    """
    Flattens model JSON to ``{path: {"value", "bbox", "uncertain"}}``.
    A dict containing a ``value`` key is a leaf; other dicts/lists are walked
    (lists contribute numeric path segments).
    """
    flat: Dict[str, Dict[str, Any]] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            if "value" in node:
                flat[prefix] = {
                    "value": node.get("value"),
                    "bbox": node.get("bbox"),
                    "uncertain": bool(node.get("uncertain", False)),
                    "confidence": node.get("confidence"),
                }
                return
            for key, child in node.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(node, list):
            for idx, child in enumerate(node):
                walk(f"{prefix}.{idx}" if prefix else str(idx), child)
        elif prefix:
            flat[prefix] = {"value": node, "bbox": None, "uncertain": False}

    walk("", raw)
    return flat


def _resolve_schema_path(schema_cls: Type[BaseModel], path: str) -> Optional[Tuple[str, Any]]:
    """
    Maps a model-emitted dotted path onto the schema, returning ``(canonical_path, leaf_type)``.
    The extraction prompt lists repeating groups without an index (``nominees.name``); these map
    to element 0 (``nominees.0.name``), matching Module C's envelope field names.
    Returns None for paths that are not part of the schema.
    """
    segments = path.split(".")
    current: Type[BaseModel] = schema_cls
    canonical: List[str] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        info = current.model_fields.get(seg)
        if info is None:
            return None
        annotation = _unwrap_optional(info.annotation)
        canonical.append(seg)
        i += 1

        if get_origin(annotation) in (list, List):
            element = _unwrap_optional((get_args(annotation) or (Any,))[0])
            if _is_model(element):
                if i < len(segments) and segments[i].isdigit():
                    canonical.append(segments[i])
                    i += 1
                else:
                    canonical.append("0")
                current = element
                if i >= len(segments):
                    return None  # a bare group has no scalar leaf
                continue
            return ".".join(canonical), annotation  # list of scalars: leaf

        if _is_model(annotation):
            current = annotation
            if i >= len(segments):
                return None
            continue

        if i != len(segments):
            return None  # scalar field cannot have children
        return ".".join(canonical), annotation

    return None


def _coerce_leaf(leaf_name: str, value: Any, leaf_type: Any) -> Any:
    """normalize_and_format first (PRD GR-002/GR-003), then type coercion for the schema leaf."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None

    normalized, error = normalize_and_format(leaf_name, value)
    # On a normalizer error keep the raw reading: Module C's field stage re-validates it and
    # records the validation failure instead of silently dropping or "fixing" the value.
    candidate = value if error is not None else normalized
    if candidate is None:
        return None

    if leaf_type is bool:
        if isinstance(candidate, bool):
            return candidate
        token = str(candidate).strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        return candidate  # let schema validation reject it loudly

    if leaf_type in (int, float) and isinstance(candidate, str):
        text = candidate.strip().rstrip("%").strip()
        try:
            number = float(text.replace(",", ""))
        except ValueError:
            amount, _currency, amount_err = normalize_currency_amount(text)
            return amount if amount_err is None and amount is not None else candidate
        return int(number) if leaf_type is int and number.is_integer() else number

    if leaf_type is str and not isinstance(candidate, str):
        return str(candidate)

    return candidate


def _valid_bbox(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    """
    Accepts ``[x1, y1, x2, y2]`` on 0-1000 (left, top, right, bottom); anything else is ungrounded.
    Swapped corners (x2 < x1 or y2 < y1) describe the same rectangle and are put in order; a box with
    no area, a non-numeric entry or a value off the grid (e.g. raw pixels) is treated as ungrounded.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if min(x1, y1, x2, y2) < 0.0 or max(x1, y1, x2, y2) > 1000.0:
        return None  # not on the normalized grid (e.g. raw pixels)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def unrotate_bbox_norm(
    bbox_norm: Tuple[float, float, float, float], clockwise_deg: int
) -> Tuple[float, float, float, float]:
    """
    Maps a ``[x1, y1, x2, y2]`` box (0-1000) read on an image that was turned UPRIGHT by rotating the
    stored page clockwise by ``clockwise_deg`` (0/90/180/270) back to the stored page's frame, so the
    Module A geometry (``CoordinateTransform``) can be applied to it. Corners are re-ordered afterwards.
    """
    x1, y1, x2, y2 = bbox_norm
    deg = int(clockwise_deg) % 360
    if deg == 0:
        return bbox_norm
    if deg == 90:
        pts = [(y1, 1000.0 - x1), (y2, 1000.0 - x2)]
    elif deg == 180:
        pts = [(1000.0 - x1, 1000.0 - y1), (1000.0 - x2, 1000.0 - y2)]
    elif deg == 270:
        pts = [(1000.0 - y1, x1), (1000.0 - y2, x2)]
    else:
        raise ValueError(f"rotation must be 0, 90, 180 or 270 degrees clockwise, got {clockwise_deg}")
    (ax, ay), (bx, by) = pts
    return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)


def _assign(tree: Dict[str, Any], canonical_path: str, value: Any) -> None:
    segments = canonical_path.split(".")
    node: Any = tree
    for idx, seg in enumerate(segments):
        last = idx == len(segments) - 1
        nxt = None if last else segments[idx + 1]
        key: Union[str, int] = int(seg) if seg.isdigit() else seg

        if isinstance(node, list):
            while len(node) <= key:  # type: ignore[operator]
                node.append({})
            if last:
                node[key] = value  # type: ignore[index]
                return
            node = node[key]  # type: ignore[index]
            continue

        if last:
            node[key] = value
            return
        if key not in node:
            node[key] = [] if (nxt is not None and nxt.isdigit()) else {}
        node = node[key]


@dataclass
class PageLeaf:
    """One normalized, typed value read from one page, with its (page-local) model box."""
    value: Any
    bbox_norm: Optional[Tuple[float, float, float, float]]
    raw_value: Any
    uncertain: bool
    confidence: Optional[float] = None   # measured token probability of the value text (FR-CNF-001)


_NULL_WORDS = {"null", "none", "nil", "n/a", "na", "not available", "not applicable", "not visible", "empty", "-", "--"}


def _null_word_to_none(value: Any) -> Any:
    """A model that writes the word "null" (or "N/A", "none", "-") means no value (FR-EXT-003)."""
    if isinstance(value, str) and value.strip().strip(".").lower() in _NULL_WORDS:
        return None
    return value


def _valid_probability(x: Any) -> Optional[float]:
    """A model probability in [0, 1]; anything else (missing, bool, NaN, out of range) is no signal."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or x != x or not 0.0 <= float(x) <= 1.0:
        return None
    return float(x)


def collect_schema_leaves(raw: Dict[str, Any], schema_cls: Type[BaseModel]) -> Dict[str, PageLeaf]:
    """
    One page of raw model JSON -> ``{canonical_path: PageLeaf}`` against ``schema_cls``.
    Repeating-group paths keep the index the model emitted on this page (``nominees.1.name``);
    un-indexed group paths map to element 0. Null/blank values and fields outside the schema
    are dropped. Every value passes through ``normalize_and_format`` before typing.
    """
    leaves: Dict[str, PageLeaf] = {}
    for raw_path, leaf in flatten_vlm_output(raw).items():
        resolved = _resolve_schema_path(schema_cls, raw_path)
        if resolved is None:
            continue  # ignore fields outside the bound schema
        canonical_path, leaf_type = resolved
        value = _coerce_leaf(
            canonical_path.split(".")[-1], _null_word_to_none(leaf.get("value")), _unwrap_optional(leaf_type)
        )
        if value is None:
            continue
        leaves[canonical_path] = PageLeaf(
            value=value,
            bbox_norm=_valid_bbox(leaf.get("bbox")),
            raw_value=leaf.get("value"),
            uncertain=bool(leaf.get("uncertain", False)),
            confidence=_valid_probability(leaf.get("confidence")),
        )
    return leaves


def collect_generic_leaves(raw: Dict[str, Any]) -> Dict[str, PageLeaf]:
    """
    Schema-less variant for unstructured/handwritten categories. String values pass through
    ``normalize_and_format`` (raw is kept on a normalizer error); lists/dicts/numbers are kept.
    """
    leaves: Dict[str, PageLeaf] = {}
    for path, leaf in flatten_vlm_output(raw).items():
        value = leaf.get("value")
        value = _null_word_to_none(value)
        if isinstance(value, str):
            if not value.strip():
                continue
            normalized, error = normalize_and_format(path.split(".")[-1], value)
            value = value if error is not None else normalized
        if value is None:
            continue
        leaves[path] = PageLeaf(
            value=value,
            bbox_norm=_valid_bbox(leaf.get("bbox")),
            raw_value=leaf.get("value"),
            uncertain=bool(leaf.get("uncertain", False)),
            confidence=_valid_probability(leaf.get("confidence")),
        )
    return leaves


def _relax_annotation(annotation: Any) -> Any:
    """Rebuilds a type so every nested model it mentions is its requiredness-relaxed twin."""
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return relaxed_schema(annotation)
        return annotation
    args = get_args(annotation)
    if origin in (list, List):
        return List[_relax_annotation(args[0])]  # type: ignore[misc]
    if origin is Union or origin is getattr(types, "UnionType", None):
        return Union[tuple(_relax_annotation(a) for a in args)]  # type: ignore[return-value]
    return annotation


@lru_cache(maxsize=256)
def relaxed_schema(schema_cls: Type[BaseModel]) -> Type[BaseModel]:
    """
    A subclass of ``schema_cls`` in which required fields (at every nesting level, including the
    items of repeating groups) default to None instead of being mandatory. Types, formats,
    patterns, validators and aliases are inherited unchanged, so a value that IS present is still
    fully validated; only absence is tolerated. Used to keep what was extracted when a required
    field is absent from every page, so completeness can be reviewed instead of discarding the
    document (the missing fields are reported separately).
    """
    overrides: Dict[str, Any] = {}
    for name, info in schema_cls.model_fields.items():
        annotation = _relax_annotation(info.annotation)
        if info.is_required():
            if annotation is not None and type(None) not in get_args(annotation):
                annotation = Optional[annotation]
            overrides[name] = (annotation, FieldInfo.merge_field_infos(info, default=None))
        elif annotation != info.annotation:
            overrides[name] = (annotation, info)
    if not overrides:
        return schema_cls
    return create_model(f"{schema_cls.__name__}Partial", __base__=schema_cls, **overrides)


def _error_path(error: Dict[str, Any]) -> str:
    return ".".join(str(p) for p in error["loc"])


def validate_allowing_missing(
    schema_cls: Type[BaseModel],
    tree: Dict[str, Any],
) -> Tuple[BaseModel, List[str]]:
    """
    Validates ``tree`` against ``schema_cls`` and separates the two kinds of failure:

    * a required field that is simply absent -> NOT an error: the extraction is kept, built on
      ``relaxed_schema`` with the absent fields as None, and their dotted paths are returned so
      the caller can record them (validation failed + needs_review + REQUIRED_FIELD_MISSING);
    * a value that is present but malformed (wrong type/format/enum) -> ``ValueError``
      (genuinely invalid output, fail closed as before).

    Returns ``(model, missing_required_paths)``; the list is empty when the schema is satisfied.
    """
    try:
        return schema_cls.model_validate(tree), []
    except ValidationError as exc:
        errors = exc.errors()
        malformed = [e for e in errors if e["type"] != "missing"]
        if malformed:
            raise ValueError(_problems(schema_cls, malformed)) from exc
        missing = list(dict.fromkeys(_error_path(e) for e in errors))
    try:
        return relaxed_schema(schema_cls).model_validate(tree), missing
    except ValidationError as exc:
        raise ValueError(_problems(schema_cls, exc.errors())) from exc


def _problems(schema_cls: Type[BaseModel], errors: List[Dict[str, Any]]) -> str:
    detail = "; ".join(f"{_error_path(e)}: {e['msg']}" for e in errors[:5])
    return f"VLM output does not satisfy schema {schema_cls.__name__}: {detail}"


def _tree_from_values(values: Dict[str, Any]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for path, value in values.items():
        _assign(tree, path, value)
    return tree


def validate_page_leaves(schema_cls: Type[BaseModel], leaves: Dict[str, PageLeaf]) -> None:
    """
    Trial-validates ONE page's values so a malformed page can be failed without losing the others.
    Fields that are simply absent from this page are not errors (they may be on another page).
    """
    validate_allowing_missing(schema_cls, _tree_from_values({p: leaf.value for p, leaf in leaves.items()}))


def _split_group_path(path: str) -> Optional[Tuple[str, int, str]]:
    """``a.0.b.1.c`` -> (``a``, 0, ``b.1.c``): split at the first list index; None for scalars."""
    segments = path.split(".")
    for i, seg in enumerate(segments):
        if seg.isdigit() and i > 0 and i < len(segments) - 1:
            return ".".join(segments[:i]), int(seg), ".".join(segments[i + 1:])
    return None


def _value_signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _grounding_from(occurrences: List[Tuple[int, PageLeaf]]) -> FieldGrounding:
    """
    First occurrence (document order) is the primary reading. Later occurrences with an equal
    value are kept as ``also_seen`` (deduplicated, evidence preserved); different values are kept
    as ``conflicts`` and are never allowed to overwrite the first reading.
    """
    first_page, first = occurrences[0]
    primary = FieldGrounding(
        page_number=first_page, bbox_norm=first.bbox_norm,
        raw_value=first.raw_value, uncertain=first.uncertain, model_confidence=first.confidence,
    )
    for page, leaf in occurrences[1:]:
        other = FieldGrounding(
            page_number=page, bbox_norm=leaf.bbox_norm,
            raw_value=leaf.raw_value, uncertain=leaf.uncertain, model_confidence=leaf.confidence,
        )
        (primary.also_seen if leaf.value == first.value else primary.conflicts).append(other)
    return primary


def aggregate_leaves(
    pages: Sequence[Tuple[int, Dict[str, PageLeaf]]],
) -> Tuple[Dict[str, Any], Dict[str, FieldGrounding]]:
    """
    Schema-aware aggregation of per-page leaves, in document page order.

    - Scalar fields: identical values on several pages collapse to one value whose page
      evidence is preserved (``also_seen``); a DIFFERENT value on a later page never replaces the
      first reading - it is recorded as a conflict (``conflicts``) for review.
    - Repeating groups (paths with a list index): each page's elements are kept as distinct
      entries, re-indexed contiguously in page order (page 1's entries, then page 2's, ...).
      Elements whose sub-fields are all identical are the same entry seen again and are
      deduplicated with their page evidence preserved. Entries are never partially merged: no
      key fields are assumed, so anything that is not identical stays a separate entry.
    """
    scalar_occurrences: Dict[str, List[Tuple[int, PageLeaf]]] = {}
    # group prefix -> ordered list of entries; entry = (signature, {sub_path: [(page, leaf)]})
    group_entries: Dict[str, List[Tuple[str, Dict[str, List[Tuple[int, PageLeaf]]]]]] = {}

    for page_number, leaves in pages:
        page_groups: Dict[str, Dict[int, Dict[str, PageLeaf]]] = {}
        for path, leaf in leaves.items():
            split = _split_group_path(path)
            if split is None:
                scalar_occurrences.setdefault(path, []).append((page_number, leaf))
            else:
                prefix, index, sub = split
                page_groups.setdefault(prefix, {}).setdefault(index, {})[sub] = leaf

        for prefix, elements in page_groups.items():
            entries = group_entries.setdefault(prefix, [])
            for index in sorted(elements):
                element = elements[index]
                signature = _value_signature({sub: leaf.value for sub, leaf in sorted(element.items())})
                existing = next((e for e in entries if e[0] == signature), None)
                if existing is not None:
                    for sub, leaf in element.items():
                        existing[1].setdefault(sub, []).append((page_number, leaf))
                else:
                    entries.append((signature, {sub: [(page_number, leaf)] for sub, leaf in element.items()}))

    values: Dict[str, Any] = {}
    grounding: Dict[str, FieldGrounding] = {}

    for path, occurrences in scalar_occurrences.items():
        values[path] = occurrences[0][1].value
        grounding[path] = _grounding_from(occurrences)

    for prefix, entries in group_entries.items():
        for entry_index, (_signature, subs) in enumerate(entries):
            for sub, occurrences in subs.items():
                path = f"{prefix}.{entry_index}.{sub}"
                values[path] = occurrences[0][1].value
                grounding[path] = _grounding_from(occurrences)

    return values, grounding


def build_partial_model_from_pages(
    pages: Sequence[Tuple[int, Dict[str, PageLeaf]]],
    schema_cls: Type[BaseModel],
) -> Tuple[BaseModel, Dict[str, FieldGrounding], List[str]]:
    """
    Aggregates per-page leaves and validates the combined values against ``schema_cls``.
    Returns ``(model, grounding, missing_required_paths)``: required fields absent from every
    page do not discard the extraction (see ``validate_allowing_missing``); malformed values
    still raise ``ValueError``.
    """
    values, grounding = aggregate_leaves(pages)
    model, missing = validate_allowing_missing(schema_cls, _tree_from_values(values))
    return model, grounding, missing


def build_model_from_pages(
    pages: Sequence[Tuple[int, Dict[str, PageLeaf]]],
    schema_cls: Type[BaseModel],
) -> Tuple[BaseModel, Dict[str, FieldGrounding]]:
    """``build_partial_model_from_pages`` without the missing-field list."""
    model, grounding, _missing = build_partial_model_from_pages(pages, schema_cls)
    return model, grounding


def build_model_from_vlm(
    raw: Dict[str, Any],
    schema_cls: Type[BaseModel],
    page_number: int = 1,
) -> Tuple[BaseModel, Dict[str, FieldGrounding]]:
    """
    Single-page entry point: typed ``schema_cls`` instance plus per-field grounding.

    Raises:
        ValueError: if the coerced values do not satisfy the schema (fail closed; the caller
            applies the extractor's strict/fallback policy).
    """
    return build_model_from_pages([(page_number, collect_schema_leaves(raw, schema_cls))], schema_cls)


def build_generic_fields(
    raw: Dict[str, Any],
    page_number: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, FieldGrounding]]:
    """Single-page schema-less entry point: ``({path: value}, {path: FieldGrounding})``."""
    return aggregate_leaves([(page_number, collect_generic_leaves(raw))])


def conflict_flags(grounding: Dict[str, FieldGrounding]) -> List[FlagItem]:
    """One FIELD_VALUE_CONFLICT flag per field whose value differs between pages (forces review)."""
    flags: List[FlagItem] = []
    for path, item in grounding.items():
        if not item.conflicts:
            continue
        readings = "; ".join(f"page {g.page_number}: {g.raw_value!r}" for g in [item] + item.conflicts)
        flags.append(FlagItem(
            code="FIELD_VALUE_CONFLICT",
            message=(
                f"Field '{path}' has conflicting values across pages ({readings}). "
                "The first page's reading is retained pending review."
            ),
            fields=[path],
        ))
    return flags


def _hint_for(item: FieldGrounding, by_page: Dict[int, PageImageRef]) -> Optional[Dict[str, Any]]:
    """Page-specific hint: the box is mapped with THIS page's own Module A geometry."""
    ref = by_page.get(item.page_number)
    if ref is None or item.bbox_norm is None:
        return None
    page = ref.page_ref
    x1, y1, x2, y2 = item.bbox_norm                       # canonical [x1, y1, x2, y2] on 0-1000
    width, height = page.preprocessed_width_px, page.preprocessed_height_px
    bbox_prep = (                                         # preprocessed-image pixels (x0, y0, x1, y1)
        x1 * width / 1000.0,
        y1 * height / 1000.0,
        x2 * width / 1000.0,
        y2 * height / 1000.0,
    )
    x0, y0, x1, y1 = CoordinateTransform.map_bbox_preprocessed_to_original(
        bbox=bbox_prep,
        original_width=page.original_width_px,
        original_height=page.original_height_px,
        applied_rotation_deg=page.applied_rotation_deg,
        applied_deskew_deg=page.applied_deskew_deg,
    )
    return {
        "page": item.page_number,
        "bbox": (x0, y0, x1, y1),
        "text": str(item.raw_value)[:120] if item.raw_value is not None else "",
        "ocr_confidence": None,  # no optical OCR confidence exists for model-read text
        "uncertain": item.uncertain,
        "model_confidence": getattr(item, "model_confidence", None),
    }


def grounding_to_evidence_hints(
    grounding: Dict[str, FieldGrounding],
    page_images: Sequence[PageImageRef],
) -> Dict[str, Dict[str, Any]]:
    """
    Converts normalized model boxes into evidence hints in original 200-DPI pixel space.

    Every occurrence is mapped with its own page's geometry. The hint's primary evidence is the
    first grounded occurrence of the retained value; the remaining occurrences (same value on
    other pages, and any conflicting readings) are listed under ``additional``. A field with
    conflicting readings is marked ``uncertain`` so it requires review. Fields without any valid
    box get no hint (they fall back to OCR grounding or EVIDENCE_MISSING).
    """
    by_page = {ref.page_number: ref for ref in page_images}
    hints: Dict[str, Dict[str, Any]] = {}

    for path, item in grounding.items():
        same_value = [item] + list(item.also_seen)
        grounded = [(g, _hint_for(g, by_page)) for g in same_value]
        grounded = [(g, h) for g, h in grounded if h is not None]
        if not grounded:
            continue
        primary = dict(grounded[0][1])
        additional = [h for _, h in grounded[1:]]
        additional += [h for h in (_hint_for(g, by_page) for g in item.conflicts) if h is not None]
        primary["additional"] = additional
        primary["uncertain"] = any(g.uncertain for g in same_value) or bool(item.conflicts)
        hints[path] = primary
    return hints
