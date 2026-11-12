"""
Module B Contracts: Visual Layout Parsing and Document Classification.

Adheres strictly to Document_Extraction_Engine_PRD.md v1.1:
- FR-CLS-001: Domain and document type classification with confidence.
- FR-CLS-002: Fallback routing for UNKNOWN / low confidence (needs_review, no extraction).
- FR-CLS-003: Approved schema version binding.
- FR-PRS-001: Layout-faithful Markdown representation preserving reading order.
- FR-PRS-002: Checkboxes rendered as [x] or [ ] with bounding boxes.
- FR-PRS-003: Segmented character grid representation.
- FR-PRS-004: Typed layout elements (tables, key-value pairs, signatures, stamps).
- FR-PRS-005: Optical confidence reported per text block (ocr_confidence in [0, 1]).
- SCH-011: JSON Schema compatibility and semantic versioning (1.0.0).
- SCH-012: DocumentStatus lifecycle compatibility.
- SCH-013: Bounding box [x0, y0, x1, y1] in pixels of 200-DPI page image with origin top-left.
"""

from enum import Enum
import re
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator

from doc_extractor.handoff import CoordinateTransform, ModuleAHandoffPayload


MODULE_B_CONTRACT_VERSION = "1.0.0"


# ============================================================================
# Enums
# ============================================================================

class MVPClassificationDomain(str, Enum):
    """PRD FR-CLS-001 domains."""
    BFSI = "BFSI"
    HEALTHCARE = "Healthcare"
    UNKNOWN = "Unknown"


class MVPDocumentType(str, Enum):
    """
    PRD Section 1.2 launch seven MVP document types:
    - DOC-01: CASA Account Opening Form
    - DOC-02: Insurance Application Form
    - DOC-03: Hospital Discharge Summary
    - UNKNOWN: Fallback for below-threshold or unrecognised documents (FR-CLS-002).
    """
    DOC_01_CASA = "DOC-01"
    DOC_02_INSURANCE_APP = "DOC-02"
    DOC_03_DISCHARGE_SUMMARY = "DOC-03"
    UNKNOWN = "UNKNOWN"


class LayoutElementType(str, Enum):
    """PRD FR-PRS-004 typed layout elements."""
    HEADER = "header"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    KEY_VALUE_PAIR = "key_value_pair"
    CHECKBOX = "checkbox"
    CHARACTER_GRID = "character_grid"
    SIGNATURE = "signature"
    STAMP = "stamp"
    FOOTER = "footer"


# ============================================================================
# Coordinate & Bounding Box Models (SCH-013)
# ============================================================================

class BoundingBox(BaseModel):
    """
    PRD SCH-013: Bounding box [x0, y0, x1, y1] in pixels with origin top-left.
    Coordinates must be non-negative and satisfy x1 >= x0, y1 >= y0.
    """
    x0: float = Field(ge=0.0, description="Left coordinate in pixels")
    y0: float = Field(ge=0.0, description="Top coordinate in pixels")
    x1: float = Field(ge=0.0, description="Right coordinate in pixels")
    y1: float = Field(ge=0.0, description="Bottom coordinate in pixels")

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox":
        if self.x1 < self.x0:
            raise ValueError(f"Invalid bounding box: x1 ({self.x1}) must be >= x0 ({self.x0}).")
        if self.y1 < self.y0:
            raise ValueError(f"Invalid bounding box: y1 ({self.y1}) must be >= y0 ({self.y0}).")
        return self

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_tuple(cls, coords: Tuple[float, float, float, float]) -> "BoundingBox":
        return cls(x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])


# ============================================================================
# Module B Input Contract (Module A Consumer)
# ============================================================================

class ModuleAPageRef(BaseModel):
    """Validated reference to a single page from Module A handoff."""
    page_number: int = Field(ge=1, description="1-based sequence")
    original_artifact_id: str
    preprocessed_artifact_id: str
    original_storage_uri: str
    preprocessed_storage_uri: str
    original_width_px: int = Field(gt=0)
    original_height_px: int = Field(gt=0)
    preprocessed_width_px: int = Field(gt=0)
    preprocessed_height_px: int = Field(gt=0)
    dpi: int = Field(default=200)
    applied_rotation_deg: int = Field(default=0)
    applied_deskew_deg: float = Field(default=0.0)
    residual_skew_deg: Optional[float] = None
    contrast_method: str = "clahe"
    metadata_storage_uri: str


class ModuleBInputPayload(BaseModel):
    """
    Validated representation of ModuleAHandoffPayload consumed by Module B.
    Enforces compatibility, state validity, and page integrity.
    """
    contract_version: str = Field(description="Must match semver 1.x.x")
    run_id: str
    document_id: str
    tenant_id: str
    status: str
    internal_state: str
    claimed_filename: str
    file_hash_sha256: str
    file_size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=1)
    mime_type: str
    pages: List[ModuleAPageRef] = Field(default_factory=list)
    created_at: str
    completed_at: str
    error: Optional[Dict[str, Any]] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("contract_version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if not re.match(r"^1\.\d+\.\d+$", v):
            raise ValueError(f"Unsupported Module A contract version '{v}'. Supported: 1.x.x")
        return v

    @model_validator(mode="after")
    def validate_pipeline_state(self) -> "ModuleBInputPayload":
        if self.status != "processing":
            raise ValueError(
                f"Module A handoff document status must be 'processing', got '{self.status}'."
            )
        if self.internal_state != "ready_for_extraction":
            raise ValueError(
                f"Module A handoff internal state must be 'ready_for_extraction', got '{self.internal_state}'."
            )
        if self.error is not None:
            raise ValueError(f"Cannot process handoff containing errors: {self.error}")
        if len(self.pages) != self.page_count:
            raise ValueError(
                f"Page count mismatch: claimed {self.page_count}, but {len(self.pages)} page records present."
            )

        # Enforce sequential 1-based page ordering
        for idx, page in enumerate(self.pages):
            expected_num = idx + 1
            if page.page_number != expected_num:
                raise ValueError(
                    f"Invalid page sequence: expected page {expected_num} at index {idx}, got {page.page_number}."
                )
        return self

    @classmethod
    def from_module_a_payload(cls, payload: ModuleAHandoffPayload) -> "ModuleBInputPayload":
        """Factory adapting an existing ModuleAHandoffPayload instance."""
        data = payload.to_dict()
        return cls(**data)


# ============================================================================
# Layout Parsing Elements (FR-PRS-001 through FR-PRS-005)
# ============================================================================

class LayoutBlock(BaseModel):
    """
    PRD FR-PRS-001, FR-PRS-004, FR-PRS-005:
    Individual recognized text block with OCR optical confidence and mapped bounding box.
    """
    block_id: str
    element_type: LayoutElementType
    text: str
    ocr_confidence: float = Field(ge=0.0, le=1.0, description="Optical confidence in [0, 1] (FR-PRS-005)")
    bbox_preprocessed: BoundingBox
    bbox_original: BoundingBox
    reading_order_index: int = Field(ge=0)


class CheckboxMark(BaseModel):
    """
    PRD FR-PRS-002:
    Visual checkbox rendered strictly as '[x]' (checked), '[ ]' (unchecked),
    or '[?]' (uncertain / degraded mark requiring review).
    """
    mark_id: str
    state: Literal["[x]", "[ ]", "[?]"]
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    bbox_preprocessed: BoundingBox
    bbox_original: BoundingBox
    is_uncertain: bool = False

    @model_validator(mode="after")
    def validate_uncertain_state(self) -> "CheckboxMark":
        if self.state == "[?]":
            self.is_uncertain = True
        return self


class KeyValuePair(BaseModel):
    """PRD FR-PRS-004: Key-value pair with dual bounding boxes."""
    key_text: str
    value_text: str
    key_bbox_original: BoundingBox
    value_bbox_original: BoundingBox
    ocr_confidence: float = Field(ge=0.0, le=1.0)


class HTMLTable(BaseModel):
    """PRD FR-PRS-001, FR-PRS-004: Table rendered in HTML."""
    table_id: str
    html_content: str
    bbox_original: BoundingBox
    row_count: int = Field(ge=1)
    col_count: int = Field(ge=1)
    ocr_confidence: float = Field(ge=0.0, le=1.0)


class CharacterBoxEvidence(BaseModel):
    """
    Evidence tracing for an individual character box in a reconstructed character grid.
    Preserves SCH-013 bounding box and optical confidence provenance.
    """
    char: str
    char_index: int = Field(ge=0)
    bbox_preprocessed: BoundingBox
    bbox_original: BoundingBox
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    is_uncertain: bool = False


class CharacterGridMark(BaseModel):
    """
    PRD FR-PRS-003:
    Reconstructed character grid merged into one value (e.g. '|J|O|H|N|' -> 'JOHN').
    Retains per-character box evidence and SCH-013 enclosing bounding box.
    """
    grid_id: str
    reconstructed_text: str
    mean_ocr_confidence: float = Field(ge=0.0, le=1.0)
    bbox_preprocessed: BoundingBox
    bbox_original: BoundingBox
    character_boxes: List[CharacterBoxEvidence] = Field(default_factory=list)
    has_ambiguity: bool = False


class PageLayoutOutput(BaseModel):
    """Complete layout parsing output for an individual page."""
    page_number: int = Field(ge=1)
    markdown_content: str
    blocks: List[LayoutBlock] = Field(default_factory=list)
    checkboxes: List[CheckboxMark] = Field(default_factory=list)
    key_value_pairs: List[KeyValuePair] = Field(default_factory=list)
    tables: List[HTMLTable] = Field(default_factory=list)
    character_grids: List[CharacterGridMark] = Field(default_factory=list)
    mean_ocr_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    page_image_path: Optional[str] = Field(
        default=None,
        description="Local path of the preprocessed 200 DPI page image, so Module C can run the vision model on it",
    )


# ============================================================================
# Classification Contract (FR-CLS-001 through FR-CLS-004, GR-004)
# ============================================================================

class ConfidenceProvenance(str, Enum):
    """Provenance tracking for confidence scores."""
    MODEL = "model"              # Derived from calibrated model logprobs / scores
    RULE_HEURISTIC = "heuristic" # Derived from deterministic rule/regex match weights
    UNCALIBRATED = "uncalibrated"# Default nominal score, not a validated probability


class ClassificationOutput(BaseModel):
    """
    PRD FR-CLS-001, FR-CLS-002, FR-CLS-003:
    Classification result for the document.
    """
    domain: MVPClassificationDomain
    document_type: MVPDocumentType
    classification_confidence: float = Field(ge=0.0, le=1.0)
    confidence_provenance: ConfidenceProvenance = ConfidenceProvenance.UNCALIBRATED
    schema_id: Optional[str] = None
    schema_version: Optional[str] = None
    review_required: bool = False
    classifier_identity: Optional[str] = None
    classification_reasons: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_classification_policy(self) -> "ClassificationOutput":
        """
        PRD FR-CLS-002 & GR-004:
        If document_type is UNKNOWN or review_required is True:
        - schema_id and schema_version MUST be None (no extraction schema bound).
        - review_required MUST be True.
        """
        if self.document_type == MVPDocumentType.UNKNOWN:
            self.review_required = True
            if self.schema_id is not None or self.schema_version is not None:
                raise ValueError("UNKNOWN document type cannot bind an extraction schema.")
        if self.review_required and (self.schema_id is not None or self.schema_version is not None):
            raise ValueError("Document requiring review cannot have bound extraction schema.")
        return self


# ============================================================================
# Terminal Module B Output Envelope (ARC-003, SCH-011, SCH-012)
# ============================================================================

class ModuleBOutputPayload(BaseModel):
    """
    Complete, versioned handoff payload emitted by Module B to Module C.
    Adheres strictly to PRD SCH-011 and SCH-012.
    """
    contract_version: str = MODULE_B_CONTRACT_VERSION
    run_id: str
    document_id: str
    tenant_id: str
    status: Literal["processing", "needs_review", "failed"]
    classification: ClassificationOutput
    pages: List[PageLayoutOutput] = Field(default_factory=list)
    combined_markdown: str
    parser_model_name: str
    classifier_model_name: str
    processing_time_ms: float = Field(ge=0.0)
    created_at: str
    completed_at: str
    error: Optional[Dict[str, Any]] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status_alignment(self) -> "ModuleBOutputPayload":
        """Enforces SCH-012 status transitions."""
        if self.classification.review_required and self.status == "processing":
            raise ValueError(
                "Document requires review; status cannot remain 'processing'. Expected 'needs_review'."
            )
        if self.status == "failed" and self.error is None:
            raise ValueError("Status 'failed' requires error details.")
        return self

