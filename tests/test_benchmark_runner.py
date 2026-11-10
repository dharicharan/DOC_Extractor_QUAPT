"""
Unit and Integration Tests for TestHarnessRunner (TASK-4.3 Stage 3).
Covers PRD Requirements and Four-Point Integrity Audit:
- SCH-007: Test harness running on sample documents with ground truth reporting per-field precision, recall, F1, calibration.
- M1: Field F1 benchmarks (≥92% all fields, ≥95% key fields, macro vs micro aggregation).
- M6: 10-bin confidence calibration gap (≤3.0 points).
- M11 / GR-027: Release gate escape evaluation (strictly on locked_test split).
- GR-003: Honest null-if-absent handling and false-fill rate reporting.
- GR-005: Schema-constrained decoding conformance and 1,000-run accounting.
- US-DOCX-010: Test a schema before approval.
- Check 1: Actual SHA-256 document integrity against source artifact bytes or canonical payload.
- Check 2: Synthetic data exclusion from release-gate metrics (diagnostic separation).
- Check 3: Trusted principal-based locked_test authorization and fail-closed security.
"""

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import pytest
from pydantic import ValidationError

from doc_extractor.contracts_b import (
    BoundingBox,
    CheckboxMark,
    ClassificationOutput,
    ConfidenceProvenance,
    LayoutBlock,
    LayoutElementType,
    MVPClassificationDomain,
    MVPDocumentType,
    ModuleBOutputPayload,
    PageLayoutOutput,
)
from doc_extractor.contracts_c import (
    ExecutionMode,
    ExecutionProvenance,
)
from doc_extractor.dynamic_adapter import DynamicExtractionAdapter
from doc_extractor.exceptions import LLMExecutionError
from doc_extractor.extractor import SchemaGuidedExtractor
from schema_lifecycle.enums import FieldType, RuleOnFail, SchemaStatus
from schema_lifecycle.exceptions import (
    InvalidStateTransitionError,
    SchemaNotFoundError,
    TenantAccessError,
)
from schema_lifecycle.manager import SchemaLifecycleManager
from schema_lifecycle.metamodel import FieldDefinition, RuleDefinition, SchemaDefinition
from schema_lifecycle.repository import InMemorySchemaRepository
from schema_lifecycle.testing.contracts import (
    DatasetSplit,
    EvaluationDataset,
    EvaluationSample,
    GroundTruthDocument,
    GroundTruthField,
    MatchStrategy,
)
from schema_lifecycle.testing.policy import EvaluationPolicy
from schema_lifecycle.testing.runner import (
    AuthorizationContext,
    AuthorizationDecision,
    BenchmarkReport,
    DefaultLockedSetAuthorizer,
    DocumentIntegrityError,
    GR005Accounting,
    InMemoryPayloadResolver,
    LocalPayloadResolver,
    LockedSetAccessError,
    SampleObservation,
    TestHarnessRunner,
    TrustedPrincipal,
    map_envelope_to_sample_prediction,
)


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

DUMMY_BYTES_1 = b"%PDF-1.4 applicant_doc_001 ananya sharma"
DUMMY_HASH_1 = hashlib.sha256(DUMMY_BYTES_1).hexdigest()

DUMMY_BYTES_2 = b"%PDF-1.4 applicant_doc_002 rohit verma"
DUMMY_HASH_2 = hashlib.sha256(DUMMY_BYTES_2).hexdigest()

DUMMY_BYTES_3 = b"%PDF-1.4 applicant_doc_003 priya menon"
DUMMY_HASH_3 = hashlib.sha256(DUMMY_BYTES_3).hexdigest()


@pytest.fixture
def lifecycle_manager() -> SchemaLifecycleManager:
    repo = InMemorySchemaRepository()
    return SchemaLifecycleManager(repository=repo)


@pytest.fixture
def trusted_authorizer() -> DefaultLockedSetAuthorizer:
    authorizer = DefaultLockedSetAuthorizer()
    authorizer.register_principal(
        TrustedPrincipal(
            principal_id="gate_auditor_bob",
            roles=["GATE_APPROVER"],
            permissions=["EXECUTE_LOCKED_TEST"],
            tenant_scopes=["tenant_alpha"],
            dataset_scopes=["ds_lock_01", "ds_locked_01", "*"],
        )
    )
    authorizer.register_principal(
        TrustedPrincipal(
            principal_id="lead_auditor",
            roles=["GATE_APPROVER"],
            permissions=["EXECUTE_LOCKED_TEST"],
            tenant_scopes=["tenant_alpha", "*"],
            dataset_scopes=["*"],
        )
    )
    authorizer.register_principal(
        TrustedPrincipal(
            principal_id="admin_auditor",
            roles=["SYSTEM_ADMIN"],
            permissions=["EXECUTE_LOCKED_TEST"],
            tenant_scopes=["*"],
            dataset_scopes=["*"],
        )
    )
    authorizer.register_principal(
        TrustedPrincipal(
            principal_id="intern_dave",
            roles=["DEVELOPER"],
            permissions=["READ_DATASET"],
            tenant_scopes=["tenant_alpha"],
            dataset_scopes=["*"],
        )
    )
    return authorizer


def build_test_schema_def(
    schema_id: str = "kyc_test_schema",
    version: int = 1,
    tenant_id: str = "tenant_alpha",
) -> SchemaDefinition:
    return SchemaDefinition(
        schema_id=schema_id,
        version=version,
        tenant_id=tenant_id,
        domain="BFSI",
        status=SchemaStatus.DRAFT,
        fields=[
            FieldDefinition(name="applicant.full_name", type=FieldType.STRING, required=True),
            FieldDefinition(name="applicant.pan_number", type=FieldType.STRING, required=False, format="PAN"),
            FieldDefinition(name="applicant.address.city", type=FieldType.STRING, required=False),
            FieldDefinition(name="services.opt_in_online", type=FieldType.BOOLEAN, required=False),
            FieldDefinition(name="billing.total_amount", type=FieldType.DECIMAL, required=False),
            FieldDefinition(name="optional_missing_field", type=FieldType.STRING, required=False),
            FieldDefinition(name="required_missing_field", type=FieldType.STRING, required=True, nullable=True),
        ],
        rules=[
            RuleDefinition(
                id="PAN_01",
                expression="regex:^[A-Z]{5}[0-9]{4}[A-Z]$",
                target_fields=["applicant.pan_number"],
                error_code="INVALID_FORMAT",
                on_fail=RuleOnFail.REVIEW,
            ),
        ],
    )


def create_sample_payload(
    doc_id: str = "doc_001",
    doc_hash: str = DUMMY_HASH_1,
    source_bytes: bytes = None,
    full_name: str = "ANANYA SHARMA",
    pan: str = "ABCDE1234F",
    city: str = "BENGALURU",
    total_amount: str = "8000",
    opt_in: bool = True,
    default_val: str = "DEFAULT_VAL",
    status: str = "processing",
) -> ModuleBOutputPayload:
    if source_bytes is None:
        if doc_hash == DUMMY_HASH_1:
            source_bytes = DUMMY_BYTES_1
        elif doc_hash == DUMMY_HASH_2:
            source_bytes = DUMMY_BYTES_2
        elif doc_hash == DUMMY_HASH_3:
            source_bytes = DUMMY_BYTES_3
        else:
            source_bytes = f"RAW_BYTES_{doc_id}_{doc_hash}".encode("utf-8")

    b_name = LayoutBlock(
        block_id="b_name",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text=f"Applicant Name: {full_name}",
        ocr_confidence=0.98,
        bbox_preprocessed=BoundingBox(x0=10, y0=50, x1=200, y1=70),
        bbox_original=BoundingBox(x0=10, y0=50, x1=200, y1=70),
        reading_order_index=1,
    )
    b_pan = LayoutBlock(
        block_id="b_pan",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text=f"PAN: {pan}",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=80, x1=150, y1=100),
        bbox_original=BoundingBox(x0=10, y0=80, x1=150, y1=100),
        reading_order_index=2,
    )
    b_city = LayoutBlock(
        block_id="b_city",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text=f"City: {city}",
        ocr_confidence=0.97,
        bbox_preprocessed=BoundingBox(x0=10, y0=110, x1=150, y1=130),
        bbox_original=BoundingBox(x0=10, y0=110, x1=150, y1=130),
        reading_order_index=3,
    )
    b_amt = LayoutBlock(
        block_id="b_amt",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text=f"Total Bill: {total_amount}",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=140, x1=200, y1=160),
        bbox_original=BoundingBox(x0=10, y0=140, x1=200, y1=160),
        reading_order_index=4,
    )
    b_req = LayoutBlock(
        block_id="b_req",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text=f"Required Missing Field: {default_val}",
        ocr_confidence=0.98,
        bbox_preprocessed=BoundingBox(x0=10, y0=200, x1=200, y1=220),
        bbox_original=BoundingBox(x0=10, y0=200, x1=200, y1=220),
        reading_order_index=5,
    )
    cb_opt_in = CheckboxMark(
        mark_id="cb_opt_in",
        state="[x]" if opt_in else "[ ]",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=170, x1=30, y1=190),
        bbox_original=BoundingBox(x0=10, y0=170, x1=30, y1=190),
    )

    page = PageLayoutOutput(
        page_number=1,
        markdown_content=f"Applicant Name: {full_name}\nPAN: {pan}\nCity: {city}\nTotal Bill: {total_amount}\nRequired Missing Field: {default_val}",
        blocks=[b_name, b_pan, b_city, b_amt, b_req],
        checkboxes=[cb_opt_in],
        mean_ocr_confidence=0.98,
    )

    return ModuleBOutputPayload(
        run_id=f"run_{doc_id}",
        document_id=doc_id,
        tenant_id="tenant_alpha",
        status=status,
        classification=ClassificationOutput(
            domain=MVPClassificationDomain.BFSI,
            document_type=MVPDocumentType.DOC_01_CASA,
            classification_confidence=0.98,
            confidence_provenance=ConfidenceProvenance.RULE_HEURISTIC,
            schema_id="kyc_test_schema",
            schema_version="1",
        ),
        pages=[page],
        combined_markdown=page.markdown_content,
        parser_model_name="mock-layout-parser",
        classifier_model_name="mock-classifier",
        processing_time_ms=50.0,
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
        extra_metadata={
            "document_hash_sha256": doc_hash,
            "source_bytes": source_bytes,
        },
    )


def build_evaluation_sample(
    sample_id: str,
    doc_hash: str,
    split: DatasetSplit = DatasetSplit.DEVELOPMENT,
    is_synthetic: bool = False,
    applicant_name: str = "ANANYA SHARMA",
    pan: str = "ABCDE1234F",
    city: str = "BENGALURU",
    total_amount: float = 8000.0,
    opt_in: bool = True,
    annotators: Optional[list[str]] = None,
    adjudicated: bool = True,
) -> EvaluationSample:
    annotator_ids = annotators or (["ann_1", "ann_2"] if split == DatasetSplit.LOCKED_TEST else ["ann_1"])
    gt_doc = GroundTruthDocument(
        sample_id=sample_id,
        document_hash_sha256=doc_hash,
        annotator_ids=annotator_ids,
        adjudicated=adjudicated,
        adjudicator_id="lead_reviewer" if adjudicated else None,
        fields={
            "applicant.full_name": GroundTruthField(
                field_path="applicant.full_name",
                expected_value=applicant_name,
                is_key_field=True,
                match_strategy=MatchStrategy.EXACT_NORMALIZED,
            ),
            "applicant.pan_number": GroundTruthField(
                field_path="applicant.pan_number",
                expected_value=pan,
                is_key_field=True,
                match_strategy=MatchStrategy.EXACT_NORMALIZED,
            ),
            "applicant.address.city": GroundTruthField(
                field_path="applicant.address.city",
                expected_value=city,
                is_key_field=False,
                match_strategy=MatchStrategy.EXACT_NORMALIZED,
            ),
            "services.opt_in_online": GroundTruthField(
                field_path="services.opt_in_online",
                expected_value=opt_in,
                is_key_field=False,
                match_strategy=MatchStrategy.BOOLEAN_EXACT,
            ),
            "billing.total_amount": GroundTruthField(
                field_path="billing.total_amount",
                expected_value=total_amount,
                is_key_field=False,
                match_strategy=MatchStrategy.NUMERIC_TOLERANCE,
            ),
            "optional_missing_field": GroundTruthField(
                field_path="optional_missing_field",
                expected_value=None,
                is_null=True,
                is_key_field=False,
            ),
            "required_missing_field": GroundTruthField(
                field_path="required_missing_field",
                expected_value="DEFAULT_VAL",
                is_key_field=False,
                match_strategy=MatchStrategy.EXACT_NORMALIZED,
            ),
        },
    )

    return EvaluationSample(
        sample_id=sample_id,
        tenant_id="tenant_alpha",
        document_type="DOC-01",
        document_ref=f"storage://eval/samples/{sample_id}.pdf",
        document_hash_sha256=doc_hash,
        stratum="clean_print",
        ground_truth=gt_doc,
        metadata={"is_synthetic": is_synthetic},
    )


# ============================================================================
# 1. Development split end-to-end
# ============================================================================
def test_runner_development_split_end_to_end(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample("SMP_DEV_01", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
    dataset = EvaluationDataset(
        dataset_id="ds_dev_01",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )

    payload = create_sample_payload("doc_dev_01", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, "req_dev_1"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
    )

    report = runner.run_benchmark(dataset)

    assert isinstance(report, BenchmarkReport)
    assert report.dataset_split == DatasetSplit.DEVELOPMENT
    assert report.total_samples == 1
    assert report.failed_samples == 0
    assert report.metrics_decision.overall_pass is True
    assert report.aggregate_metrics.macro_f1_all_fields == 1.0
    assert report.aggregate_metrics.macro_f1_key_fields == 1.0
    assert report.overall_pass is True
    assert report.gate_eligible is False
    assert any("GR005_INCOMPLETE" in r for r in report.gate_eligibility_reasons)


# ============================================================================
# 2. Tuning split end-to-end
# ============================================================================
def test_runner_tuning_split_end_to_end(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample("SMP_TUNE_01", DUMMY_HASH_1, split=DatasetSplit.TUNING)
    dataset = EvaluationDataset(
        dataset_id="ds_tune_01",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.TUNING,
        samples=[sample],
    )

    payload = create_sample_payload("doc_tune_01", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, "req_tune_1"

    extractor = SchemaGuidedExtractor(
        prompt_version="2.1.0-tune",
        use_llm=True,
        allow_fallback=False,
        llm_caller=mock_caller,
    )
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
    )

    report = runner.run_benchmark(dataset)

    assert report.dataset_split == DatasetSplit.TUNING
    assert "2.1.0-tune" in report.prompt_versions
    assert report.metrics_decision.overall_pass is True


# ============================================================================
# 3. Authorized locked-test execution
# ============================================================================
def test_runner_authorized_locked_test_execution(lifecycle_manager, trusted_authorizer):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample(
        "SMP_LOCK_01",
        DUMMY_HASH_1,
        split=DatasetSplit.LOCKED_TEST,
        annotators=["ann_lead", "ann_second"],
        adjudicated=True,
    )
    dataset = EvaluationDataset(
        dataset_id="ds_lock_01",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=[sample],
    )

    payload = create_sample_payload("doc_lock_01", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, "req_lock_1"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
        authorizer=trusted_authorizer,
        min_locked_samples=1,  # unit test boundary
    )

    auth_context = AuthorizationContext(
        principal_id="gate_auditor_bob",
        event_reason="MVP_RELEASE_GATE_G1",
    )

    report = runner.run_benchmark(dataset, auth_context=auth_context)

    assert report.dataset_split == DatasetSplit.LOCKED_TEST
    assert report.metrics_decision.overall_pass is True
    assert report.aggregate_metrics.escape_count == 0
    assert report.metrics_decision.meets_m11_escapes is True
    assert report.gr005_accounting.status == "INCOMPLETE"
    assert report.gate_eligible is False
    assert report.overall_pass is False
    assert any("GR005_INCOMPLETE" in r for r in report.gate_eligibility_reasons)


# ============================================================================
# 4. Unauthorized locked-test rejection (Check 3)
# ============================================================================
def test_runner_unauthorized_locked_test_rejection(lifecycle_manager, trusted_authorizer):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample(
        "SMP_LOCK_01",
        DUMMY_HASH_1,
        split=DatasetSplit.LOCKED_TEST,
        annotators=["ann_1", "ann_2"],
        adjudicated=True,
    )
    dataset = EvaluationDataset(
        dataset_id="ds_lock_01",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=[sample],
    )

    payload = create_sample_payload("doc_lock_01", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
        authorizer=trusted_authorizer,
    )

    # intern_dave lacks EXECUTE_LOCKED_TEST permission in trusted registry
    unauthorized_context = AuthorizationContext(
        principal_id="intern_dave",
        event_reason="Trying unauthorized run",
    )

    with pytest.raises(LockedSetAccessError) as exc_info:
        runner.run_benchmark(dataset, auth_context=unauthorized_context)
    assert "authorization denied" in str(exc_info.value).lower()


# ============================================================================
# 5. Non-gate-grade ground truth rejection
# ============================================================================
def test_runner_non_gate_grade_ground_truth_rejection():
    # Attempt to construct locked_test dataset with single annotator
    sample = build_evaluation_sample(
        "SMP_INVALID_LOCK",
        DUMMY_HASH_1,
        split=DatasetSplit.DEVELOPMENT,  # build with 1 annotator
        annotators=["single_annotator"],
        adjudicated=False,
    )

    with pytest.raises(ValueError) as exc_info:
        EvaluationDataset(
            dataset_id="ds_lock_invalid",
            tenant_id="tenant_alpha",
            schema_id="kyc_test_schema",
            schema_version=1,
            document_type="DOC-01",
            dataset_split=DatasetSplit.LOCKED_TEST,
            samples=[sample],
        )
    assert "requires at least 2 distinct annotators" in str(exc_info.value)


# ============================================================================
# 6. M11 escape detection on locked_test
# ============================================================================
def test_runner_m11_escape_detection_on_locked_test(lifecycle_manager, trusted_authorizer):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample(
        "SMP_LOCK_ESC",
        DUMMY_HASH_1,
        split=DatasetSplit.LOCKED_TEST,
        applicant_name="ANANYA SHARMA",
        pan="ABCDE1234F",  # Ground Truth expects ABCDE1234F
        annotators=["ann_1", "ann_2"],
        adjudicated=True,
    )
    dataset = EvaluationDataset(
        dataset_id="ds_lock_esc",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=[sample],
    )

    # Document contains visually grounded PAN: XYZWR9999K
    payload = create_sample_payload("doc_lock_esc", DUMMY_HASH_1, pan="XYZWR9999K")
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    # Model predicts grounded PAN (XYZWR9999K); auto-approved, but mismatches ground truth (ABCDE1234F) -> ESCAPE!
    def mock_caller_escape(p, schema_cls):
        return {
            "applicant": {
                "full_name": "ANANYA SHARMA",
                "pan_number": "XYZWR9999K",  # Wrong key field relative to ground truth, but auto-approved!
                "address": {"city": "BENGALURU"},
            },
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, "req_esc"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller_escape)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
        authorizer=trusted_authorizer,
        min_locked_samples=1,
    )

    auth_context = AuthorizationContext(
        principal_id="lead_auditor",
        event_reason="M11_AUDIT",
    )

    report = runner.run_benchmark(dataset, auth_context=auth_context)

    assert report.aggregate_metrics.escape_count == 1
    assert report.metrics_decision.meets_m11_escapes is False
    assert report.overall_pass is False
    assert report.gate_eligible is False
    assert any("M11 escape violation" in v for v in report.metrics_decision.violations)


# ============================================================================
# 7. Payload hash mismatch rejection (Check 1)
# ============================================================================
def test_runner_payload_hash_mismatch_rejection(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample("SMP_HASH_01", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
    dataset = EvaluationDataset(
        dataset_id="ds_hash_01",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )

    # Payload carries hash DUMMY_HASH_2 and bytes DUMMY_BYTES_2 != sample's DUMMY_HASH_1
    payload_drift = create_sample_payload("doc_drift_01", doc_hash=DUMMY_HASH_2)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload_drift})

    adapter = DynamicExtractionAdapter(manager=lifecycle_manager)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    with pytest.raises(DocumentIntegrityError) as exc_info:
        runner.run_benchmark(dataset)
    assert "hash mismatch" in str(exc_info.value).lower()


# ============================================================================
# 8. Tenant and exact schema-version isolation
# ============================================================================
def test_runner_tenant_and_exact_version_isolation(lifecycle_manager):
    schema_v1 = build_test_schema_def(version=1)
    schema_v2 = build_test_schema_def(version=2)
    schema_v2.fields.append(FieldDefinition(name="v2_field", type=FieldType.STRING, required=False))

    lifecycle_manager.create_schema("tenant_alpha", schema_v1, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    lifecycle_manager.create_version("tenant_alpha", "kyc_test_schema", schema_v2, author="alice")
    # v2 left in DRAFT status!

    sample = build_evaluation_sample("SMP_V2_01", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
    dataset_v2 = EvaluationDataset(
        dataset_id="ds_v2",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=2,  # Requests v2 which is in DRAFT
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )

    payload = create_sample_payload("doc_v2", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    # Rejection of DRAFT status version
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        runner.run_benchmark(dataset_v2)
    assert "draft" in str(exc_info.value).lower()

    # Rejection of wrong tenant
    dataset_cross_tenant = EvaluationDataset(
        dataset_id="ds_cross",
        tenant_id="tenant_other",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[
            EvaluationSample(
                sample_id="SMP_CROSS",
                tenant_id="tenant_other",
                document_type="DOC-01",
                document_ref="storage://test",
                document_hash_sha256=DUMMY_HASH_1,
                ground_truth=GroundTruthDocument(
                    sample_id="SMP_CROSS",
                    document_hash_sha256=DUMMY_HASH_1,
                    annotator_ids=["ann"],
                ),
            )
        ],
    )
    with pytest.raises(SchemaNotFoundError):
        runner.run_benchmark(dataset_cross_tenant)


# ============================================================================
# 9. Heuristic/fallback provenance rejection
# ============================================================================
def test_runner_heuristic_fallback_provenance_rejection(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample("SMP_FALLBACK_01", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
    dataset = EvaluationDataset(
        dataset_id="ds_fb",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )

    payload = create_sample_payload("doc_fb", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    # Extractor with fallback enabled that triggers heuristic execution
    extractor = SchemaGuidedExtractor(use_llm=False, allow_fallback=True)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    with pytest.raises(LLMExecutionError) as exc_info:
        runner.run_benchmark(dataset)
    assert "untrusted" in str(exc_info.value).lower() or "provenance" in str(exc_info.value).lower()


# ============================================================================
# 10. Mock execution reporting and gate ineligibility
# ============================================================================
def test_runner_mock_execution_reporting_and_gate_ineligibility(lifecycle_manager, trusted_authorizer):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample(
        "SMP_MOCK_LOCK",
        DUMMY_HASH_1,
        split=DatasetSplit.LOCKED_TEST,
        annotators=["ann_1", "ann_2"],
        adjudicated=True,
    )
    dataset = EvaluationDataset(
        dataset_id="ds_mock_lock",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=[sample],
    )

    payload = create_sample_payload("doc_mock", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    # Mock extractor with allow_mock=True
    extractor_mock = SchemaGuidedExtractor(is_mock=True)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor_mock, allow_mock=True)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
        authorizer=trusted_authorizer,
        allow_mock=True,
        min_locked_samples=1,
    )

    auth_context = AuthorizationContext(
        principal_id="admin_auditor",
        event_reason="MOCK_TEST_RUN",
    )

    report = runner.run_benchmark(dataset, auth_context=auth_context)

    # Mock run is explicitly identified
    assert report.is_mock_run is True
    assert ExecutionMode.MOCK.value in report.execution_modes
    # Mock executions CANNOT qualify a release gate
    assert report.gate_eligible is False
    assert any("MOCK_EXECUTION" in r for r in report.gate_eligibility_reasons)


# ============================================================================
# 11. Missing versus explicitly null predictions
# ============================================================================
def test_runner_missing_vs_explicitly_null_predictions(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample("SMP_NULLS_01", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
    dataset = EvaluationDataset(
        dataset_id="ds_nulls",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )

    payload = create_sample_payload("doc_nulls", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    # Model explicitly returns null for optional_missing_field, and completely omits required_missing_field
    def mock_caller_omission(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "optional_missing_field": None,  # explicitly null
            "required_missing_field": None,  # missing required field
        }, "req_omission"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller_omission)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    report = runner.run_benchmark(dataset)

    obs = report.sample_observations[0]
    assert "optional_missing_field" in obs.explicit_null_fields
    # Ground truth expected a value for required_missing_field, but model predicted null
    assert report.field_metrics["required_missing_field"].false_negatives == 1
    assert report.field_metrics["required_missing_field"].f1 == 0.0


# ============================================================================
# 12. Synthetic-only and mixed synthetic/real datasets (Check 2)
# ============================================================================
def test_runner_synthetic_and_mixed_datasets(lifecycle_manager, trusted_authorizer):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    # Pure synthetic dataset on locked_test
    synthetic_sample = build_evaluation_sample(
        "SMP_SYNTH_01",
        DUMMY_HASH_1,
        split=DatasetSplit.LOCKED_TEST,
        is_synthetic=True,
        annotators=["ann_1", "ann_2"],
        adjudicated=True,
    )
    dataset_synthetic = EvaluationDataset(
        dataset_id="ds_synth_only",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=[synthetic_sample],
    )

    payload = create_sample_payload("doc_synth", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({synthetic_sample.sample_id: payload})

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, "req_synth"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
        authorizer=trusted_authorizer,
        min_locked_samples=1,
    )

    auth = AuthorizationContext(principal_id="lead_auditor", event_reason="SYNTH_DIAGNOSTICS")
    report_synth = runner.run_benchmark(dataset_synthetic, auth_context=auth)

    assert report_synth.synthetic_samples == 1
    assert report_synth.real_samples == 0
    # Synthetic-only cannot qualify release gate (PRD §13.1)
    assert report_synth.gate_eligible is False
    assert any("SYNTHETIC_ONLY_DATASET" in r for r in report_synth.gate_eligibility_reasons)
    # Diagnostic synthetic aggregate metrics are captured
    assert report_synth.synthetic_aggregate_metrics is not None
    assert report_synth.synthetic_aggregate_metrics.total_samples == 1


# ============================================================================
# 13. Insufficient data across all applicable metric gates
# ============================================================================
def test_runner_insufficient_data_rejection(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    # Sample with ZERO key fields (all fields are is_key_field=False)
    gt_no_key = GroundTruthDocument(
        sample_id="SMP_NO_KEY",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["ann_1"],
        fields={
            "applicant.full_name": GroundTruthField(
                field_path="applicant.full_name",
                expected_value="ANANYA SHARMA",
                is_key_field=False,
            ),
            "applicant.address.city": GroundTruthField(
                field_path="applicant.address.city",
                expected_value="BENGALURU",
                is_key_field=False,
            ),
        },
    )
    sample_no_key = EvaluationSample(
        sample_id="SMP_NO_KEY",
        tenant_id="tenant_alpha",
        document_type="DOC-01",
        document_ref="storage://test",
        document_hash_sha256=DUMMY_HASH_1,
        ground_truth=gt_no_key,
    )
    dataset = EvaluationDataset(
        dataset_id="ds_no_key",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample_no_key],
    )

    payload = create_sample_payload("doc_no_key", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample_no_key.sample_id: payload})

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "address": {"city": "BENGALURU"}},
            "required_missing_field": "DEFAULT_VAL",
        }, "req_nk"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    report = runner.run_benchmark(dataset)

    # Missing key fields prevents accidental pass!
    assert report.metrics_decision.has_insufficient_data is True
    assert report.metrics_decision.overall_pass is False
    assert any("INSUFFICIENT_DATA" in v for v in report.metrics_decision.violations)


# ============================================================================
# 14. Dataset-level key-field consistency
# ============================================================================
def test_runner_key_field_consistency(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample("SMP_KF_01", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
    dataset = EvaluationDataset(
        dataset_id="ds_kf",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )

    payload = create_sample_payload("doc_kf", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, "req_kf"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    report = runner.run_benchmark(dataset)

    assert report.aggregate_metrics.key_field_count == 2
    assert report.field_metrics["applicant.full_name"].is_key_field is True
    assert report.field_metrics["applicant.pan_number"].is_key_field is True
    assert report.field_metrics["applicant.address.city"].is_key_field is False
    assert report.metrics_decision.meets_m1_key_fields is True


# ============================================================================
# 15. Local resolver path traversal and symlink protection (Check 1)
# ============================================================================
def test_local_resolver_security(tmp_path):
    dataset_dir = tmp_path / "dataset_root"
    dataset_dir.mkdir()
    outside_dir = tmp_path / "outside_root"
    outside_dir.mkdir()

    # Legitimate payload
    legit_payload = create_sample_payload("doc_legit", DUMMY_HASH_1)
    legit_file = dataset_dir / "sample_001.json"
    legit_file.write_text(legit_payload.model_dump_json())
    actual_hash_1 = DUMMY_HASH_1  # Document artifact hash is DUMMY_HASH_1

    resolver = LocalPayloadResolver(base_dir=dataset_dir)

    # 1. Valid resolution
    sample_valid = EvaluationSample(
        sample_id="SMP_001",
        tenant_id="tenant_alpha",
        document_type="DOC-01",
        document_ref="sample_001.json",
        document_hash_sha256=actual_hash_1,
        ground_truth=GroundTruthDocument(
            sample_id="SMP_001",
            document_hash_sha256=actual_hash_1,
            annotator_ids=["ann_1"],
        ),
    )
    resolved = resolver.resolve_payload(sample_valid)
    assert resolved.document_id == "doc_legit"

    # 2. Path traversal rejection
    sample_traversal = EvaluationSample(
        sample_id="SMP_TRAV",
        tenant_id="tenant_alpha",
        document_type="DOC-01",
        document_ref="../outside_root/secret.json",
        document_hash_sha256=DUMMY_HASH_1,
        ground_truth=GroundTruthDocument(
            sample_id="SMP_TRAV",
            document_hash_sha256=DUMMY_HASH_1,
            annotator_ids=["ann_1"],
        ),
    )
    with pytest.raises(DocumentIntegrityError, match="Path traversal or symlink escape"):
        resolver.resolve_payload(sample_traversal)

    # 3. Symlink escape rejection
    outside_file = outside_dir / "target.json"
    outside_file.write_text(legit_payload.model_dump_json())
    symlink_file = dataset_dir / "escape_symlink.json"
    try:
        symlink_file.symlink_to(outside_file)
        sample_symlink = EvaluationSample(
            sample_id="SMP_SYM",
            tenant_id="tenant_alpha",
            document_type="DOC-01",
            document_ref="escape_symlink.json",
            document_hash_sha256=actual_hash_1,
            ground_truth=GroundTruthDocument(
                sample_id="SMP_SYM",
                document_hash_sha256=actual_hash_1,
                annotator_ids=["ann_1"],
            ),
        )
        with pytest.raises(DocumentIntegrityError, match="Path traversal or symlink escape"):
            resolver.resolve_payload(sample_symlink)
    finally:
        if symlink_file.is_symlink():
            symlink_file.unlink()

    # 4. Missing file rejection
    sample_missing = EvaluationSample(
        sample_id="SMP_MISS",
        tenant_id="tenant_alpha",
        document_type="DOC-01",
        document_ref="non_existent.json",
        document_hash_sha256=DUMMY_HASH_1,
        ground_truth=GroundTruthDocument(
            sample_id="SMP_MISS",
            document_hash_sha256=DUMMY_HASH_1,
            annotator_ids=["ann_1"],
        ),
    )
    with pytest.raises(DocumentIntegrityError, match="file not found"):
        resolver.resolve_payload(sample_missing)


# ============================================================================
# 16. Per-sample provenance preservation
# ============================================================================
def test_runner_per_sample_provenance_preservation(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample1 = build_evaluation_sample("SMP_PROV_1", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
    sample2 = build_evaluation_sample("SMP_PROV_2", DUMMY_HASH_2, split=DatasetSplit.DEVELOPMENT)
    dataset = EvaluationDataset(
        dataset_id="ds_prov",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample1, sample2],
    )

    payload1 = create_sample_payload("doc_prov_1", DUMMY_HASH_1)
    payload2 = create_sample_payload("doc_prov_2", DUMMY_HASH_2)
    resolver = InMemoryPayloadResolver({
        sample1.sample_id: payload1,
        sample2.sample_id: payload2,
    })

    # Return distinct provider request IDs for each document
    def mock_caller(payload, schema_cls):
        req_id = f"req_{payload.document_id}"
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, req_id

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    report = runner.run_benchmark(dataset)

    obs1 = next(o for o in report.sample_observations if o.sample_id == "SMP_PROV_1")
    obs2 = next(o for o in report.sample_observations if o.sample_id == "SMP_PROV_2")

    assert obs1.provenance.request_id == "req_doc_prov_1"
    assert obs2.provenance.request_id == "req_doc_prov_2"
    assert obs1.provenance.mode == ExecutionMode.GENUINE_LLM
    assert obs2.provenance.mode == ExecutionMode.GENUINE_LLM


# ============================================================================
# 17. GR-005 run accounting and the 1,000-run boundary
# ============================================================================
def test_runner_gr005_accounting_and_boundary(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample("SMP_GR005", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
    dataset = EvaluationDataset(
        dataset_id="ds_gr005",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )

    payload = create_sample_payload("doc_gr005", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, "req_gr005"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
        gr005_target_runs=1000,
    )

    report = runner.run_benchmark(dataset)

    acc: GR005Accounting = report.gr005_accounting
    assert acc.status == "INCOMPLETE"
    assert acc.total_attempted_runs == 1
    assert acc.successful_validations == 1
    assert acc.unique_qualifying_runs == 1
    assert acc.validation_exceptions == 0
    assert acc.required_runs == 1000
    assert acc.is_conformance_verified is False
    assert "remaining 999 runs" in acc.limitation_note


# ============================================================================
# 18. Deterministic BenchmarkReport serialization
# ============================================================================
def test_runner_deterministic_report_serialization(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample("SMP_SER_01", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
    dataset = EvaluationDataset(
        dataset_id="ds_ser",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )

    payload = create_sample_payload("doc_ser", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, "req_ser"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    report = runner.run_benchmark(dataset)

    json_str = report.to_json(indent=2)
    parsed = json.loads(json_str)

    assert parsed["benchmark_id"] == report.benchmark_id
    assert parsed["dataset_id"] == "ds_ser"
    assert parsed["schema_id"] == "kyc_test_schema"
    assert parsed["aggregate_metrics"]["macro_f1_all_fields"] == 1.0


# ============================================================================
# 19. Failed sample handling without silent exclusion
# ============================================================================
def test_runner_failed_sample_handling(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample("SMP_FAIL_01", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
    dataset = EvaluationDataset(
        dataset_id="ds_fail",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )

    payload = create_sample_payload("doc_fail", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    # Extractor caller raises an unhandled exception
    def mock_caller_failing(p, schema_cls):
        raise RuntimeError("GPU OOM / Internal Provider Connection Refused")

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller_failing)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    report = runner.run_benchmark(dataset)

    assert report.failed_samples == 1
    assert report.gate_eligible is False
    assert any("EXECUTION_FAILURES" in r for r in report.gate_eligibility_reasons)
    assert report.sample_observations[0].error is not None
    assert "Connection Refused" in report.sample_observations[0].error


# ============================================================================
# 20. Locked-test access without a trusted authorization provider
# ============================================================================
def test_runner_locked_test_without_auth_context(lifecycle_manager):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample(
        "SMP_NO_AUTH",
        DUMMY_HASH_1,
        split=DatasetSplit.LOCKED_TEST,
        annotators=["ann_1", "ann_2"],
        adjudicated=True,
    )
    dataset = EvaluationDataset(
        dataset_id="ds_no_auth",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=[sample],
    )

    payload = create_sample_payload("doc_no_auth", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    # Calling run_benchmark without auth_context fails closed
    with pytest.raises(LockedSetAccessError) as exc_info:
        runner.run_benchmark(dataset, auth_context=None)
    assert "requires a trusted authorizationcontext" in str(exc_info.value).lower()


# ============================================================================
# CHECK 1 REGRESSION TESTS: Actual SHA-256 Document Integrity
# ============================================================================

def test_check1_actual_source_bytes_sha256_match_succeeds():
    """Verify that payload resolution succeeds when actual source bytes hash matches document_hash_sha256."""
    actual_bytes = b"real document raw binary or text content"
    expected_hash = hashlib.sha256(actual_bytes).hexdigest()

    payload = create_sample_payload("doc_c1_ok", doc_hash=expected_hash, source_bytes=actual_bytes)
    resolver = InMemoryPayloadResolver()
    resolver.register("SMP_C1_OK", payload, raw_bytes=actual_bytes)

    sample = build_evaluation_sample("SMP_C1_OK", expected_hash)
    resolved = resolver.resolve_payload(sample)
    assert resolved.document_id == "doc_c1_ok"


def test_check1_modified_source_bytes_sha256_mismatch_fails():
    """Verify that modified source bytes are detected and rejected with DocumentIntegrityError."""
    original_bytes = b"original content"
    original_hash = hashlib.sha256(original_bytes).hexdigest()

    tampered_bytes = b"tampered content"
    # Payload contains tampered bytes, while sample expects original_hash
    payload = create_sample_payload("doc_c1_tamper", doc_hash=original_hash, source_bytes=tampered_bytes)
    resolver = InMemoryPayloadResolver({"SMP_C1_TAMPER": payload})

    sample = build_evaluation_sample("SMP_C1_TAMPER", original_hash)
    with pytest.raises(DocumentIntegrityError, match="hash mismatch"):
        resolver.resolve_payload(sample)


def test_check1_metadata_claims_expected_hash_when_bytes_differ_rejected():
    """
    PRD Check 1 Critical: Comparing two metadata fields is not sufficient.
    Reject metadata claiming the expected hash when bytes differ.
    """
    actual_bytes = b"actual document bytes on disk"
    actual_hash = hashlib.sha256(actual_bytes).hexdigest()
    spoofed_expected_hash = "f" * 64

    # The payload metadata asserts document_hash_sha256 = spoofed_expected_hash, but actual bytes hash to actual_hash
    payload = create_sample_payload("doc_c1_spoof", doc_hash=spoofed_expected_hash, source_bytes=actual_bytes)
    # Both sample and metadata claim spoofed_expected_hash!
    sample = build_evaluation_sample("SMP_C1_SPOOF", spoofed_expected_hash)
    resolver = InMemoryPayloadResolver({"SMP_C1_SPOOF": payload})

    with pytest.raises(DocumentIntegrityError) as exc_info:
        resolver.resolve_payload(sample)
    assert "hash mismatch" in str(exc_info.value).lower() or "untrusted" in str(exc_info.value).lower()


def test_check1_missing_source_artifacts_rejected(tmp_path):
    """Verify that missing source artifacts are strictly rejected."""
    # 1. InMemory resolver with missing raw bytes
    payload = create_sample_payload("doc_missing_bytes", DUMMY_HASH_1)
    # Strip source_bytes from metadata
    payload.extra_metadata.pop("source_bytes", None)
    resolver_inmem = InMemoryPayloadResolver({"SMP_NO_BYTES": payload})
    sample_inmem = build_evaluation_sample("SMP_NO_BYTES", DUMMY_HASH_1)

    with pytest.raises(DocumentIntegrityError, match="Missing source artifact bytes"):
        resolver_inmem.resolve_payload(sample_inmem)

    # 2. Local resolver with missing file on filesystem
    dataset_dir = tmp_path / "check1_dir"
    dataset_dir.mkdir()
    resolver_local = LocalPayloadResolver(base_dir=dataset_dir)
    sample_local = build_evaluation_sample("SMP_MISSING_FILE", DUMMY_HASH_1)
    sample_local = sample_local.model_copy(update={"document_ref": "nonexistent_doc.pdf"})

    with pytest.raises(DocumentIntegrityError, match="file not found"):
        resolver_local.resolve_payload(sample_local)


def test_check1_canonical_payload_hashing_contract():
    """
    Verify that canonical payload hashing is permitted ONLY when the dataset contract
    explicitly defines sample.metadata['hash_target'] == 'canonical_payload'.
    """
    payload = create_sample_payload("doc_c1_canonical", DUMMY_HASH_1)
    payload.extra_metadata.pop("source_bytes", None)

    # Calculate actual canonical serialization hash
    canonical_bytes = payload.model_dump_json().encode("utf-8")
    canonical_hash = hashlib.sha256(canonical_bytes).hexdigest()

    resolver = InMemoryPayloadResolver({"SMP_CANONICAL": payload})

    # Sample with explicit hash_target contract
    sample_valid = build_evaluation_sample("SMP_CANONICAL", canonical_hash)
    sample_valid = sample_valid.model_copy(update={"metadata": {"hash_target": "canonical_payload"}})

    resolved = resolver.resolve_payload(sample_valid)
    assert resolved.document_id == "doc_c1_canonical"

    # If canonical hash does not match, it must fail
    sample_invalid = build_evaluation_sample("SMP_CANONICAL", "0" * 64)
    sample_invalid = sample_invalid.model_copy(update={"metadata": {"hash_target": "canonical_payload"}})

    with pytest.raises(DocumentIntegrityError, match="Canonical payload SHA-256 hash mismatch"):
        resolver.resolve_payload(sample_invalid)


# ============================================================================
# CHECK 2 REGRESSION TESTS: Synthetic Data Exclusion from Release Metrics
# ============================================================================

def test_check2_mixed_dataset_synthetic_does_not_inflate_real_gate_f1(lifecycle_manager):
    """
    PRD Check 2 Critical:
    In a mixed dataset, synthetic samples with 100% accuracy must NOT inflate
    the real-data release gate metrics.
    """
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    # Real sample where model makes a mistake on key field (PAN) -> Real F1 is low!
    real_sample = build_evaluation_sample("SMP_REAL_01", DUMMY_HASH_1, is_synthetic=False)

    # Synthetic sample where model gets 100% accuracy!
    synth_sample = build_evaluation_sample("SMP_SYNTH_01", DUMMY_HASH_2, is_synthetic=True)

    dataset = EvaluationDataset(
        dataset_id="ds_mixed_inflate",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[real_sample, synth_sample],
    )

    payload_real = create_sample_payload("doc_real", DUMMY_HASH_1)
    payload_synth = create_sample_payload("doc_synth", DUMMY_HASH_2)
    resolver = InMemoryPayloadResolver({
        real_sample.sample_id: payload_real,
        synth_sample.sample_id: payload_synth,
    })

    def mock_caller(p, schema_cls):
        if p.document_id == "doc_real":
            # Key error: wrong PAN
            return {
                "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "WRONG9999X", "address": {"city": "BENGALURU"}},
                "services": {"opt_in_online": True},
                "billing": {"total_amount": 8000.0},
                "required_missing_field": "DEFAULT_VAL",
            }, "req_real"
        else:
            # Synthetic: perfect match!
            return {
                "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
                "services": {"opt_in_online": True},
                "billing": {"total_amount": 8000.0},
                "required_missing_field": "DEFAULT_VAL",
            }, "req_synth"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    report = runner.run_benchmark(dataset)

    # Real metrics reflect ONLY real samples
    assert report.real_samples == 1
    assert report.synthetic_samples == 1
    assert report.total_samples == 2

    # In real sample, PAN F1 is 0.0, applicant.full_name F1 is 1.0 -> key macro F1 = 0.50
    assert report.aggregate_metrics.macro_f1_key_fields == 0.50
    assert report.metrics_decision.overall_pass is False  # Real metrics fail Gate threshold (≥95%)!

    # Diagnostic synthetic metrics reflect synthetic samples separately
    assert report.synthetic_aggregate_metrics is not None
    assert report.synthetic_aggregate_metrics.macro_f1_key_fields == 1.0


def test_check2_mixed_dataset_synthetic_does_not_degrade_real_gate_f1(lifecycle_manager):
    """
    PRD Check 2 Critical:
    In a mixed dataset, synthetic samples with poor predictions must NOT degrade
    the real-data release gate metrics.
    """
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    # Real sample where model gets 100% accuracy -> Real F1 is 1.0!
    real_sample = build_evaluation_sample("SMP_REAL_PERFECT", DUMMY_HASH_1, is_synthetic=False)

    # Synthetic sample where model fails completely -> Synthetic F1 is 0.0!
    synth_sample = build_evaluation_sample("SMP_SYNTH_FAIL", DUMMY_HASH_2, is_synthetic=True)

    dataset = EvaluationDataset(
        dataset_id="ds_mixed_degrade",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[real_sample, synth_sample],
    )

    payload_real = create_sample_payload("doc_real", DUMMY_HASH_1)
    payload_synth = create_sample_payload("doc_synth", DUMMY_HASH_2)
    resolver = InMemoryPayloadResolver({
        real_sample.sample_id: payload_real,
        synth_sample.sample_id: payload_synth,
    })

    def mock_caller(p, schema_cls):
        if p.document_id == "doc_real":
            # Real: perfect match!
            return {
                "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
                "services": {"opt_in_online": True},
                "billing": {"total_amount": 8000.0},
                "required_missing_field": "DEFAULT_VAL",
            }, "req_real"
        else:
            # Synthetic: completely wrong values
            return {
                "applicant": {"full_name": "WRONG_NAME", "pan_number": "WRONG1234K", "address": {"city": "UNKNOWN"}},
                "services": {"opt_in_online": False},
                "billing": {"total_amount": 1.0},
                "required_missing_field": "WRONG",
            }, "req_synth"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    report = runner.run_benchmark(dataset)

    # Real metrics remain 1.0 despite synthetic complete failure
    assert report.real_samples == 1
    assert report.synthetic_samples == 1
    assert report.aggregate_metrics.macro_f1_all_fields == 1.0
    assert report.aggregate_metrics.macro_f1_key_fields == 1.0
    assert report.metrics_decision.overall_pass is True

    # Synthetic diagnostic aggregate metrics reflect poor performance
    assert report.synthetic_aggregate_metrics.macro_f1_key_fields == 0.0
    assert report.synthetic_aggregate_metrics.macro_f1_all_fields < 0.20
    assert report.synthetic_aggregate_metrics.macro_f1_all_fields < report.aggregate_metrics.macro_f1_all_fields


# ============================================================================
# CHECK 3 REGRESSION TESTS: Trusted Locked-Test Authorization
# ============================================================================

def test_check3_caller_cannot_self_assert_permissions_or_roles():
    """
    PRD Check 3 Critical:
    Caller cannot bypass authorization by self-asserting permissions or roles.
    AuthorizationContext forbids caller-supplied permissions or roles.
    """
    with pytest.raises(ValidationError):
        # Extra fields forbidden by ConfigDict(extra="forbid")
        AuthorizationContext(
            principal_id="attacker",
            permissions=["EXECUTE_LOCKED_TEST"],  # type: ignore
        )

    with pytest.raises(ValidationError):
        AuthorizationContext(
            principal_id="attacker",
            roles=["GATE_APPROVER"],  # type: ignore
        )


def test_check3_unregistered_principal_rejected():
    """Verify that an unregistered principal fails closed immediately."""
    authorizer = DefaultLockedSetAuthorizer()
    context = AuthorizationContext(principal_id="unregistered_user")

    decision = authorizer.authorize(
        context=context,
        tenant_id="tenant_alpha",
        dataset_id="ds_lock_01",
        split=DatasetSplit.LOCKED_TEST,
    )
    assert decision.is_authorized is False
    assert "not recognized in the trusted authorization boundary" in decision.reason


def test_check3_tenant_scope_enforcement():
    """Verify that principal cannot access a tenant outside their authorized scope."""
    authorizer = DefaultLockedSetAuthorizer()
    authorizer.register_principal(
        TrustedPrincipal(
            principal_id="tenant_a_auditor",
            roles=["GATE_APPROVER"],
            permissions=["EXECUTE_LOCKED_TEST"],
            tenant_scopes=["tenant_alpha"],  # Only tenant_alpha
            dataset_scopes=["*"],
        )
    )

    context = AuthorizationContext(principal_id="tenant_a_auditor")

    # Allowed on tenant_alpha
    decision_ok = authorizer.authorize(
        context=context,
        tenant_id="tenant_alpha",
        dataset_id="ds_lock_01",
        split=DatasetSplit.LOCKED_TEST,
    )
    assert decision_ok.is_authorized is True

    # Denied on tenant_beta (fail closed)
    decision_denied = authorizer.authorize(
        context=context,
        tenant_id="tenant_beta",
        dataset_id="ds_lock_01",
        split=DatasetSplit.LOCKED_TEST,
    )
    assert decision_denied.is_authorized is False
    assert "not authorized for tenant 'tenant_beta'" in decision_denied.reason


def test_check3_dataset_scope_enforcement():
    """Verify that principal cannot access a dataset outside their authorized scope."""
    authorizer = DefaultLockedSetAuthorizer()
    authorizer.register_principal(
        TrustedPrincipal(
            principal_id="ds1_auditor",
            roles=["GATE_APPROVER"],
            permissions=["EXECUTE_LOCKED_TEST"],
            tenant_scopes=["*"],
            dataset_scopes=["ds_lock_01"],  # Only ds_lock_01
        )
    )

    context = AuthorizationContext(principal_id="ds1_auditor")

    # Allowed on ds_lock_01
    decision_ok = authorizer.authorize(
        context=context,
        tenant_id="tenant_alpha",
        dataset_id="ds_lock_01",
        split=DatasetSplit.LOCKED_TEST,
    )
    assert decision_ok.is_authorized is True

    # Denied on ds_lock_02
    decision_denied = authorizer.authorize(
        context=context,
        tenant_id="tenant_alpha",
        dataset_id="ds_lock_02",
        split=DatasetSplit.LOCKED_TEST,
    )
    assert decision_denied.is_authorized is False
    assert "not authorized for dataset 'ds_lock_02'" in decision_denied.reason


def test_check3_cryptographic_hmac_token_verification():
    """Verify cryptographic HMAC token verification, tampering rejection, and scope enforcement."""
    secret = "super_secure_vault_secret_key_32bytes"
    authorizer = DefaultLockedSetAuthorizer(secret_key=secret)
    authorizer.register_principal(
        TrustedPrincipal(
            principal_id="crypto_auditor",
            roles=["GATE_APPROVER"],
            permissions=["EXECUTE_LOCKED_TEST"],
            tenant_scopes=["*"],
            dataset_scopes=["*"],
        )
    )

    # 1. Valid token succeeds
    valid_token = authorizer.mint_token("crypto_auditor", "tenant_alpha", "ds_lock_01")
    context_valid = AuthorizationContext(
        principal_id="crypto_auditor",
        auth_token=valid_token,
    )
    decision_valid = authorizer.authorize(
        context=context_valid,
        tenant_id="tenant_alpha",
        dataset_id="ds_lock_01",
        split=DatasetSplit.LOCKED_TEST,
    )
    assert decision_valid.is_authorized is True

    # 2. Tampered token fails
    tampered_token = valid_token[:-4] + "ffff"
    context_tampered = AuthorizationContext(
        principal_id="crypto_auditor",
        auth_token=tampered_token,
    )
    decision_tampered = authorizer.authorize(
        context=context_tampered,
        tenant_id="tenant_alpha",
        dataset_id="ds_lock_01",
        split=DatasetSplit.LOCKED_TEST,
    )
    assert decision_tampered.is_authorized is False
    assert "Invalid cryptographic signature" in decision_tampered.reason

    # 3. Token scoped to different tenant fails
    wrong_tenant_token = authorizer.mint_token("crypto_auditor", "tenant_other", "ds_lock_01")
    context_wrong_tenant = AuthorizationContext(
        principal_id="crypto_auditor",
        auth_token=wrong_tenant_token,
    )
    decision_wrong_tenant = authorizer.authorize(
        context=context_wrong_tenant,
        tenant_id="tenant_alpha",
        dataset_id="ds_lock_01",
        split=DatasetSplit.LOCKED_TEST,
    )
    assert decision_wrong_tenant.is_authorized is False
    assert "does not match requested tenant" in decision_wrong_tenant.reason


# ============================================================================
# CHECK 4 REGRESSION TESTS: PRD GR-005 Schema Conformance Gate
# ============================================================================

def test_check4_gr005_999_successful_unique_runs_incomplete(lifecycle_manager, trusted_authorizer):
    """
    PRD GR-005 Requirement: 999 successful unique runs means GR-005 is incomplete.
    Incomplete GR-005 must prevent final release-gate PASS.
    """
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    # Build 999 unique samples and payloads
    samples = [
        build_evaluation_sample(f"SMP_999_{i:04d}", DUMMY_HASH_1, split=DatasetSplit.LOCKED_TEST)
        for i in range(999)
    ]
    payloads = {
        s.sample_id: create_sample_payload(doc_id=f"doc_999_{i:04d}", doc_hash=DUMMY_HASH_1)
        for i, s in enumerate(samples)
    }
    resolver = InMemoryPayloadResolver(payloads)

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, f"req_{p.document_id}"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
        authorizer=trusted_authorizer,
        min_locked_samples=100,
    )

    dataset = EvaluationDataset(
        dataset_id="ds_lock_999",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=samples,
    )

    auth = AuthorizationContext(principal_id="gate_auditor_bob", event_reason="TEST_999_RUNS")
    report = runner.run_benchmark(dataset, auth_context=auth)

    # Metrics compliance passed, but GR-005 is incomplete!
    assert report.metrics_decision.overall_pass is True
    acc = report.gr005_accounting
    assert acc.status == "INCOMPLETE"
    assert acc.total_attempted_runs == 999
    assert acc.unique_qualifying_runs == 999
    assert acc.validation_exceptions == 0
    assert acc.is_conformance_verified is False
    assert "remaining 1 runs" in acc.limitation_note

    # Incomplete GR-005 prevents release gate pass
    assert report.gate_eligible is False
    assert report.overall_pass is False
    assert any("GR005_INCOMPLETE" in r for r in report.gate_eligibility_reasons)


def test_check4_gr005_1000_successful_unique_runs_verified(lifecycle_manager, trusted_authorizer):
    """
    PRD GR-005 Requirement: 1,000 successful unique runs with zero exceptions means GR-005 is verified.
    Release-gate qualification passes.
    """
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    # Build 1000 unique samples and payloads
    samples = [
        build_evaluation_sample(f"SMP_1000_{i:04d}", DUMMY_HASH_1, split=DatasetSplit.LOCKED_TEST)
        for i in range(1000)
    ]
    payloads = {
        s.sample_id: create_sample_payload(doc_id=f"doc_1000_{i:04d}", doc_hash=DUMMY_HASH_1)
        for i, s in enumerate(samples)
    }
    resolver = InMemoryPayloadResolver(payloads)

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, f"req_{p.document_id}"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
        authorizer=trusted_authorizer,
        min_locked_samples=100,
    )

    dataset = EvaluationDataset(
        dataset_id="ds_lock_1000",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=samples,
    )

    auth = AuthorizationContext(principal_id="gate_auditor_bob", event_reason="TEST_1000_RUNS")
    report = runner.run_benchmark(dataset, auth_context=auth)

    acc = report.gr005_accounting
    assert acc.status == "VERIFIED"
    assert acc.total_attempted_runs == 1000
    assert acc.unique_qualifying_runs == 1000
    assert acc.successful_validations == 1000
    assert acc.validation_exceptions == 0
    assert acc.is_conformance_verified is True
    assert "PRD GR-005 VERIFIED" in acc.limitation_note

    # Both metrics and release gate PASS
    assert report.metrics_decision.overall_pass is True
    assert report.gate_eligible is True
    assert report.overall_pass is True
    assert any("GR005_VERIFIED" in r for r in report.gate_eligibility_reasons)


def test_check4_gr005_1000_runs_with_one_schema_validation_exception_fails(lifecycle_manager, trusted_authorizer):
    """
    PRD GR-005 Requirement: 1,000 runs with 1 schema-validation exception means GR-005 fails.
    Failed GR-005 prevents final release-gate PASS regardless of metrics.
    """
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    samples = [
        build_evaluation_sample(f"SMP_EXC_{i:04d}", DUMMY_HASH_1, split=DatasetSplit.LOCKED_TEST)
        for i in range(1000)
    ]
    payloads = {
        s.sample_id: create_sample_payload(doc_id=f"doc_exc_{i:04d}", doc_hash=DUMMY_HASH_1)
        for i, s in enumerate(samples)
    }
    resolver = InMemoryPayloadResolver(payloads)

    # Sample index 500 triggers a schema validation failure / exception
    def mock_caller_with_single_failure(p, schema_cls):
        if p.document_id == "doc_exc_0500":
            raise ValueError("Schema validation exception on sample doc_exc_0500")
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, f"req_{p.document_id}"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller_with_single_failure)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
        authorizer=trusted_authorizer,
        min_locked_samples=100,
    )

    dataset = EvaluationDataset(
        dataset_id="ds_lock_exc",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=samples,
    )

    auth = AuthorizationContext(principal_id="gate_auditor_bob", event_reason="TEST_EXC_RUN")
    report = runner.run_benchmark(dataset, auth_context=auth)

    acc = report.gr005_accounting
    assert acc.status == "FAILED"
    assert acc.total_attempted_runs == 1000
    assert acc.unique_qualifying_runs == 999
    assert acc.validation_exceptions == 1
    assert acc.is_conformance_verified is False
    assert "PRD GR-005 FAILED" in acc.limitation_note

    # Failed GR-005 prevents release gate PASS
    assert report.gate_eligible is False
    assert report.overall_pass is False
    assert any("GR005_FAILED" in r for r in report.gate_eligibility_reasons)


def test_check4_gr005_duplicate_execution_ids_not_counted_twice(lifecycle_manager):
    """
    PRD GR-005 Requirement: Duplicate execution IDs must not increase the qualifying run count.
    """
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    # 10 samples, but all share only 2 distinct payload run_ids: 'duplicate_run_A' and 'duplicate_run_B'
    samples = [
        build_evaluation_sample(f"SMP_DUP_{i:02d}", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
        for i in range(10)
    ]
    payloads = {}
    for i, s in enumerate(samples):
        dup_run_id = "duplicate_run_A" if i < 5 else "duplicate_run_B"
        payload = create_sample_payload(doc_id=f"doc_dup_{i:02d}", doc_hash=DUMMY_HASH_1)
        payload = payload.model_copy(update={"run_id": dup_run_id})
        payloads[s.sample_id] = payload

    resolver = InMemoryPayloadResolver(payloads)

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, f"req_{p.document_id}"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    dataset = EvaluationDataset(
        dataset_id="ds_dup",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=samples,
    )

    report = runner.run_benchmark(dataset)

    acc = report.gr005_accounting
    assert acc.total_attempted_runs == 10
    # Exactly 2 unique run IDs were executed!
    assert acc.unique_qualifying_runs == 2
    assert acc.successful_validations == 2
    assert sorted(acc.conformance_run_ids) == ["duplicate_run_A", "duplicate_run_B"]
    assert acc.status == "INCOMPLETE"


def test_check4_gr005_sample_count_greater_than_execution_count_remains_incomplete(lifecycle_manager):
    """
    PRD GR-005 Requirement: Dataset sample count must never be substituted for execution run count.
    A dataset with 1,000 samples where only 500 unique executions occur remains incomplete.
    """
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    # 1,000 samples in dataset, but payloads only have 500 unique run IDs (each run_id repeated twice)
    samples = [
        build_evaluation_sample(f"SMP_SAMP_500_{i:04d}", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
        for i in range(1000)
    ]
    payloads = {}
    for i, s in enumerate(samples):
        # 500 unique run IDs
        run_idx = i % 500
        payload = create_sample_payload(doc_id=f"doc_s500_{i:04d}", doc_hash=DUMMY_HASH_1)
        payload = payload.model_copy(update={"run_id": f"run_unique_{run_idx:03d}"})
        payloads[s.sample_id] = payload

    resolver = InMemoryPayloadResolver(payloads)

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, f"req_{p.document_id}"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    dataset = EvaluationDataset(
        dataset_id="ds_samp_vs_exec",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=samples,
    )

    report = runner.run_benchmark(dataset)

    assert report.total_samples == 1000
    acc = report.gr005_accounting
    assert acc.total_attempted_runs == 1000
    assert acc.unique_qualifying_runs == 500  # NOT 1000!
    assert acc.status == "INCOMPLETE"
    assert acc.is_conformance_verified is False
    assert report.gate_eligible is False


def test_check4_gr005_incomplete_prevents_release_gate_pass(lifecycle_manager, trusted_authorizer):
    """
    PRD GR-005 Requirement: Incomplete GR-005 must prevent final release-gate PASS.
    """
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    # 50 samples on locked_test (well below 1,000 runs)
    samples = [
        build_evaluation_sample(f"SMP_INCOMP_{i:02d}", DUMMY_HASH_1, split=DatasetSplit.LOCKED_TEST)
        for i in range(50)
    ]
    payloads = {
        s.sample_id: create_sample_payload(doc_id=f"doc_incomp_{i:02d}", doc_hash=DUMMY_HASH_1)
        for i, s in enumerate(samples)
    }
    resolver = InMemoryPayloadResolver(payloads)

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, f"req_{p.document_id}"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
        authorizer=trusted_authorizer,
        min_locked_samples=10,
    )

    dataset = EvaluationDataset(
        dataset_id="ds_lock_incomp",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=samples,
    )

    auth = AuthorizationContext(principal_id="gate_auditor_bob", event_reason="TEST_INCOMPLETE")
    report = runner.run_benchmark(dataset, auth_context=auth)

    # Metrics pass
    assert report.metrics_decision.overall_pass is True
    # GR-005 incomplete
    assert report.gr005_accounting.status == "INCOMPLETE"
    assert report.gr005_accounting.unique_qualifying_runs == 50
    # Incomplete GR-005 blocks release gate
    assert report.gate_eligible is False
    assert report.overall_pass is False


def test_check4_gr005_failed_prevents_release_gate_pass(lifecycle_manager, trusted_authorizer):
    """
    PRD GR-005 Requirement: Failed GR-005 must prevent final release-gate PASS.
    """
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample("SMP_FAIL_GATE", DUMMY_HASH_1, split=DatasetSplit.LOCKED_TEST)
    payload = create_sample_payload("doc_fail_gate", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    def mock_failing_caller(p, schema_cls):
        raise ValueError("Schema validation crash on decoding")

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_failing_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(
        lifecycle_manager=lifecycle_manager,
        adapter=adapter,
        payload_resolver=resolver,
        authorizer=trusted_authorizer,
        min_locked_samples=1,
    )

    dataset = EvaluationDataset(
        dataset_id="ds_lock_failed",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=[sample],
    )

    auth = AuthorizationContext(principal_id="gate_auditor_bob", event_reason="TEST_FAILED")
    report = runner.run_benchmark(dataset, auth_context=auth)

    assert report.gr005_accounting.status == "FAILED"
    assert report.gate_eligible is False
    assert report.overall_pass is False


def test_check4_gr005_report_serialization_retains_accounting_and_status(lifecycle_manager):
    """
    PRD GR-005 Requirement: Report serialization retains complete GR-005 accounting and status.
    """
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_test_schema", 1, SchemaStatus.TESTING, actor="alice")

    sample = build_evaluation_sample("SMP_SER_GR", DUMMY_HASH_1, split=DatasetSplit.DEVELOPMENT)
    payload = create_sample_payload("doc_ser_gr", DUMMY_HASH_1)
    resolver = InMemoryPayloadResolver({sample.sample_id: payload})

    def mock_caller(p, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": "ABCDE1234F", "address": {"city": "BENGALURU"}},
            "services": {"opt_in_online": True},
            "billing": {"total_amount": 8000.0},
            "required_missing_field": "DEFAULT_VAL",
        }, "req_ser_gr"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)
    runner = TestHarnessRunner(lifecycle_manager=lifecycle_manager, adapter=adapter, payload_resolver=resolver)

    dataset = EvaluationDataset(
        dataset_id="ds_ser_gr",
        tenant_id="tenant_alpha",
        schema_id="kyc_test_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )

    report = runner.run_benchmark(dataset)

    json_str = report.to_json(indent=2)
    deserialized = BenchmarkReport.model_validate_json(json_str)

    acc = deserialized.gr005_accounting
    assert acc.status == "INCOMPLETE"
    assert acc.total_attempted_runs == 1
    assert acc.unique_qualifying_runs == 1
    assert acc.successful_validations == 1
    assert acc.validation_exceptions == 0
    assert acc.required_runs == 1000
    assert acc.is_conformance_verified is False
    assert acc.conformance_run_ids == ["run_doc_ser_gr"]
    assert "remaining 999 runs" in acc.limitation_note
