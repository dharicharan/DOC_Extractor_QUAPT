"""
Module B Pipeline Orchestration, Handoff Packaging, and Reviewer Overrides (Stage 4C).

Fulfills PRD:
- ARC-003, SCH-011, SCH-012: End-to-end Module B pipeline emitting validated ModuleBOutputPayload.
- FR-CLS-001..003: Integrated visual layout parsing, table/checkbox/grid reconstruction,
  classification, and schema binding.
- FR-CLS-004: Reviewer override foundation allowing verified document-type reassignment
  with full run-history lineage and prior-run preservation.
- GR-004: Fail-safe uncertainty routing — low-confidence or UNKNOWN documents route strictly
  to 'needs_review' without schema binding.
- API-013 / Idempotency integration: Repeated invocations return cached terminal payload.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from doc_extractor.classification.base import (
    ClassificationEngineError,
    DocumentClassificationService,
    InMemorySchemaRegistry,
    SchemaRegistry,
    SchemaResolutionError,
)
from doc_extractor.classification.rules import RuleBasedClassifierAdapter
from doc_extractor.contracts_b import (
    MODULE_B_CONTRACT_VERSION,
    ClassificationOutput,
    ConfidenceProvenance,
    ModuleBInputPayload,
    ModuleBOutputPayload,
    MVPClassificationDomain,
    MVPDocumentType,
    PageLayoutOutput,
)
from doc_extractor.exceptions import (
    DocumentExtractorError,
    DocumentProcessingError,
    StorageError,
)
from doc_extractor.handoff import ModuleAHandoffPayload
from doc_extractor.idempotency import IdempotencyStore, LocalIdempotencyStore
from doc_extractor.loader_b import LoadedPageImage, ModuleBLoadResult, SecureModuleBLoader
from doc_extractor.models import DocumentStatus, IdempotencyRecord
from doc_extractor.parsing.adapter import LocalOCRParserAdapter
from doc_extractor.parsing.base import ParserEngine, ParserEngineError
from doc_extractor.parsing.checkboxes import CandidateCheckboxRegion, CheckboxDetector
from doc_extractor.parsing.character_grids import CandidateCharacterGrid, CharacterGridReconstructor
from doc_extractor.parsing.tables import CandidateTableRegion, TableReconstructor
from doc_extractor.storage import StorageBackend


logger = logging.getLogger("doc_extractor.module_b")


@dataclass(frozen=True)
class ReviewerOverrideRequest:
    """Authorized request by a QC/human reviewer to override document type (FR-CLS-004)."""
    document_id: str
    tenant_id: str
    reviewer_id: str
    new_document_type: MVPDocumentType
    override_reason: str
    target_schema_version: Optional[str] = None  # None resolves default active approved version


class ModuleBPipeline:
    """
    Unified end-to-end coordinator for Module B: Visual Layout Parsing & Document Classification.
    """

    def __init__(
        self,
        storage: StorageBackend,
        loader: Optional[SecureModuleBLoader] = None,
        parser: Optional[ParserEngine] = None,
        classification_service: Optional[DocumentClassificationService] = None,
        schema_registry: Optional[SchemaRegistry] = None,
        idempotency_store: Optional[IdempotencyStore] = None,
        tenant_id: str = "default_tenant",
    ) -> None:
        self.storage = storage
        self.tenant_id = tenant_id
        self.loader = loader or SecureModuleBLoader(storage=self.storage)
        self.parser = parser or LocalOCRParserAdapter()
        self.schema_registry = schema_registry or InMemorySchemaRegistry()
        
        if classification_service is not None:
            self.classification_service = classification_service
        else:
            default_classifier = RuleBasedClassifierAdapter()
            self.classification_service = DocumentClassificationService(
                classifier=default_classifier,
                schema_registry=self.schema_registry,
            )

        self.idempotency_store = idempotency_store
        self.table_reconstructor = TableReconstructor()
        self.checkbox_detector = CheckboxDetector()
        self.grid_reconstructor = CharacterGridReconstructor()

    # ------------------------------------------------------------------------
    # Pipeline Execution
    # ------------------------------------------------------------------------

    def process_handoff(
        self,
        handoff_data: Union[ModuleAHandoffPayload, Dict[str, Any]],
        run_id: Optional[str] = None,
        candidate_tables_by_page: Optional[Dict[int, List[CandidateTableRegion]]] = None,
        candidate_checkboxes_by_page: Optional[Dict[int, List[CandidateCheckboxRegion]]] = None,
        candidate_grids_by_page: Optional[Dict[int, List[CandidateCharacterGrid]]] = None,
        idempotency_key: Optional[str] = None,
    ) -> ModuleBOutputPayload:
        """
        Executes end-to-end Module B pipeline from Module A handoff to ModuleBOutputPayload.
        
        Guarantees:
        - Checks idempotency store if configured and idempotency_key is provided.
        - Securely loads and verifies preprocessed page images (loader_b).
        - Parses layout per page and merges reconstructed tables, checkboxes, character grids.
        - Combines markdown with explicit page separators: '\n\n--- Page {N} ---\n\n'.
        - Executes classification and binds approved schema (Stage 4A/4B).
        - Packages result into ModuleBOutputPayload with correct status ('processing' vs 'needs_review').
        - Fails closed on any unexpected or infrastructure error, returning status='failed'.
        """
        start_time = time.perf_counter()
        created_at_iso = datetime.now(timezone.utc).isoformat()
        
        # 1. Inspect raw input for IDs to attach to failures
        raw_doc_id = "unknown_doc"
        raw_tenant_id = self.tenant_id
        raw_source_run_id = "unknown_run"

        if isinstance(handoff_data, ModuleAHandoffPayload):
            raw_doc_id = handoff_data.document_id
            raw_tenant_id = handoff_data.tenant_id
            raw_source_run_id = handoff_data.run_id
        elif isinstance(handoff_data, dict):
            raw_doc_id = handoff_data.get("document_id", raw_doc_id)
            raw_tenant_id = handoff_data.get("tenant_id", raw_tenant_id)
            raw_source_run_id = handoff_data.get("run_id", raw_source_run_id)

        active_run_id = run_id or f"run_b_{uuid.uuid4().hex[:12]}"

        # 2. Idempotency Check (API-013)
        if self.idempotency_store and idempotency_key:
            existing_record = self.idempotency_store.get_record(raw_tenant_id, idempotency_key)
            if existing_record and existing_record.response_payload:
                try:
                    cached_payload = ModuleBOutputPayload(**existing_record.response_payload)
                    logger.info("Idempotency cache hit for key %s (doc: %s)", idempotency_key, raw_doc_id)
                    return cached_payload
                except Exception:
                    pass  # If cached payload format is corrupt, recompute safely

        # 3. Main processing block wrapped in fail-closed safety
        try:
            # 3a. Secure Loading & Verification
            load_result = self.loader.load_handoff(
                handoff_data=handoff_data,
                expected_tenant_id=self.tenant_id if self.tenant_id != "default_tenant" else None,
            )
            # Update canonical identifiers from validated handoff
            canonical_doc_id = load_result.document_id
            canonical_tenant_id = load_result.tenant_id

            # 3b. Per-Page Visual Layout Parsing & Reconstruction
            parsed_pages: List[PageLayoutOutput] = []
            for loaded_page in load_result.pages:
                page_layout = self.parser.parse_page(loaded_page)
                page_layout.page_image_path = str(loaded_page.file_path)

                # Optional Table Structure Reconstruction (Stage 3A)
                if candidate_tables_by_page and loaded_page.page_number in candidate_tables_by_page:
                    cand_tables = candidate_tables_by_page[loaded_page.page_number]
                    reconstructed_tables = []
                    for cand_tbl in cand_tables:
                        html_tbl = self.table_reconstructor.reconstruct_table(cand_tbl, loaded_page)
                        reconstructed_tables.append(html_tbl)
                    page_layout.tables.extend(reconstructed_tables)

                # Optional Checkbox Detection (Stage 3B)
                if candidate_checkboxes_by_page and loaded_page.page_number in candidate_checkboxes_by_page:
                    cand_cbs = candidate_checkboxes_by_page[loaded_page.page_number]
                    detected_cbs = []
                    for cand_cb in cand_cbs:
                        cb_mark = self.checkbox_detector.process_candidate(cand_cb, loaded_page)
                        detected_cbs.append(cb_mark)
                    page_layout.checkboxes.extend(detected_cbs)

                # Optional Character Grid Reconstruction (Stage 3C)
                if candidate_grids_by_page and loaded_page.page_number in candidate_grids_by_page:
                    cand_grids = candidate_grids_by_page[loaded_page.page_number]
                    reconstructed_grids = []
                    for cand_grid in cand_grids:
                        grid_mark = self.grid_reconstructor.reconstruct_grid(cand_grid, loaded_page)
                        reconstructed_grids.append(grid_mark)
                    page_layout.character_grids.extend(reconstructed_grids)

                parsed_pages.append(page_layout)

            # 3c. Combined Markdown with Explicit Page Separators
            markdown_parts: List[str] = []
            for p in parsed_pages:
                header = f"--- Page {p.page_number} ---"
                markdown_parts.append(f"{header}\n\n{p.markdown_content}")
            combined_markdown = "\n\n".join(markdown_parts)

            # 3d. Document Classification & Approved Schema Binding (Stage 4A & 4B)
            classification = self.classification_service.classify_and_bind(
                load_result=load_result,
                parsed_pages=parsed_pages,
            )

            # Determine public status strictly aligned with review requirement
            status = "needs_review" if classification.review_required else "processing"

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            completed_at_iso = datetime.now(timezone.utc).isoformat()

            parser_meta = self.parser.get_metadata()
            classifier_meta = self.classification_service.classifier.get_metadata()

            payload = ModuleBOutputPayload(
                contract_version=MODULE_B_CONTRACT_VERSION,
                run_id=active_run_id,
                document_id=canonical_doc_id,
                tenant_id=canonical_tenant_id,
                status=status,
                classification=classification,
                pages=parsed_pages,
                combined_markdown=combined_markdown,
                parser_model_name=parser_meta.engine_name,
                classifier_model_name=classifier_meta.classifier_name,
                processing_time_ms=round(duration_ms, 2),
                created_at=created_at_iso,
                completed_at=completed_at_iso,
                error=None,
                extra_metadata={
                    "source_run_id": raw_source_run_id,
                    "page_count": len(parsed_pages),
                },
            )

            # Store in idempotency store if requested
            if self.idempotency_store and idempotency_key:
                now_ts = datetime.now(timezone.utc).timestamp()
                record = IdempotencyRecord(
                    idempotency_key=idempotency_key,
                    tenant_id=canonical_tenant_id,
                    request_fingerprint=hashlib.sha256(canonical_doc_id.encode()).hexdigest(),
                    document_id=canonical_doc_id,
                    response_payload=payload.model_dump(),
                    created_at_utc=now_ts,
                    expires_at_utc=now_ts + 86400.0,
                )
                self.idempotency_store.save_record(record)

            return payload

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            completed_at_iso = datetime.now(timezone.utc).isoformat()
            logger.error("Module B pipeline processing failed for doc %s: %s", raw_doc_id, str(exc))

            error_envelope = {
                "code": getattr(exc, "code", "MODULE_B_PIPELINE_ERROR"),
                "message": str(exc),
                "details": getattr(exc, "details", {}),
            }

            # Return terminal failed payload adhering to SCH-011
            fallback_classification = ClassificationOutput(
                domain=MVPClassificationDomain.UNKNOWN,
                document_type=MVPDocumentType.UNKNOWN,
                classification_confidence=0.0,
                confidence_provenance=ConfidenceProvenance.UNCALIBRATED,
                schema_id=None,
                schema_version=None,
                review_required=True,
                classifier_identity="pipeline-fallback",
                classification_reasons=["Pipeline failed prior to successful classification."],
            )

            parser_meta_name = getattr(getattr(self, "parser", None), "engine_name", "unknown_parser")
            classifier_meta_name = "unknown_classifier"

            return ModuleBOutputPayload(
                contract_version=MODULE_B_CONTRACT_VERSION,
                run_id=active_run_id,
                document_id=raw_doc_id,
                tenant_id=raw_tenant_id,
                status="failed",
                classification=fallback_classification,
                pages=[],
                combined_markdown="",
                parser_model_name=parser_meta_name,
                classifier_model_name=classifier_meta_name,
                processing_time_ms=round(duration_ms, 2),
                created_at=created_at_iso,
                completed_at=completed_at_iso,
                error=error_envelope,
                extra_metadata={"failed": True},
            )

    # ------------------------------------------------------------------------
    # Workstream 3: Reviewer Override Handler (FR-CLS-004)
    # ------------------------------------------------------------------------

    def apply_reviewer_override(
        self,
        original_payload: ModuleBOutputPayload,
        override_request: ReviewerOverrideRequest,
        new_run_id: Optional[str] = None,
    ) -> ModuleBOutputPayload:
        """
        Processes human reviewer override of document type (FR-CLS-004).
        
        Guarantees:
        - Preserves prior run payload immutably (never modifies original_payload).
        - Enforces tenant isolation (caller context must match document payload).
        - Validates reviewer identity and override reason (cannot be blank).
        - Validates new document type and resolves approved schema reference from registry.
        - Emits a fresh ModuleBOutputPayload with new run_id, updated classification,
          bound schema, review_required=False, and complete audit metadata.
        """
        # 1. Enforce tenant isolation
        if override_request.tenant_id != original_payload.tenant_id:
            raise StorageError(
                message="Tenant isolation violation: override request tenant does not match payload.",
                details={
                    "request_tenant": override_request.tenant_id,
                    "document_tenant": original_payload.tenant_id,
                },
            )

        if override_request.document_id != original_payload.document_id:
            raise DocumentProcessingError(
                message="Document ID mismatch in reviewer override request.",
                code="OVERRIDE_DOC_ID_MISMATCH",
            )

        # 2. Validate reviewer identity and reason
        if not override_request.reviewer_id.strip():
            raise DocumentProcessingError(
                message="Reviewer ID is required for document type override (FR-CLS-004).",
                code="MISSING_REVIEWER_ID",
            )
        if not override_request.override_reason.strip():
            raise DocumentProcessingError(
                message="Override reason is required for document type override audit trail.",
                code="MISSING_OVERRIDE_REASON",
            )

        # 3. Resolve target approved schema
        if override_request.new_document_type == MVPDocumentType.UNKNOWN:
            # Reaffirming UNKNOWN keeps review required and binds no schema
            new_classification = ClassificationOutput(
                domain=MVPClassificationDomain.UNKNOWN,
                document_type=MVPDocumentType.UNKNOWN,
                classification_confidence=1.0,
                confidence_provenance=ConfidenceProvenance.RULE_HEURISTIC,
                schema_id=None,
                schema_version=None,
                review_required=True,
                classifier_identity=f"reviewer_override:{override_request.reviewer_id}",
                classification_reasons=[
                    f"Human reviewer {override_request.reviewer_id} set type to UNKNOWN: {override_request.override_reason}"
                ],
            )
            new_status = "needs_review"
        else:
            # Resolve approved schema for recognized type
            schema_ref = self.schema_registry.resolve_approved_schema(
                document_type=override_request.new_document_type,
                tenant_id=override_request.tenant_id,
            )

            # Map domain from document type
            if override_request.new_document_type in (MVPDocumentType.DOC_01_CASA, MVPDocumentType.DOC_02_INSURANCE_APP):
                domain = MVPClassificationDomain.BFSI
            elif override_request.new_document_type == MVPDocumentType.DOC_03_DISCHARGE_SUMMARY:
                domain = MVPClassificationDomain.HEALTHCARE
            else:
                domain = MVPClassificationDomain.UNKNOWN

            new_classification = ClassificationOutput(
                domain=domain,
                document_type=override_request.new_document_type,
                classification_confidence=1.0,  # Human verified
                confidence_provenance=ConfidenceProvenance.RULE_HEURISTIC,
                schema_id=schema_ref.schema_id,
                schema_version=override_request.target_schema_version or schema_ref.schema_version,
                review_required=False,
                classifier_identity=f"reviewer_override:{override_request.reviewer_id}",
                classification_reasons=[
                    f"Human reviewer {override_request.reviewer_id} overrode type to {override_request.new_document_type.value}: {override_request.override_reason}"
                ],
            )
            new_status = "processing"

        target_run_id = new_run_id or f"run_b_override_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        # Emit new versioned run payload
        new_payload = ModuleBOutputPayload(
            contract_version=MODULE_B_CONTRACT_VERSION,
            run_id=target_run_id,
            document_id=original_payload.document_id,
            tenant_id=original_payload.tenant_id,
            status=new_status,
            classification=new_classification,
            pages=list(original_payload.pages),  # Reuses layout parsing faithfully
            combined_markdown=original_payload.combined_markdown,
            parser_model_name=original_payload.parser_model_name,
            classifier_model_name=f"reviewer_override:{override_request.reviewer_id}",
            processing_time_ms=0.0,
            created_at=now_iso,
            completed_at=now_iso,
            error=None,
            extra_metadata={
                "prior_run_id": original_payload.run_id,
                "prior_document_type": original_payload.classification.document_type.value,
                "reviewer_id": override_request.reviewer_id,
                "override_reason": override_request.override_reason,
            },
        )

        logger.info(
            "Reviewer %s overrode doc %s from %s to %s (prior run: %s, new run: %s)",
            override_request.reviewer_id,
            original_payload.document_id,
            original_payload.classification.document_type.value,
            override_request.new_document_type.value,
            original_payload.run_id,
            target_run_id,
        )

        return new_payload
