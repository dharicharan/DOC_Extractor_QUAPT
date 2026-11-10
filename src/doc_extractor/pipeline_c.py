"""
Module C Canonical Envelope and Pipeline Orchestrator — 1-Stage Multimodal.
Adheres strictly to PRD §6.3 Canonical Output Envelope, FR-EXT-001..007,
FR-VAL-001..004, FR-CNF-001..003, GR-002, GR-003, GR-027.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel

from doc_extractor.contracts_b import ModuleBOutputPayload
from doc_extractor.contracts_c import (
    CanonicalExtractionEnvelope,
    CanonicalOutputEnvelope,
    DocumentMetadata,
    EvidenceSnippet,
    EvidenceModel,
    ExecutionMode,
    ExecutionProvenance,
    ExtractedField,
    FieldOutput,
    FlagItem,
    RunMetadata,
)
from doc_extractor.extractor import SchemaGuidedExtractor, SCHEMA_MAP, find_evidence_for_value
from doc_extractor.field_assembly import (
    assemble_canonical_envelope,
    assemble_field_entry,
)
from doc_extractor.handoff import CoordinateTransform, PageHandoffRecord
from doc_extractor.validation_rules import (
    normalize_and_format,
    normalize_date_to_iso,
    normalize_phone_number,
    normalize_currency_amount,
    validate_pan,
    validate_aadhaar,
    validate_ifsc,
    validate_chronology,
    validate_arithmetic,
    calculate_calibrated_confidence,
)


class ModuleCPipeline:
    """
    Module C Pipeline Orchestrator.
    Takes ModuleBOutputPayload, extracts structured fields according to schema,
    performs evidence grounding (FR-EXT-004, GR-002), normalizes and validates values (FR-VAL-001..004),
    calculates calibrated confidences (FR-CNF-001), enforces null-if-absent (FR-EXT-003, GR-003),
    applies document-level status gating (FR-CNF-003, GR-027), and emits CanonicalExtractionEnvelope (SCH-011..013).
    """

    def __init__(self, extractor: Optional[SchemaGuidedExtractor] = None):
        self.extractor = extractor or SchemaGuidedExtractor()

    def process(self, payload: ModuleBOutputPayload) -> CanonicalExtractionEnvelope:
        # Route UNKNOWN or low-confidence classification directly to needs_review (FR-CLS-002)
        if payload.status == "needs_review" or payload.classification.document_type.value == "UNKNOWN":
            return CanonicalExtractionEnvelope(
                document_id=payload.document_id,
                document_metadata=DocumentMetadata(
                    document_type=payload.classification.document_type.value,
                    schema_id=payload.classification.schema_id or "UNKNOWN",
                    schema_version=payload.classification.schema_version or "1.0.0",
                    classification_confidence=payload.classification.classification_confidence,
                    status="needs_review",
                    page_count=len(payload.pages),
                ),
                fields={},
                flags=[FlagItem(code="CLASSIFICATION_UNCERTAIN", message="Document routed to review prior to extraction", fields=[])],
                run=RunMetadata(
                    run_id=payload.run_id,
                    model_versions={"parser": payload.parser_model_name, "extractor": self.extractor.model_name},
                    prompt_version="1.0.0",
                ),
            )

        doc_type = payload.classification.document_type.value
        schema_cls = SCHEMA_MAP.get(doc_type)

        if not schema_cls:
            return CanonicalExtractionEnvelope(
                document_id=payload.document_id,
                document_metadata=DocumentMetadata(
                    document_type=doc_type,
                    schema_id=payload.classification.schema_id or "UNKNOWN",
                    schema_version=payload.classification.schema_version or "1.0.0",
                    classification_confidence=payload.classification.classification_confidence,
                    status="needs_review",
                    page_count=len(payload.pages),
                ),
                fields={},
                flags=[FlagItem(code="NO_SCHEMA_BOUND", message=f"No schema registered for document type {doc_type}", fields=[])],
                run=RunMetadata(
                    run_id=payload.run_id,
                    model_versions={"parser": payload.parser_model_name, "extractor": self.extractor.model_name},
                    prompt_version=self.extractor.prompt_version,
                ),
            )

        # Extract structured data using SchemaGuidedExtractor (multimodal or deterministic)
        if hasattr(self.extractor, "extract_with_provenance"):
            extraction_result = self.extractor.extract_with_provenance(payload, schema_cls=schema_cls)
            extracted_data = extraction_result.payload
            execution_provenance = extraction_result.provenance
        else:
            extracted_data = self.extractor.extract(payload, schema_cls=schema_cls)
            execution_provenance = getattr(self.extractor, "last_provenance", None)

        if execution_provenance is None:
            execution_provenance = ExecutionProvenance(
                mode=ExecutionMode.HEURISTIC,
                model_name=self.extractor.model_name,
                provider="local_extractor",
                prompt_version=self.extractor.prompt_version,
                fallback_occurred=False,
            )

        fields_envelope: Dict[str, ExtractedField] = {}
        flags: List[FlagItem] = []

        if doc_type == "DOC-01":
            self._process_doc_01(extracted_data, payload, fields_envelope, flags)
        elif doc_type == "DOC-02":
            self._process_doc_02(extracted_data, payload, fields_envelope, flags)
        elif doc_type == "DOC-03":
            self._process_doc_03(extracted_data, payload, fields_envelope, flags)

        return assemble_canonical_envelope(
            document_id=payload.document_id,
            document_type=payload.classification.document_type.value,
            schema_id=payload.classification.schema_id or doc_type,
            schema_version=payload.classification.schema_version or "1.0.0",
            classification_confidence=payload.classification.classification_confidence,
            page_count=len(payload.pages),
            fields=fields_envelope,
            flags=flags,
            run_id=payload.run_id,
            parser_model_name=payload.parser_model_name,
            extractor_model_name=getattr(self.extractor, "model_name", "unknown"),
            prompt_version=self.extractor.prompt_version,
            provenance=execution_provenance,
            payload_status=payload.status,
        )

    def _process_doc_01(
        self,
        extracted_data: Any,
        payload: ModuleBOutputPayload,
        fields_envelope: Dict[str, ExtractedField],
        flags: List[FlagItem],
    ) -> None:
        """Processes DOC-01 CASA Account Opening Form fields."""
        app = getattr(extracted_data, "applicant", None)
        acc = getattr(extracted_data, "account_details", None)

        # applicant.full_name (Required)
        name_val = getattr(app, "full_name", None) if app else None
        fields_envelope["applicant.full_name"] = self._create_field_entry(
            "applicant.full_name", name_val, payload, flags, required=True
        )

        # applicant.pan_card_number (Optional, format check FR-VAL-002)
        pan_val = getattr(app, "pan_card_number", None) if app else None
        norm_pan, pan_err = validate_pan(pan_val)
        fields_envelope["applicant.pan_card_number"] = self._create_field_entry(
            "applicant.pan_card_number", norm_pan, payload, flags, error=pan_err, required=False
        )

        # applicant.date_of_birth (Optional, ISO normalizer FR-VAL-001)
        dob_val = getattr(app, "date_of_birth", None) if app else None
        norm_dob, dob_err = normalize_date_to_iso(dob_val)
        fields_envelope["applicant.date_of_birth"] = self._create_field_entry(
            "applicant.date_of_birth", norm_dob, payload, flags, error=dob_err, required=False
        )

        # applicant.phone_number (Optional, E.164 normalizer FR-VAL-001)
        phone_val = getattr(app, "phone_number", None) if app else None
        norm_phone, phone_err = normalize_phone_number(phone_val)
        fields_envelope["applicant.phone_number"] = self._create_field_entry(
            "applicant.phone_number", norm_phone, payload, flags, error=phone_err, required=False
        )

        # account_details.account_type (Optional)
        acc_type = getattr(acc, "account_type", None) if acc else None
        fields_envelope["account_details.account_type"] = self._create_field_entry(
            "account_details.account_type", acc_type, payload, flags, required=False
        )

        # account_details.initial_deposit_amount (Optional, decimal currency normalizer)
        dep_val = getattr(acc, "initial_deposit_amount", None) if acc else None
        norm_amt, curr, amt_err = normalize_currency_amount(dep_val)
        fields_envelope["account_details.initial_deposit_amount"] = self._create_field_entry(
            "account_details.initial_deposit_amount", norm_amt, payload, flags, error=amt_err, required=False
        )

        # account_details.currency
        fields_envelope["account_details.currency"] = ExtractedField(
            value=curr,
            confidence=1.0,
            null_reason=None,
            review_required=False,
            validation="passed",
            evidence=None,
        )

        # Checkboxes (FR-EXT-007)
        opt_in = getattr(acc, "opt_in_services", None) if acc else None
        chq_val = getattr(opt_in, "cheque_book", None) if opt_in else None
        fields_envelope["account_details.opt_in_services.cheque_book"] = self._create_field_entry(
            "account_details.opt_in_services.cheque_book", chq_val, payload, flags, required=False
        )

        deb_val = getattr(opt_in, "debit_card", None) if opt_in else None
        fields_envelope["account_details.opt_in_services.debit_card"] = self._create_field_entry(
            "account_details.opt_in_services.debit_card", deb_val, payload, flags, required=False
        )

        net_val = getattr(opt_in, "internet_banking", None) if opt_in else None
        fields_envelope["account_details.opt_in_services.internet_banking"] = self._create_field_entry(
            "account_details.opt_in_services.internet_banking", net_val, payload, flags, required=False
        )

    def _process_doc_02(
        self,
        extracted_data: Any,
        payload: ModuleBOutputPayload,
        fields_envelope: Dict[str, ExtractedField],
        flags: List[FlagItem],
    ) -> None:
        """Processes DOC-02 Insurance Application Form fields."""
        p_name = getattr(extracted_data, "proposer_name", None)
        fields_envelope["proposer_name"] = self._create_field_entry(
            "proposer_name", p_name, payload, flags, required=True
        )

        sum_ins = getattr(extracted_data, "sum_insured", None)
        norm_sum, curr, sum_err = normalize_currency_amount(sum_ins)
        fields_envelope["sum_insured"] = self._create_field_entry(
            "sum_insured", norm_sum, payload, flags, error=sum_err, required=False
        )

        p_term = getattr(extracted_data, "policy_term_years", None)
        fields_envelope["policy_term_years"] = self._create_field_entry(
            "policy_term_years", p_term, payload, flags, required=False
        )

        # Checkbox medical declarations (FR-EXT-007)
        med = getattr(extracted_data, "medical_declaration", None)
        diab = getattr(med, "has_diabetes", None) if med else None
        fields_envelope["medical_declaration.has_diabetes"] = self._create_field_entry(
            "medical_declaration.has_diabetes", diab, payload, flags, required=False
        )

        hyp = getattr(med, "has_hypertension", None) if med else None
        fields_envelope["medical_declaration.has_hypertension"] = self._create_field_entry(
            "medical_declaration.has_hypertension", hyp, payload, flags, required=False
        )

        tob = getattr(med, "has_tobacco_use", None) if med else None
        fields_envelope["medical_declaration.has_tobacco_use"] = self._create_field_entry(
            "medical_declaration.has_tobacco_use", tob, payload, flags, required=False
        )

        # Repeating groups: Nominees (FR-EXT-005)
        nominees = getattr(extracted_data, "nominees", []) or []
        for idx, nom in enumerate(nominees):
            nom_dict = nom.model_dump() if hasattr(nom, "model_dump") else (nom if isinstance(nom, dict) else {})
            fields_envelope[f"nominees.{idx}.name"] = self._create_field_entry(
                f"nominees.{idx}.name", nom_dict.get("name"), payload, flags, required=False
            )
            fields_envelope[f"nominees.{idx}.relationship"] = self._create_field_entry(
                f"nominees.{idx}.relationship", nom_dict.get("relationship"), payload, flags, required=False
            )
            fields_envelope[f"nominees.{idx}.share_percentage"] = self._create_field_entry(
                f"nominees.{idx}.share_percentage", nom_dict.get("share_percentage"), payload, flags, required=False
            )

        # Check nominee share percentage sum if nominees present
        shares = [
            float(n.get("share_percentage", 0) if isinstance(n, dict) else getattr(n, "share_percentage", 0))
            for n in nominees
            if (n.get("share_percentage") if isinstance(n, dict) else getattr(n, "share_percentage", None)) is not None
        ]
        if shares and len(shares) > 1:
            arith_flag = validate_arithmetic(shares, 100.0)
            if arith_flag:
                flags.append(FlagItem(code="ARITHMETIC_MISMATCH", message="Nominee share percentages do not sum to 100%", fields=["nominees.share_percentage"]))

    def _process_doc_03(
        self,
        extracted_data: Any,
        payload: ModuleBOutputPayload,
        fields_envelope: Dict[str, ExtractedField],
        flags: List[FlagItem],
    ) -> None:
        """Processes DOC-03 Hospital Discharge Summary fields."""
        pat_name = getattr(extracted_data, "patient_name", None)
        fields_envelope["patient_name"] = self._create_field_entry(
            "patient_name", pat_name, payload, flags, required=True
        )

        ip_no = getattr(extracted_data, "ip_number", None)
        fields_envelope["ip_number"] = self._create_field_entry(
            "ip_number", ip_no, payload, flags, required=False
        )

        adm_date = getattr(extracted_data, "admission_date", None)
        norm_adm, adm_err = normalize_date_to_iso(adm_date)
        fields_envelope["admission_date"] = self._create_field_entry(
            "admission_date", norm_adm, payload, flags, error=adm_err, required=False
        )

        dis_date = getattr(extracted_data, "discharge_date", None)
        norm_dis, dis_err = normalize_date_to_iso(dis_date)
        fields_envelope["discharge_date"] = self._create_field_entry(
            "discharge_date", norm_dis, payload, flags, error=dis_err, required=False
        )

        diag = getattr(extracted_data, "diagnosis", None)
        fields_envelope["diagnosis"] = self._create_field_entry(
            "diagnosis", diag, payload, flags, required=False
        )

        # Chronology & Cross-Field Check (FR-VAL-004)
        chronology_flag = validate_chronology(norm_adm, norm_dis)
        if chronology_flag:
            flags.append(FlagItem(**chronology_flag))
            if "admission_date" in fields_envelope:
                fields_envelope["admission_date"].review_required = True
            if "discharge_date" in fields_envelope:
                fields_envelope["discharge_date"].review_required = True

        # Repeating group: Medications (FR-EXT-005)
        meds = getattr(extracted_data, "medications", []) or []
        for idx, m in enumerate(meds):
            m_dict = m.model_dump() if hasattr(m, "model_dump") else (m if isinstance(m, dict) else {})
            fields_envelope[f"medications.{idx}.medicine_name"] = self._create_field_entry(
                f"medications.{idx}.medicine_name", m_dict.get("medicine_name"), payload, flags, required=False
            )
            fields_envelope[f"medications.{idx}.dosage"] = self._create_field_entry(
                f"medications.{idx}.dosage", m_dict.get("dosage"), payload, flags, required=False
            )
            fields_envelope[f"medications.{idx}.frequency"] = self._create_field_entry(
                f"medications.{idx}.frequency", m_dict.get("frequency"), payload, flags, required=False
            )
            fields_envelope[f"medications.{idx}.duration"] = self._create_field_entry(
                f"medications.{idx}.duration", m_dict.get("duration"), payload, flags, required=False
            )

    def _create_field_entry(
        self,
        field_name: str,
        value: Any,
        payload: ModuleBOutputPayload,
        flags: List[FlagItem],
        error: Optional[str] = None,
        required: bool = False,
    ) -> ExtractedField:
        """
        Helper to create a validated ExtractedField with evidence grounding,
        null-if-absent logic (FR-EXT-003, GR-003), and calibrated confidence (FR-CNF-001).
        Delegates to shared assemble_field_entry service.
        """
        return assemble_field_entry(
            field_name=field_name,
            value=value,
            pages=payload.pages,
            flags=flags,
            error=error,
            required=required,
            strict_boolean_evidence=False,  # preserve existing static pipeline compatibility
        )
