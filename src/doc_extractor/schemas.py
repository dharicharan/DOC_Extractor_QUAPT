"""
Pydantic schemas and metadata for Module C extraction (DOC-01, DOC-02, DOC-03).
Strictly adheres to PRD §3.1, §6.1, §6.2, FR-EXT-001, FR-EXT-003, FR-EXT-005, FR-EXT-007.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# DOC-01: CASA Account Opening Form (BFSI)
# ============================================================================

class ApplicantDetails(BaseModel):
    full_name: Optional[str] = Field(
        default=None,
        description="Applicant full name, merged from character boxes, uppercase as printed (first + middle + last)",
    )
    first_name: Optional[str] = Field(default=None, description="First name row, if the form splits the name")
    middle_name: Optional[str] = Field(default=None, description="Middle name row, if the form splits the name")
    last_name: Optional[str] = Field(default=None, description="Last name / surname row, if the form splits the name")
    aadhaar_number: Optional[str] = Field(default=None, description="12-digit Aadhaar number, digits only")
    gross_annual_income: Optional[str] = Field(
        default=None, description="Ticked gross annual income range, as printed (e.g. '1-5 Lakhs')"
    )
    pan_card_number: Optional[str] = Field(
        default=None,
        description="10-character PAN number matching ^[A-Z]{5}[0-9]{4}[A-Z]$",
    )
    date_of_birth: Optional[str] = Field(
        default=None,
        description="Applicant date of birth, normalized to ISO 8601 (YYYY-MM-DD)",
    )
    phone_number: Optional[str] = Field(
        default=None,
        description="Applicant contact number, normalized to E.164 (+91...)",
    )


class AccountOptInServices(BaseModel):
    cheque_book: Optional[bool] = Field(
        default=None,
        description="Cheque book requested checkbox flag ([x]=True, [ ]=False)",
    )
    debit_card: Optional[bool] = Field(
        default=None,
        description="Debit card requested checkbox flag ([x]=True, [ ]=False)",
    )
    internet_banking: Optional[bool] = Field(
        default=None,
        description="Internet banking opt-in checkbox flag ([x]=True, [ ]=False)",
    )


class AccountDetails(BaseModel):
    account_type: Optional[str] = Field(
        default=None,
        description="Type of account (e.g. SAVINGS, CURRENT, SALARY)",
    )
    initial_deposit_amount: Optional[float] = Field(
        default=None,
        description="Initial deposit amount as numeric decimal",
    )
    currency: Optional[str] = Field(
        default="INR",
        description="ISO 4217 currency code",
    )
    opt_in_services: Optional[AccountOptInServices] = Field(
        default_factory=AccountOptInServices,
        description="Requested account services checkboxes",
    )


class BankDetails(BaseModel):
    account_number: Optional[str] = Field(default=None, description="Bank account number, kept as a string with leading zeros")
    bank_name: Optional[str] = Field(default=None, description="Bank name as written")
    branch_address: Optional[str] = Field(default=None, description="Branch name / address as written")
    city: Optional[str] = Field(default=None, description="City / town / village")
    state: Optional[str] = Field(default=None, description="State")
    pin_code: Optional[str] = Field(default=None, description="6-digit PIN code")
    country: Optional[str] = Field(default=None, description="Country")
    ifsc_code: Optional[str] = Field(default=None, description="11-character IFSC code")
    micr_code: Optional[str] = Field(default=None, description="9-digit MICR code")


class CASASchema(BaseModel):
    """DOC-01: CASA / account opening form extraction target schema."""
    applicant: ApplicantDetails = Field(default_factory=ApplicantDetails)
    account_details: AccountDetails = Field(default_factory=AccountDetails)
    bank_details: BankDetails = Field(default_factory=BankDetails)


# ============================================================================
# DOC-02: Insurance Application Form (BFSI)
# ============================================================================

class NomineeDetails(BaseModel):
    name: Optional[str] = Field(default=None, description="Nominee full name")
    relationship: Optional[str] = Field(default=None, description="Relationship with proposer/insured")
    age: Optional[int] = Field(default=None, description="Age of nominee in years")
    share_percentage: Optional[float] = Field(default=None, description="Nominee share percentage (0-100)")


class MedicalDeclaration(BaseModel):
    has_diabetes: Optional[bool] = Field(default=None, description="Diabetes history declared checkbox")
    has_hypertension: Optional[bool] = Field(default=None, description="Hypertension history declared checkbox")
    has_tobacco_use: Optional[bool] = Field(default=None, description="Tobacco / smoking use declared checkbox")


class InsuranceApplicationSchema(BaseModel):
    """DOC-02: Insurance Application Form extraction target schema."""
    proposer_name: Optional[str] = Field(default=None, description="Proposer / insured full name")
    policy_term_years: Optional[int] = Field(default=None, description="Policy term in years")
    sum_insured: Optional[float] = Field(default=None, description="Total sum insured / cover amount")
    nominees: List[NomineeDetails] = Field(default_factory=list, description="List of nominees")
    medical_declaration: MedicalDeclaration = Field(default_factory=MedicalDeclaration)


# ============================================================================
# DOC-03: Hospital Discharge Summary (Healthcare)
# ============================================================================

class MedicationItem(BaseModel):
    medicine_name: Optional[str] = Field(default=None, description="Name of medication/drug")
    dosage: Optional[str] = Field(default=None, description="Dosage quantity (e.g. 500mg)")
    frequency: Optional[str] = Field(default=None, description="Frequency (e.g. OD, BD, TDS, Twice daily)")
    duration: Optional[str] = Field(default=None, description="Duration in days or weeks")


class DischargeSummarySchema(BaseModel):
    """DOC-03: Hospital Discharge Summary extraction target schema."""
    patient_name: Optional[str] = Field(default=None, description="Full patient name")
    ip_number: Optional[str] = Field(default=None, description="In-patient admission number / record ID")
    admission_date: Optional[str] = Field(default=None, description="Admission date in ISO 8601 (YYYY-MM-DD)")
    discharge_date: Optional[str] = Field(default=None, description="Discharge date in ISO 8601 (YYYY-MM-DD)")
    diagnosis: Optional[str] = Field(default=None, description="Final or primary diagnosis text")
    medications: List[MedicationItem] = Field(default_factory=list, description="Prescribed discharge medications")
