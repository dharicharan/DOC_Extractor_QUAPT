"""
Domain-level Document Processing Service (Phase 2 Architecture).

Decouples the end-to-end document processing pipeline from Gradio and HTTP transport layers.
Orchestrates:
1. Module A: Ingestion, magic bytes, hashing, ClamAV malware check, 200 DPI rendering, image preprocessing.
2. Module B: Handoff loading, layout parsing, classification hints, reviewer override handling.
3. Category Routing: Document classification and specialist model routing per page.
4. Routed Extraction: Vision/OCR multimodal extraction per page with partial coverage resilience.
5. Module C: Normalization, bounding box coordinate transformation to 200 DPI original geometry,
   evidence completeness gating, schema validation, and canonical envelope assembly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import uuid

from PIL import Image

from doc_extractor.config import Settings, load_settings
from doc_extractor.contracts_b import MVPDocumentType, ModuleBOutputPayload
from doc_extractor.contracts_c import (
    CanonicalExtractionEnvelope,
    FlagItem,
    PageImageRef,
)
from doc_extractor.exceptions import (
    ComponentConfigurationError,
    ConfigurationError,
    CrossTenantStorageAccessError,
    DocumentExtractorError,
    DocumentProcessingError,
    InvalidTenantIdentifierError,
    StorageError,
    TenantMismatchError,
    TenantStorageIsolationError,
    TenantStorageUnavailableError,
)

TYPED_DOMAIN_EXCEPTIONS: Tuple[type, ...] = (
    CrossTenantStorageAccessError,
    TenantStorageUnavailableError,
    TenantStorageIsolationError,
    TenantMismatchError,
    InvalidTenantIdentifierError,
    ComponentConfigurationError,
    ConfigurationError,
)
from doc_extractor.extractor import SCHEMA_MAP, SchemaGuidedExtractor
from doc_extractor.handoff import ModuleAHandoffPayload
from doc_extractor.idempotency import IdempotencyStore, LocalIdempotencyStore
from doc_extractor.pipeline import ModuleAPipeline
from doc_extractor.pipeline_b import ModuleBPipeline, ReviewerOverrideRequest
from doc_extractor.pipeline_c import ModuleCPipeline
from doc_extractor.routed_pages import RoutedPageResult
from doc_extractor.routing import (
    DEFAULT_MODEL,
    LANGUAGES,
    DocumentCategory,
    RoutedExtractor,
    build_generic_prompt,
    heuristic_classify,
)
from doc_extractor.storage import LocalDiskStorage, StorageBackend
from doc_extractor.tenancy import require_path_safe_tenant_id

AUTO = "Auto-Detect"

logger = logging.getLogger("doc_extractor.processing_service")

ENGINE_CHOICES: Dict[str, Optional[str]] = {
    "Follow routing table": None,
    "GLM-OCR + Qwen3-4B-Instruct text (experimental)": "ocr_text",
    "GLM-OCR + Qwen2.5-7B text (experimental, comparison)": "ocr_text_7b",
    "Full page text (OCR as JSON, no field picking)": "page_text",
}


def resolve_engine_choice(engine_choice: Optional[str]) -> Optional[str]:
    """
    Resolves the requested engine choice to an engine key or None (routing table default).
    - None, "", "Follow routing table", "vlm" -> None
    - Explicit valid choices in ENGINE_CHOICES (e.g. "GLM-OCR + Qwen3-4B-Instruct text (experimental)") -> mapped value
    - Explicitly supported alternative engine keys ("ocr_text", "ocr_text_7b") -> returned as-is
    - Any unrecognized engine choice falls back safely to None (default routing table) with a warning log,
      never raising an unhandled ValueError("unknown engine ...") (F5).
    """
    if not engine_choice or engine_choice in ("Follow routing table", "vlm"):
        return None
    if engine_choice in ENGINE_CHOICES:
        return ENGINE_CHOICES[engine_choice]
    if engine_choice in ("ocr_text", "ocr_text_7b", "page_text"):
        return engine_choice
    logger.warning("Unrecognized engine choice %r; falling back to default routing table.", engine_choice)
    return None


def resolve_claimed_filename(file_input: Any, claimed_filename: Optional[str] = None) -> str:
    """
    Extracts the display/claimed filename from the request parameter or file object.
    Falls back to 'uploaded_document' when no name can be derived (shared helper).
    """
    if claimed_filename:
        return claimed_filename
    if isinstance(file_input, (str, Path)):
        return Path(file_input).name
    if hasattr(file_input, "name") and file_input.name:
        return Path(str(file_input.name)).name
    return "uploaded_document"


def _safe_diagnostic_sink(
    sink: Optional[Callable[[str, str], None]],
    stage: str,
    message: str,
) -> None:
    """
    Safely dispatches diagnostic messages to an in-process diagnostic_sink callback.
    Guarantees that a failing or misbehaving sink cannot abort document processing,
    mask original failures, or recursively invoke itself.
    """
    if sink is None:
        return
    try:
        sink(stage, message)
    except Exception as sink_exc:
        logger.error(
            "diagnostic_sink callback failed for stage %s: %s",
            stage,
            sink_exc,
            exc_info=True,
        )


def _safe_rotation(value: Any) -> int:
    """Normalize integer rotation degrees to standard orthogonal rotations (90, 180, 270) or 0."""
    try:
        r = int(value) % 360
    except (TypeError, ValueError):
        return 0
    return r if r in (90, 180, 270) else 0


def _rotate_page(
    path: Path,
    clockwise_deg: int,
    workspace: RequestTemporaryWorkspace,
    page_no: int = 1,
) -> Path:
    """
    Turn a page image upright (clockwise degrees) and save a copy securely inside an active workspace.
    Requires an explicit RequestTemporaryWorkspace instance (mode 0o700, files 0o600).
    Disallows unmanaged rotation without an active workspace.
    """
    if not isinstance(workspace, RequestTemporaryWorkspace):
        raise TypeError("A valid RequestTemporaryWorkspace instance is required to rotate pages.")
    if not workspace.has_path:
        raise RuntimeError("The provided RequestTemporaryWorkspace is not active.")

    deg = _safe_rotation(clockwise_deg)
    target_dir = workspace.path

    unique_token = uuid.uuid4().hex[:12]
    safe_name = Path(path).name
    out = target_dir / f"upright_{deg}_p{page_no}_{unique_token}_{safe_name}"

    with Image.open(path) as im:
        im.rotate(-deg, expand=True).save(out)

    try:
        os.chmod(out, 0o600)
    except OSError:
        pass

    return out


class RequestTemporaryWorkspace:
    """
    Private temporary workspace for a single document-processing request (F2 remediation).
    Guarantees:
    - Unique private directory per request with restrictive mode 0o700.
    - Generated rotated files created with mode 0o600 and UUID collision prevention.
    - Deterministic cleanup upon request completion, generator termination, or failure.
    """

    def __init__(self, prefix: str = "doc_workspace_") -> None:
        self._prefix = prefix
        self._temp_dir: Optional[Path] = None

    def __enter__(self) -> "RequestTemporaryWorkspace":
        raw = tempfile.mkdtemp(prefix=self._prefix)
        self._temp_dir = Path(raw)
        try:
            os.chmod(self._temp_dir, 0o700)
        except OSError:
            pass
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    @property
    def has_path(self) -> bool:
        return self._temp_dir is not None

    @property
    def path(self) -> Path:
        if self._temp_dir is None:
            raise RuntimeError("Temporary workspace is not active.")
        return self._temp_dir

    def rotate_page(self, path: Path, clockwise_deg: int, page_no: int = 1) -> Path:
        """Rotate a page image and persist it inside this private workspace."""
        if self._temp_dir is None:
            raise RuntimeError("Temporary workspace is not active.")
        return _rotate_page(path, clockwise_deg, workspace=self, page_no=page_no)

    def cleanup(self) -> None:
        """Removes the entire private temporary directory and all generated files."""
        if self._temp_dir is not None:
            dir_to_remove = self._temp_dir
            self._temp_dir = None
            if dir_to_remove.exists():
                try:
                    shutil.rmtree(dir_to_remove)
                except Exception as exc:
                    logger.error("Failed to clean up temporary workspace %s: %s", dir_to_remove, exc)


def _validate_tenant_string(value: Any, component_desc: str, expected_tenant: str) -> None:
    """
    Validates that value is a valid, non-empty string matching expected_tenant.
    Rejects missing, None, int, bytes, empty, or whitespace-only tenant IDs.
    Fails closed with sanitized error message (no tenant ID disclosure).
    Raises ComponentConfigurationError on server misconfiguration.
    """
    if not isinstance(value, str) or not value.strip():
        raise ComponentConfigurationError(f"Tenant mismatch: {component_desc} tenant ID is missing or invalid.")
    if value != expected_tenant:
        raise ComponentConfigurationError(f"Tenant mismatch: {component_desc} belongs to another tenant.")


def _validate_component_tenant(component: Any, expected_tenant: str, component_name: str) -> None:
    """
    Verifies that an injected component and its nested dependencies belong to expected_tenant (F1).
    Fails closed with a sanitized error message (no tenant ID disclosure).
    Raises ComponentConfigurationError on server misconfiguration.
    """
    if component is None:
        return

    if component_name == "storage":
        storage_tid = getattr(component, "tenant_id", None)
        _validate_tenant_string(storage_tid, "injected storage", expected_tenant)
        return

    if component_name == "module_a":
        # 1. Direct tenant_id on module_a
        comp_tid = getattr(component, "tenant_id", None)
        _validate_tenant_string(comp_tid, "injected module_a", expected_tenant)

        # 2. Canonical module_a.storage
        pipe_storage = getattr(component, "storage", None)
        if pipe_storage is None:
            raise ComponentConfigurationError("Tenant mismatch: module_a storage backend is missing.")
        storage_tid = getattr(pipe_storage, "tenant_id", None)
        _validate_tenant_string(storage_tid, "module_a storage backend", expected_tenant)

        # 3. Ingestion service inside module_a
        ingestion = getattr(component, "ingestion_service", None)
        if ingestion is not None:
            ing_tid = getattr(ingestion, "tenant_id", None)
            _validate_tenant_string(ing_tid, "module_a ingestion service", expected_tenant)
            ing_storage = getattr(ingestion, "storage", None)
            if ing_storage is None:
                raise ComponentConfigurationError("Tenant mismatch: module_a ingestion storage is missing.")
            ing_storage_tid = getattr(ing_storage, "tenant_id", None)
            _validate_tenant_string(ing_storage_tid, "module_a ingestion storage", expected_tenant)
            if ing_storage is not pipe_storage:
                raise ComponentConfigurationError("Tenant mismatch: module_a ingestion storage does not match module_a storage.")
        elif isinstance(component, ModuleAPipeline):
            raise ComponentConfigurationError("Tenant mismatch: module_a ingestion service is missing.")

        # 4. Renderer inside module_a
        renderer = getattr(component, "renderer", None)
        if renderer is not None:
            rend_storage = getattr(renderer, "storage", None)
            if rend_storage is None:
                raise ComponentConfigurationError("Tenant mismatch: module_a renderer storage is missing.")
            rend_storage_tid = getattr(rend_storage, "tenant_id", None)
            _validate_tenant_string(rend_storage_tid, "module_a renderer storage", expected_tenant)
            if rend_storage is not pipe_storage:
                raise ComponentConfigurationError("Tenant mismatch: module_a renderer storage does not match module_a storage.")
        elif isinstance(component, ModuleAPipeline):
            raise ComponentConfigurationError("Tenant mismatch: module_a renderer is missing.")

        # 5. Preprocessor inside module_a
        preprocessor = getattr(component, "preprocessor", None)
        if preprocessor is not None:
            prep_storage = getattr(preprocessor, "storage", None)
            if prep_storage is None:
                raise ComponentConfigurationError("Tenant mismatch: module_a preprocessor storage is missing.")
            prep_storage_tid = getattr(prep_storage, "tenant_id", None)
            _validate_tenant_string(prep_storage_tid, "module_a preprocessor storage", expected_tenant)
            if prep_storage is not pipe_storage:
                raise ComponentConfigurationError("Tenant mismatch: module_a preprocessor storage does not match module_a storage.")
        elif isinstance(component, ModuleAPipeline):
            raise ComponentConfigurationError("Tenant mismatch: module_a preprocessor is missing.")

        return

    if component_name == "module_b":
        # 1. Direct tenant_id on module_b
        comp_tid = getattr(component, "tenant_id", None)
        _validate_tenant_string(comp_tid, "injected module_b", expected_tenant)

        # 2. module_b.storage (if present)
        b_storage = getattr(component, "storage", None)
        if b_storage is not None:
            b_storage_tid = getattr(b_storage, "tenant_id", None)
            _validate_tenant_string(b_storage_tid, "module_b storage backend", expected_tenant)

        # 3. Loader inside module_b
        loader = getattr(component, "loader", None)
        if loader is not None:
            loader_storage = getattr(loader, "storage", None)
            if loader_storage is None:
                raise ComponentConfigurationError("Tenant mismatch: module_b loader storage is missing.")
            loader_storage_tid = getattr(loader_storage, "tenant_id", None)
            _validate_tenant_string(loader_storage_tid, "module_b loader storage", expected_tenant)
            if b_storage is not None and loader_storage is not b_storage:
                raise ComponentConfigurationError("Tenant mismatch: module_b loader storage does not match module_b storage.")
        elif isinstance(component, ModuleBPipeline):
            raise ComponentConfigurationError("Tenant mismatch: module_b loader is missing.")

        return

    # Generic fallback
    comp_tid = getattr(component, "tenant_id", None)
    if comp_tid is not None:
        _validate_tenant_string(comp_tid, component_name, expected_tenant)


@dataclass(frozen=True)
class DocumentProcessingRequest:
    """
    Transport-neutral request for document processing.
    Contains explicit tenant and actor boundaries with processing options.
    """
    file_path_or_stream: Union[str, Path, BinaryIO]
    tenant_id: str
    claimed_filename: Optional[str] = None
    actor_id: Optional[str] = None
    category_choice: Optional[str] = None
    doc_type_choice: Optional[str] = None
    subtype_choice: Optional[str] = None
    language_choice: Optional[str] = None
    engine_choice: Optional[str] = None
    max_pages: Optional[int] = None
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str):
            raise InvalidTenantIdentifierError("not_a_string")
        if not self.tenant_id:
            raise InvalidTenantIdentifierError("empty")
        if not self.tenant_id.strip():
            raise InvalidTenantIdentifierError("whitespace_only")
        require_path_safe_tenant_id(self.tenant_id)


@dataclass
class DocumentProcessingResult:
    """
    Structured outcome of a complete or terminal document processing execution.
    """
    document_id: str
    tenant_id: str
    status: str
    envelope: Optional[CanonicalExtractionEnvelope] = None
    payload_a: Optional[ModuleAHandoffPayload] = None
    payload_b: Optional[ModuleBOutputPayload] = None
    page_results: List[RoutedPageResult] = field(default_factory=list)
    pages_info: List[Dict[str, Any]] = field(default_factory=list)
    page_items: List[Tuple[int, Path, Optional[Path]]] = field(default_factory=list)
    routes: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    execution_time_seconds: float = 0.0

    @property
    def is_success(self) -> bool:
        return self.status in ("auto_approved", "needs_review", "reviewed", "processing") and self.envelope is not None

    @property
    def is_rejected(self) -> bool:
        return self.status == "rejected"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"


@dataclass(frozen=True)
class ProcessingProgressEvent:
    """
    Progress event emitted during pipeline stage transitions.
    Enables live status updates in Gradio and monitoring in API/background execution.
    """
    step_key: str  # "upload", "split", "cls", "ext{n}", "merge"
    label: str
    state: str     # "pending", "running", "done", "failed"
    detail: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    result: Optional[DocumentProcessingResult] = None


class DocumentProcessingService:
    """
    Shared domain orchestrator for document processing.
    Reusable by both FastAPI REST endpoints and Gradio UI testing without duplication.
    """

    def __init__(
        self,
        module_a: Optional[ModuleAPipeline] = None,
        module_b: Optional[ModuleBPipeline] = None,
        module_c: Optional[ModuleCPipeline] = None,
        routed_extractor: Optional[RoutedExtractor] = None,
        storage: Optional[StorageBackend] = None,
        idempotency_store: Optional[IdempotencyStore] = None,
        settings: Optional[Settings] = None,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> None:
        self.settings = settings if settings is not None else load_settings()
        if tenant_id is not None:
            if not isinstance(tenant_id, str):
                raise InvalidTenantIdentifierError("not_a_string")
            if not tenant_id:
                raise InvalidTenantIdentifierError("empty")
            if not tenant_id.strip():
                raise InvalidTenantIdentifierError("whitespace_only")
            require_path_safe_tenant_id(tenant_id)
        self.tenant_id = tenant_id
        self.actor_id = actor_id

        # F1: Fail-closed tenant validation of injected components if service is bound to a tenant
        if self.tenant_id is not None:
            _validate_component_tenant(storage, self.tenant_id, "storage")
            _validate_component_tenant(module_a, self.tenant_id, "module_a")
            _validate_component_tenant(module_b, self.tenant_id, "module_b")

        self.storage = storage
        if self.storage is None and self.tenant_id is not None:
            self.storage = LocalDiskStorage(self.settings.storage_base_dir, tenant_id=self.tenant_id)

        self.idempotency_store = idempotency_store or LocalIdempotencyStore(self.settings.storage_base_dir)

        self.module_a = module_a
        if self.module_a is None and self.tenant_id is not None:
            self.module_a = ModuleAPipeline(
                settings=self.settings,
                storage=self.storage,
                idempotency_store=self.idempotency_store,
                tenant_id=self.tenant_id,
            )

        self.module_b = module_b
        if self.module_b is None and self.tenant_id is not None:
            self.module_b = ModuleBPipeline(
                storage=self.storage,
                tenant_id=self.tenant_id,
                settings=self.settings,
                idempotency_store=self.idempotency_store,
            )

        self.module_c = module_c if module_c is not None else ModuleCPipeline(
            extractor=SchemaGuidedExtractor(use_llm=False),
            settings=self.settings,
        )

        self.routed_extractor = routed_extractor if routed_extractor is not None else RoutedExtractor(allow_fallback=False)

        self.max_pages_default = int(os.environ.get("DOC_MAX_PAGES", "0"))

    def _resolve_tenant(self, request: Union[DocumentProcessingRequest, str, None] = None) -> str:
        """
        Resolves and validates the effective tenant ID for this processing execution.
        Fails closed if the tenant is missing, blank, invalid, or mismatches the service's bound tenant.
        """
        if isinstance(request, str):
            req_tenant = request
        else:
            req_tenant = getattr(request, "tenant_id", None) if request is not None else None

        if req_tenant is None:
            if self.tenant_id is not None:
                req_tenant = self.tenant_id
            else:
                raise ValueError("Tenant ID is required and cannot be empty.")

        if not isinstance(req_tenant, str):
            raise InvalidTenantIdentifierError("not_a_string")
        if not req_tenant:
            raise InvalidTenantIdentifierError("empty")
        if not req_tenant.strip():
            raise InvalidTenantIdentifierError("whitespace_only")
        require_path_safe_tenant_id(req_tenant)

        if self.tenant_id is not None and req_tenant != self.tenant_id:
            raise TenantMismatchError(
                "Tenant mismatch: request tenant does not match service tenant."
            )

        return req_tenant

    def resolve_page(
        self,
        img_path: Path,
        filename: str,
        rule_type: Optional[str],
        rule_confidence: Optional[float],
        choices: Dict[str, str],
        page_no: int,
    ) -> Dict[str, Any]:
        """
        Category, type, sub-type and language for one page: user choice wins, else auto-detect (FR-CLS-005).
        """
        user_cat = bool(choices.get("category")) and str(choices["category"]).strip().lower() != AUTO.lower()
        user_type = bool(choices.get("doc_type")) and str(choices["doc_type"]).strip().lower() != AUTO.lower()
        user_lang = bool(choices.get("language")) and str(choices["language"]).strip().lower() != AUTO.lower()
        user_sub = bool(choices.get("subtype")) and str(choices["subtype"]).strip().lower() != AUTO.lower()

        auto: Dict[str, Any] = {}
        if not (user_cat and user_type and user_lang):
            hint_type = rule_type if page_no == 1 else None
            hint_conf = rule_confidence if page_no == 1 else None
            if img_path and img_path.exists():
                auto = self.routed_extractor.classify(str(img_path), filename, hint_type, hint_conf)
            else:
                auto = heuristic_classify(filename, hint_type, hint_conf)

        category = DocumentCategory(choices["category"].lower() if user_cat else auto.get("category", "structured"))
        suggested = auto.get("document_type") or (rule_type if page_no == 1 else None)
        doc_type = choices["doc_type"] if user_type else None
        if doc_type not in SCHEMA_MAP or category != DocumentCategory.STRUCTURED:
            doc_type = None  # fixed schema applies to structured category only
        if suggested == doc_type:
            suggested = None
        language = LANGUAGES[choices["language"]] if user_lang else str(auto.get("language") or "en")
        subtype = choices["subtype"] if user_sub else auto.get("subtype")
        if category != DocumentCategory.UNSTRUCTURED or subtype in (None, "null"):
            subtype = None

        return {
            "page": page_no,
            "category": category.value,
            "category_source": "user" if user_cat else "auto",
            "document_type": doc_type or "generic",
            "schema_id": doc_type,
            "suggested_document_type": suggested,
            "type_source": "user" if doc_type else "auto",
            "subtype": subtype,
            "language": language,
            "language_source": "user" if user_lang else "auto",
            "confidence": None if user_cat else auto.get("confidence"),
            "method": "user" if user_cat else auto.get("method", "-"),
            "rotation": _safe_rotation(auto.get("rotation")),
            "router_model": auto.get("router_model"),
            "filled_by": auto.get("filled_by"),
        }

    def process_document_stream(
        self,
        request: DocumentProcessingRequest,
        diagnostic_sink: Optional[Callable[[str, str], None]] = None,
    ) -> Iterator[ProcessingProgressEvent]:
        """
        Executes document processing and yields ProcessingProgressEvents at stage boundaries.
        The terminal event carries the complete DocumentProcessingResult in its `result` attribute.
        """
        t_start = time.perf_counter()

        if request is None:
            raise ValueError("DocumentProcessingRequest is required.")

        file_input = request.file_path_or_stream
        if file_input is None:
            raise ValueError("No file provided for processing.")

        # Single consolidated claimed filename resolution
        claimed_name = resolve_claimed_filename(file_input, request.claimed_filename)

        effective_tenant = self._resolve_tenant(request)
        effective_actor = request.actor_id or self.actor_id or "system"

        # F1: Fail-closed tenant validation of injected components against request tenant
        _validate_component_tenant(self.storage, effective_tenant, "storage")
        _validate_component_tenant(self.module_a, effective_tenant, "module_a")
        _validate_component_tenant(self.module_b, effective_tenant, "module_b")

        exec_storage = (
            self.storage
            if self.storage is not None
            else LocalDiskStorage(self.settings.storage_base_dir, tenant_id=effective_tenant)
        )
        exec_module_a = (
            self.module_a
            if self.module_a is not None
            else ModuleAPipeline(
                settings=self.settings,
                storage=exec_storage,
                idempotency_store=self.idempotency_store,
                tenant_id=effective_tenant,
            )
        )
        exec_module_b = (
            self.module_b
            if self.module_b is not None
            else ModuleBPipeline(
                storage=exec_storage,
                tenant_id=effective_tenant,
                settings=self.settings,
                idempotency_store=self.idempotency_store,
            )
        )
        _validate_component_tenant(exec_storage, effective_tenant, "storage")
        _validate_component_tenant(exec_module_a, effective_tenant, "module_a")
        _validate_component_tenant(exec_module_b, effective_tenant, "module_b")

        # F2: Private temporary workspace for per-request isolated rotated artifacts
        workspace = RequestTemporaryWorkspace(prefix="doc_workspace_")
        workspace.__enter__()
        try:
            choices = {
                "category": request.category_choice or AUTO,
                "doc_type": request.doc_type_choice or AUTO,
                "subtype": request.subtype_choice or AUTO,
                "language": request.language_choice or AUTO,
            }

            # Resolve engine safely per F5
            engine = resolve_engine_choice(request.engine_choice)

            max_pages_limit = request.max_pages if request.max_pages is not None else self.max_pages_default

            # Event: Upload received
            yield ProcessingProgressEvent(
                step_key="upload",
                label="Upload received",
                state="done",
                detail=claimed_name,
                data={"claimed_name": claimed_name},
            )

            # Event: Split running
            yield ProcessingProgressEvent(
                step_key="split",
                label="Split into page images (200 DPI) and clean up",
                state="running",
            )

            # Step 1: Module A Ingestion & Preprocessing
            try:
                payload_a = exec_module_a.process_document(
                    file_input,
                    claimed_filename=claimed_name,
                    request_id=request.request_id,
                    idempotency_key=request.idempotency_key,
                )
            except TYPED_DOMAIN_EXCEPTIONS:
                raise
            except Exception as exc:
                logger.error("Module A processing error: %s", exc, exc_info=True)
                sanitized_msg = "Document ingestion and preprocessing failed."
                res = DocumentProcessingResult(
                    document_id="unknown_doc",
                    tenant_id=effective_tenant,
                    status="failed",
                    error={"code": "INGESTION_ERROR", "message": sanitized_msg},
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "split", str(exc))
                yield ProcessingProgressEvent(
                    step_key="split",
                    label="Split into page images (200 DPI) and clean up",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            # Validate Module A output structure per finding 3B
            if not isinstance(payload_a, ModuleAHandoffPayload):
                logger.error("Module A returned invalid payload: %r", type(payload_a))
                sanitized_msg = "Document ingestion failed: invalid payload produced."
                res = DocumentProcessingResult(
                    document_id="unknown_doc",
                    tenant_id=effective_tenant,
                    status="failed",
                    error={"code": "INGESTION_ERROR", "message": sanitized_msg},
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "split", f"Module A returned invalid payload: {type(payload_a).__name__}")
                yield ProcessingProgressEvent(
                    step_key="split",
                    label="Split into page images (200 DPI) and clean up",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            if payload_a.status == "rejected":
                err_dict = payload_a.error if isinstance(payload_a.error, dict) else {"code": "REJECTED", "message": "File rejected."}
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="rejected",
                    payload_a=payload_a,
                    error=err_dict,
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "split", f"Rejected: {err_dict}")
                yield ProcessingProgressEvent(
                    step_key="split",
                    label="Split into page images (200 DPI) and clean up",
                    state="failed",
                    detail=str(err_dict.get("message", "File rejected")),
                    data={"payload_a": payload_a.to_dict(), "error": err_dict},
                    result=res,
                )
                return

            if payload_a.status == "failed":
                raw_err = payload_a.error
                raw_msg = f"Module A failed: {raw_err}" if raw_err else "Module A ingestion failed."
                logger.error("Module A ingestion failed: %s", raw_msg)
                _safe_diagnostic_sink(diagnostic_sink, "split", raw_msg)

                sanitized_msg = "Document ingestion and preprocessing failed."
                sanitized_a_error = {
                    "code": "INGESTION_ERROR",
                    "message": sanitized_msg,
                }
                sanitized_payload_a = replace(payload_a, error=sanitized_a_error)
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=sanitized_payload_a,
                    error=sanitized_a_error,
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                yield ProcessingProgressEvent(
                    step_key="split",
                    label="Split into page images (200 DPI) and clean up",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_a_error},
                    result=res,
                )
                return

            # Resolve page items from storage
            try:
                page_items: List[Tuple[int, Path, Optional[Path]]] = []
                for n, rec in enumerate(payload_a.pages, start=1):
                    prep = exec_module_a.storage.resolve_uri_to_path(rec.preprocessed_storage_uri)
                    orig = exec_module_a.storage.resolve_uri_to_path(rec.original_storage_uri)
                    page_items.append((getattr(rec, "page_number", n), prep if prep.exists() else orig, orig))

                if not page_items:
                    if isinstance(file_input, (str, Path)):
                        page_items = [(1, Path(file_input), None)]
                    else:
                        page_items = [(1, Path(claimed_name), None)]

                total_pages = len(page_items)
                if max_pages_limit > 0:
                    page_items = page_items[:max_pages_limit]

                split_detail = f"{total_pages} page image(s)" + (
                    f", reading the first {len(page_items)}" if total_pages > len(page_items) else ""
                )
            except TYPED_DOMAIN_EXCEPTIONS:
                raise
            except Exception as exc:
                logger.error("Storage URI resolution error: %s", exc, exc_info=True)
                sanitized_msg = "Failed to resolve storage URIs for document pages."
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    error={"code": "STORAGE_RESOLUTION_FAILED", "message": sanitized_msg},
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "split", str(exc))
                yield ProcessingProgressEvent(
                    step_key="split",
                    label="Split into page images (200 DPI) and clean up",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            yield ProcessingProgressEvent(
                step_key="split",
                label="Split into page images (200 DPI) and clean up",
                state="done",
                detail=split_detail,
                data={"total_pages": total_pages, "page_items": page_items},
            )

            # Step 2: Module B hand-off
            try:
                page_refs = {ref.page_number: ref for ref in exec_module_b.load_page_images(payload_a)}
                payload_b = exec_module_b.process_handoff(payload_a)
            except TYPED_DOMAIN_EXCEPTIONS:
                raise
            except Exception as exc:
                logger.error("Module B processing error: %s", exc, exc_info=True)
                sanitized_msg = "Module B processing failed."
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    error={"code": "MODULE_B_ERROR", "message": sanitized_msg},
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "cls", str(exc))
                yield ProcessingProgressEvent(
                    step_key="cls",
                    label="Classify document from page 1 (router model, once)",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            if not isinstance(payload_b, ModuleBOutputPayload):
                logger.error("Module B returned invalid payload: %r", type(payload_b))
                sanitized_msg = "Module B processing failed: invalid payload produced."
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    error={"code": "MODULE_B_ERROR", "message": sanitized_msg},
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "cls", f"Module B returned invalid payload: {type(payload_b).__name__}")
                yield ProcessingProgressEvent(
                    step_key="cls",
                    label="Classify document from page 1 (router model, once)",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            if payload_b.status == "failed":
                raw_err = payload_b.error if isinstance(payload_b.error, dict) else {}
                raw_msg = f"Module B failed: {raw_err.get('code', 'UNKNOWN')}: {raw_err.get('message', 'no detail')}"
                logger.error("Module B processing failed: %s", raw_msg)
                _safe_diagnostic_sink(diagnostic_sink, "cls", raw_msg)

                sanitized_b_error = {
                    "code": "MODULE_B_ERROR",
                    "message": "Document page classification and layout analysis failed.",
                }
                sanitized_payload_b = payload_b.model_copy(update={"error": sanitized_b_error})
                res = DocumentProcessingResult(
                    document_id=payload_b.document_id,
                    tenant_id=payload_b.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    payload_b=sanitized_payload_b,
                    error=sanitized_b_error,
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                yield ProcessingProgressEvent(
                    step_key="cls",
                    label="Classify document from page 1 (router model, once)",
                    state="failed",
                    detail=sanitized_b_error["message"],
                    data={"error": sanitized_b_error},
                    result=res,
                )
                return

            module_b_type = payload_b.classification.document_type.value
            rule_type = module_b_type if module_b_type != "UNKNOWN" else None
            rule_confidence = payload_b.classification.classification_confidence if rule_type else None

            # Step 3: Classify document from page 1 (router model, once)
            yield ProcessingProgressEvent(
                step_key="cls",
                label="Classify document from page 1 (router model, once)",
                state="running",
                detail="detecting category, type and language",
            )

            try:
                doc_info = self.resolve_page(
                    page_items[0][1],
                    claimed_name,
                    rule_type,
                    rule_confidence,
                    choices,
                    1,
                )
            except TYPED_DOMAIN_EXCEPTIONS:
                raise
            except Exception as exc:
                logger.error("Classification/routing error: %s", exc, exc_info=True)
                sanitized_msg = "Document classification and routing failed."
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    payload_b=payload_b,
                    error={"code": "ROUTING_FAILED", "message": sanitized_msg},
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "cls", str(exc))
                yield ProcessingProgressEvent(
                    step_key="cls",
                    label="Classify document from page 1 (router model, once)",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            cls_detail = (
                f"{doc_info['category']} · {doc_info['document_type']} · {doc_info['language']} · {doc_info['category_source']}"
                + (f" · filled by {doc_info['filled_by']}" if doc_info.get("filled_by") else "")
                + (f" · {doc_info['router_model'].split('/')[-1]}" if doc_info.get("router_model") else "")
            )

            rotation = doc_info.get("rotation", 0)
            read_items = page_items
            if rotation:
                try:
                    read_items = [
                        (n, workspace.rotate_page(p, rotation, page_no=n), o)
                        for n, p, o in page_items
                    ]
                except TYPED_DOMAIN_EXCEPTIONS:
                    raise
                except Exception as exc:
                    logger.error("Page rotation error: %s", exc, exc_info=True)
                    sanitized_msg = "Document page rotation failed."
                    res = DocumentProcessingResult(
                        document_id=payload_a.document_id,
                        tenant_id=payload_a.tenant_id,
                        status="failed",
                        payload_a=payload_a,
                        payload_b=payload_b,
                        error={"code": "ROTATION_FAILED", "message": sanitized_msg},
                        execution_time_seconds=time.perf_counter() - t_start,
                    )
                    _safe_diagnostic_sink(diagnostic_sink, "cls", str(exc))
                    yield ProcessingProgressEvent(
                        step_key="cls",
                        label="Classify document from page 1 (router model, once)",
                        state="failed",
                        detail=sanitized_msg,
                        data={"error": sanitized_msg},
                        result=res,
                    )
                    return
                cls_detail += f" · rotated {rotation}°"

            yield ProcessingProgressEvent(
                step_key="cls",
                label="Classify document from page 1 (router model, once)",
                state="done",
                detail=cls_detail,
                data={"doc_info": doc_info, "rotation": rotation},
            )

            category = DocumentCategory(doc_info["category"])
            schema_cls = SCHEMA_MAP.get(doc_info["schema_id"]) if doc_info["schema_id"] else None
            doc_type = doc_info["schema_id"]

            if doc_type and payload_b.classification.document_type.value != doc_type:
                try:
                    payload_b = exec_module_b.apply_reviewer_override(
                        payload_b,
                        ReviewerOverrideRequest(
                            document_id=payload_b.document_id,
                            tenant_id=payload_b.tenant_id,
                            reviewer_id=effective_actor,
                            new_document_type=MVPDocumentType(doc_type),
                            override_reason="Document type explicitly selected in the extraction UI",
                        ),
                    )
                except TYPED_DOMAIN_EXCEPTIONS:
                    raise
                except Exception as exc:
                    logger.error("Reviewer override error: %s", exc, exc_info=True)
                    sanitized_msg = "Module B reviewer override failed."
                    res = DocumentProcessingResult(
                        document_id=payload_a.document_id,
                        tenant_id=payload_a.tenant_id,
                        status="failed",
                        payload_a=payload_a,
                        payload_b=payload_b,
                        error={"code": "MODULE_B_ERROR", "message": sanitized_msg},
                        execution_time_seconds=time.perf_counter() - t_start,
                    )
                    _safe_diagnostic_sink(diagnostic_sink, "cls", str(exc))
                    yield ProcessingProgressEvent(
                        step_key="cls",
                        label="Classify document from page 1 (router model, once)",
                        state="failed",
                        detail=sanitized_msg,
                        data={"error": sanitized_msg},
                        result=res,
                    )
                    return

            # Step 4: Per-page extraction with specialist model
            pages_info: List[Dict[str, Any]] = []
            page_results: List[RoutedPageResult] = []
            routes: List[Dict[str, Any]] = []

            for page_no, img_path, _orig in read_items:
                info = dict(doc_info, page=page_no)
                pages_info.append(info)

                yield ProcessingProgressEvent(
                    step_key=f"ext{page_no}",
                    label=f"Page {page_no}: extract fields",
                    state="running",
                    detail=f"routing to the {category.value} model",
                    data={
                        "page_no": page_no,
                        "read_count": len(read_items),
                        "category": category.value,
                        "language": info["language"],
                        "pages_info": list(pages_info),
                    },
                )

                try:
                    if not (img_path and img_path.exists()):
                        raise FileNotFoundError(f"Page {page_no} image not found.")
                    raw_dict, prov, route = self.routed_extractor.extract(
                        str(img_path), category, info["language"], schema_cls, info["subtype"], engine=engine
                    )
                    # Validate route metadata dictionary per finding 3A
                    route_dict = route if isinstance(route, dict) else {}
                    model_used = route_dict.get("model_used") or "unknown_model"
                    fallback_occurred = bool(route_dict.get("fallback_occurred", False))
                except Exception as exc:
                    logger.error("Page %d extraction error: %s", page_no, exc, exc_info=True)
                    sanitized_page_err = f"{type(exc).__name__}: Extraction failed for page {page_no}."
                    page_results.append(RoutedPageResult(
                        page_number=page_no, rotation=rotation, error=sanitized_page_err
                    ))
                    info["model_used"] = "-"
                    _safe_diagnostic_sink(diagnostic_sink, f"ext{page_no}", str(exc))
                    yield ProcessingProgressEvent(
                        step_key=f"ext{page_no}",
                        label=f"Page {page_no}: extract fields",
                        state="failed",
                        detail=f"Page {page_no} extraction failed",
                        data={"error": sanitized_page_err, "page_no": page_no},
                    )
                    continue

                info["model_used"] = model_used
                info["fallback_occurred"] = fallback_occurred
                routes.append({"page": page_no, **route_dict})
                page_results.append(RoutedPageResult(
                    page_number=page_no, raw=raw_dict, provenance=prov, route=route_dict, rotation=rotation
                ))

                yield ProcessingProgressEvent(
                    step_key=f"ext{page_no}",
                    label=f"Page {page_no}: extract fields",
                    state="done",
                    detail=f"{model_used}{' (fallback)' if fallback_occurred else ''} · {len(raw_dict or {})} fields",
                    data={"route": route_dict, "raw_fields": raw_dict},
                )

            # Step 5: Canonical processing and envelope assembly (Module C)
            yield ProcessingProgressEvent(
                step_key="merge",
                label="Validate and build JSON",
                state="running",
            )

            try:
                refs = [page_refs[n] for n, _, _ in page_items if n in page_refs]
                env: CanonicalExtractionEnvelope = self.module_c.process_routed_pages(
                    payload_b,
                    page_results,
                    refs,
                    system_prompt=build_generic_prompt(category, doc_info["subtype"], doc_info["language"]),
                    document_type=doc_type or "generic",
                    schema_bound=schema_cls is not None,
                    force_review=category == DocumentCategory.HANDWRITTEN,
                )

                if total_pages > len(page_items):
                    env.flags.append(FlagItem(
                        code="PAGES_NOT_PROCESSED",
                        message=f"Read pages 1-{len(page_items)} of {total_pages} (demo cap DOC_MAX_PAGES={max_pages_limit}).",
                        fields=[],
                    ))

                first = pages_info[0]
                specialists = sorted({p.get("model_used") for p in pages_info if p.get("model_used") not in (None, "-")})
                md = env.document_metadata
                md.document_category = first["category"]
                md.document_subtype = first["subtype"]
                md.language = first["language"]
                md.pages_processed = len(page_items)
                md.pages = [
                    {k: p.get(k) for k in ("page", "category", "document_type", "subtype", "language", "model_used")}
                    for p in pages_info
                ]
                env.run.routing = routes
                env.run.confidence_calibrated = False
                env.run.model_versions.update({
                    k: v for k, v in {
                        "router": first.get("router_model") or first.get("method"),
                        "specialist": ", ".join(specialists),
                        "normalizer": "deterministic-v1",
                    }.items() if v
                })
            except TYPED_DOMAIN_EXCEPTIONS:
                raise
            except Exception as exc:
                logger.error("Module C canonical assembly error: %s", exc, exc_info=True)
                sanitized_msg = "Canonical output assembly failed."
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    payload_b=payload_b,
                    page_results=page_results,
                    pages_info=pages_info,
                    page_items=page_items,
                    routes=routes,
                    error={"code": "CANONICAL_ASSEMBLY_FAILED", "message": sanitized_msg},
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "merge", str(exc))
                yield ProcessingProgressEvent(
                    step_key="merge",
                    label="Validate and build JSON",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            final_res = DocumentProcessingResult(
                document_id=payload_a.document_id,
                tenant_id=payload_a.tenant_id,
                status=env.document_metadata.status,
                envelope=env,
                payload_a=payload_a,
                payload_b=payload_b,
                page_results=page_results,
                pages_info=pages_info,
                page_items=page_items,
                routes=routes,
                execution_time_seconds=time.perf_counter() - t_start,
            )

            yield ProcessingProgressEvent(
                step_key="merge",
                label="Validate and build JSON",
                state="done",
                detail=f"{len(env.fields)} fields · {len(env.flags)} flag(s)",
                data={"rotation": rotation, "total_pages": total_pages},
                result=final_res,
            )
        finally:
            workspace.cleanup()

    def process_document(
        self,
        request: Union[DocumentProcessingRequest, str, Path, BinaryIO],
        claimed_filename: Optional[str] = None,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        category_choice: Optional[str] = None,
        doc_type_choice: Optional[str] = None,
        subtype_choice: Optional[str] = None,
        language_choice: Optional[str] = None,
        engine_choice: Optional[str] = None,
        max_pages: Optional[int] = None,
        progress_callback: Optional[Callable[[ProcessingProgressEvent], None]] = None,
        request_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        diagnostic_sink: Optional[Callable[[str, str], None]] = None,
    ) -> DocumentProcessingResult:
        """
        Synchronous entry point returning a complete DocumentProcessingResult.
        """
        if not isinstance(request, DocumentProcessingRequest):
            resolved_tenant = tenant_id if tenant_id is not None else self.tenant_id
            if resolved_tenant is None:
                raise ValueError("Tenant ID is required and cannot be empty.")
            req = DocumentProcessingRequest(
                file_path_or_stream=request,
                tenant_id=resolved_tenant,
                claimed_filename=claimed_filename,
                actor_id=actor_id or self.actor_id or "system",
                category_choice=category_choice,
                doc_type_choice=doc_type_choice,
                subtype_choice=subtype_choice,
                language_choice=language_choice,
                engine_choice=engine_choice,
                max_pages=max_pages,
                request_id=request_id,
                idempotency_key=idempotency_key,
            )
        else:
            req = request

        last_result: Optional[DocumentProcessingResult] = None
        for event in self.process_document_stream(req, diagnostic_sink=diagnostic_sink):
            if progress_callback is not None:
                progress_callback(event)
            if event.result is not None:
                last_result = event.result

        if last_result is None:
            raise DocumentProcessingError(
                message="Document processing finished without producing a result.",
                details={"document_id": getattr(req, "request_id", "unknown")},
            )
        return last_result

    def process_stored_page_stream(
        self,
        document_id: str,
        page_number: int,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        category_choice: Optional[str] = None,
        doc_type_choice: Optional[str] = None,
        subtype_choice: Optional[str] = None,
        language_choice: Optional[str] = None,
        engine_choice: Optional[str] = None,
        diagnostic_sink: Optional[Callable[[str, str], None]] = None,
        request_id: Optional[str] = None,
    ) -> Iterator[ProcessingProgressEvent]:
        """
        Executes single-page processing using the stored Module A handoff and yields ProcessingProgressEvents.
        The terminal event carries the complete DocumentProcessingResult in its `result` attribute.

        Performs synchronous input and tenant validation before returning the stream generator.
        """
        if not isinstance(document_id, str) or isinstance(document_id, bool) or not document_id.strip():
            raise ValueError("Document ID is required and must be a non-empty string.")

        if "/" in document_id or "\\" in document_id or ".." in document_id:
            raise StorageError(
                message="Path traversal attempt detected in document storage path.",
                details={"document_id": document_id},
            )

        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
            raise ValueError(
                f"Invalid page number: {page_number}. Page numbers must be 1-based positive integers."
            )

        effective_tenant = self._resolve_tenant(tenant_id)

        return self._process_stored_page_stream_generator(
            document_id=document_id,
            page_number=page_number,
            effective_tenant=effective_tenant,
            actor_id=actor_id,
            category_choice=category_choice,
            doc_type_choice=doc_type_choice,
            subtype_choice=subtype_choice,
            language_choice=language_choice,
            engine_choice=engine_choice,
            diagnostic_sink=diagnostic_sink,
            request_id=request_id,
        )

    def _process_stored_page_stream_generator(
        self,
        document_id: str,
        page_number: int,
        effective_tenant: str,
        actor_id: Optional[str] = None,
        category_choice: Optional[str] = None,
        doc_type_choice: Optional[str] = None,
        subtype_choice: Optional[str] = None,
        language_choice: Optional[str] = None,
        engine_choice: Optional[str] = None,
        diagnostic_sink: Optional[Callable[[str, str], None]] = None,
        request_id: Optional[str] = None,
    ) -> Iterator[ProcessingProgressEvent]:
        t_start = time.perf_counter()
        effective_actor = actor_id or self.actor_id or "system"

        def _make_error(code: str, message: str) -> Dict[str, Any]:
            err: Dict[str, Any] = {"code": code, "message": message}
            if request_id is not None:
                err["request_id"] = request_id
            return err

        def _make_event(
            step_key: str,
            label: str,
            state: str,
            detail: Optional[str] = None,
            data: Optional[Dict[str, Any]] = None,
            result: Optional[DocumentProcessingResult] = None,
        ) -> ProcessingProgressEvent:
            ev_data = dict(data) if data is not None else {}
            if request_id is not None:
                ev_data.setdefault("request_id", request_id)
            return ProcessingProgressEvent(
                step_key=step_key,
                label=label,
                state=state,
                detail=detail,
                data=ev_data if ev_data else None,
                result=result,
            )

        _validate_component_tenant(self.storage, effective_tenant, "storage")
        _validate_component_tenant(self.module_a, effective_tenant, "module_a")
        _validate_component_tenant(self.module_b, effective_tenant, "module_b")

        exec_storage = (
            self.storage
            if self.storage is not None
            else LocalDiskStorage(self.settings.storage_base_dir, tenant_id=effective_tenant)
        )
        exec_module_a = (
            self.module_a
            if self.module_a is not None
            else ModuleAPipeline(
                settings=self.settings,
                storage=exec_storage,
                idempotency_store=self.idempotency_store,
                tenant_id=effective_tenant,
            )
        )
        exec_module_b = (
            self.module_b
            if self.module_b is not None
            else ModuleBPipeline(
                storage=exec_storage,
                tenant_id=effective_tenant,
                settings=self.settings,
                idempotency_store=self.idempotency_store,
            )
        )
        _validate_component_tenant(exec_storage, effective_tenant, "storage")
        _validate_component_tenant(exec_module_a, effective_tenant, "module_a")
        _validate_component_tenant(exec_module_b, effective_tenant, "module_b")

        workspace = RequestTemporaryWorkspace(prefix="doc_workspace_")
        workspace.__enter__()
        try:
            choices = {
                "category": category_choice or AUTO,
                "doc_type": doc_type_choice or AUTO,
                "subtype": subtype_choice or AUTO,
                "language": language_choice or AUTO,
            }

            engine = resolve_engine_choice(engine_choice)

            yield _make_event(
                step_key="load",
                label=f"Load stored page {page_number} handoff",
                state="running",
                detail=f"Resolving handoff for document {document_id}",
            )

            try:
                raw_handoff = exec_storage.get_handoff_payload(document_id)
            except TYPED_DOMAIN_EXCEPTIONS:
                raise
            except Exception as exc:
                logger.error("Failed to retrieve handoff payload: %s", exc, exc_info=True)
                sanitized_msg = "Failed to retrieve stored document handoff."
                res = DocumentProcessingResult(
                    document_id=document_id,
                    tenant_id=effective_tenant,
                    status="failed",
                    error=_make_error("STORAGE_ERROR", sanitized_msg),
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "load", str(exc))
                yield _make_event(
                    step_key="load",
                    label=f"Load stored page {page_number} handoff",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            if raw_handoff is None:
                logger.warning("Module A handoff not found for document %s", document_id)
                sanitized_msg = "Module A handoff payload not found for document."
                res = DocumentProcessingResult(
                    document_id=document_id,
                    tenant_id=effective_tenant,
                    status="failed",
                    error=_make_error("HANDOFF_NOT_FOUND", sanitized_msg),
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "load", f"Handoff missing for document {document_id}")
                yield _make_event(
                    step_key="load",
                    label=f"Load stored page {page_number} handoff",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            if isinstance(raw_handoff, ModuleAHandoffPayload):
                payload_a = raw_handoff
            elif isinstance(raw_handoff, dict):
                try:
                    payload_a = ModuleAHandoffPayload.from_dict(raw_handoff)
                except Exception as exc:
                    logger.error("Malformed Module A handoff payload: %s", exc, exc_info=True)
                    sanitized_msg = "Module A handoff payload is malformed or invalid."
                    res = DocumentProcessingResult(
                        document_id=document_id,
                        tenant_id=effective_tenant,
                        status="failed",
                        error=_make_error("MALFORMED_HANDOFF", sanitized_msg),
                        execution_time_seconds=time.perf_counter() - t_start,
                    )
                    _safe_diagnostic_sink(diagnostic_sink, "load", str(exc))
                    yield _make_event(
                        step_key="load",
                        label=f"Load stored page {page_number} handoff",
                        state="failed",
                        detail=sanitized_msg,
                        data={"error": sanitized_msg},
                        result=res,
                    )
                    return
            else:
                logger.error("Unexpected handoff type: %r", type(raw_handoff))
                sanitized_msg = "Module A handoff payload is malformed or invalid."
                res = DocumentProcessingResult(
                    document_id=document_id,
                    tenant_id=effective_tenant,
                    status="failed",
                    error=_make_error("MALFORMED_HANDOFF", sanitized_msg),
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "load", f"Invalid handoff type: {type(raw_handoff).__name__}")
                yield _make_event(
                    step_key="load",
                    label=f"Load stored page {page_number} handoff",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            # Verify document ID binding
            if payload_a.document_id != document_id:
                logger.error(
                    "Handoff document_id mismatch: expected %s, got %s",
                    document_id,
                    payload_a.document_id,
                )
                sanitized_msg = "Module A handoff payload is malformed or invalid."
                res = DocumentProcessingResult(
                    document_id=document_id,
                    tenant_id=effective_tenant,
                    status="failed",
                    error=_make_error("MALFORMED_HANDOFF", sanitized_msg),
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(
                    diagnostic_sink, "load", "Handoff document ID mismatch for requested document"
                )
                yield _make_event(
                    step_key="load",
                    label=f"Load stored page {page_number} handoff",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            # Verify tenant ownership of the retrieved handoff
            if payload_a.tenant_id != effective_tenant:
                raise TenantMismatchError(
                    "Tenant mismatch: stored document belongs to another tenant."
                )

            # Check for rejected or failed handoff status
            if payload_a.status == "rejected":
                raw_err = payload_a.error if isinstance(payload_a.error, dict) else {"code": "REJECTED", "message": "File rejected."}
                err_dict = _make_error(str(raw_err.get("code", "REJECTED")), str(raw_err.get("message", "File rejected.")))
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="rejected",
                    payload_a=payload_a,
                    error=err_dict,
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "load", f"Rejected: {err_dict}")
                yield _make_event(
                    step_key="load",
                    label=f"Load stored page {page_number} handoff",
                    state="failed",
                    detail=str(err_dict.get("message", "File rejected")),
                    data={"payload_a": payload_a.to_dict(), "error": err_dict},
                    result=res,
                )
                return

            if payload_a.status == "failed":
                raw_err = payload_a.error
                raw_msg = f"Module A failed: {raw_err}" if raw_err else "Module A ingestion failed."
                logger.error("Module A handoff recorded failure: %s", raw_msg)
                _safe_diagnostic_sink(diagnostic_sink, "load", raw_msg)

                sanitized_msg = "Document ingestion and preprocessing failed."
                sanitized_a_error = _make_error("INGESTION_ERROR", sanitized_msg)
                sanitized_payload_a = replace(payload_a, error=sanitized_a_error)
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=sanitized_payload_a,
                    error=sanitized_a_error,
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                yield _make_event(
                    step_key="load",
                    label=f"Load stored page {page_number} handoff",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_a_error},
                    result=res,
                )
                return

            # Find requested page record
            target_page_rec: Optional[PageHandoffRecord] = None
            for p_rec in payload_a.pages:
                if getattr(p_rec, "page_number", None) == page_number:
                    target_page_rec = p_rec
                    break

            if target_page_rec is None:
                sanitized_msg = f"Page {page_number} not found in document (document has {len(payload_a.pages)} page(s))."
                res = DocumentProcessingResult(
                    document_id=document_id,
                    tenant_id=effective_tenant,
                    status="failed",
                    payload_a=payload_a,
                    error=_make_error("PAGE_NOT_FOUND", sanitized_msg),
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "load", sanitized_msg)
                yield _make_event(
                    step_key="load",
                    label=f"Load stored page {page_number} handoff",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg, "page_number": page_number},
                    result=res,
                )
                return

            # Resolve page image paths from storage
            try:
                prep = exec_storage.resolve_uri_to_path(target_page_rec.preprocessed_storage_uri)
                orig = exec_storage.resolve_uri_to_path(target_page_rec.original_storage_uri)
                img_path = prep if prep.exists() else orig
                if not img_path.exists():
                    raise FileNotFoundError(f"Page image for page {page_number} not found in storage.")
                page_items: List[Tuple[int, Path, Optional[Path]]] = [(page_number, img_path, orig)]
            except TYPED_DOMAIN_EXCEPTIONS:
                raise
            except Exception as exc:
                logger.error("Storage URI resolution error: %s", exc, exc_info=True)
                sanitized_msg = "Failed to resolve storage URIs for document page."
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    error=_make_error("STORAGE_RESOLUTION_FAILED", sanitized_msg),
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "load", str(exc))
                yield _make_event(
                    step_key="load",
                    label=f"Load stored page {page_number} handoff",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            yield _make_event(
                step_key="load",
                label=f"Load stored page {page_number} handoff",
                state="done",
                detail=f"Resolved page {page_number} of {len(payload_a.pages)}",
                data={"total_pages": len(payload_a.pages), "page_items": page_items},
            )

            # Step 2: Module B hand-off (isolate requested page)
            try:
                page_refs = {ref.page_number: ref for ref in exec_module_b.load_page_images(payload_a, page_numbers=[page_number])}
                payload_b = exec_module_b.process_handoff(payload_a, page_numbers=[page_number])
            except TYPED_DOMAIN_EXCEPTIONS:
                raise
            except Exception as exc:
                logger.error("Module B processing error: %s", exc, exc_info=True)
                sanitized_msg = "Module B processing failed."
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    error=_make_error("MODULE_B_ERROR", sanitized_msg),
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "cls", str(exc))
                yield _make_event(
                    step_key="cls",
                    label=f"Classify document for page {page_number}",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            if not isinstance(payload_b, ModuleBOutputPayload):
                logger.error("Module B returned invalid payload: %r", type(payload_b))
                sanitized_msg = "Module B processing failed: invalid payload produced."
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    error=_make_error("MODULE_B_ERROR", sanitized_msg),
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "cls", f"Module B returned invalid payload: {type(payload_b).__name__}")
                yield _make_event(
                    step_key="cls",
                    label=f"Classify document for page {page_number}",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            if payload_b.status == "failed":
                raw_err = payload_b.error if isinstance(payload_b.error, dict) else {}
                raw_msg = f"Module B failed: {raw_err.get('code', 'UNKNOWN')}: {raw_err.get('message', 'no detail')}"
                logger.error("Module B processing failed: %s", raw_msg)
                _safe_diagnostic_sink(diagnostic_sink, "cls", raw_msg)

                sanitized_b_error = _make_error("MODULE_B_ERROR", "Document page classification and layout analysis failed.")
                sanitized_payload_b = payload_b.model_copy(update={"error": sanitized_b_error})
                res = DocumentProcessingResult(
                    document_id=payload_b.document_id,
                    tenant_id=payload_b.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    payload_b=sanitized_payload_b,
                    error=sanitized_b_error,
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                yield _make_event(
                    step_key="cls",
                    label=f"Classify document for page {page_number}",
                    state="failed",
                    detail=sanitized_b_error["message"],
                    data={"error": sanitized_b_error},
                    result=res,
                )
                return

            module_b_type = payload_b.classification.document_type.value
            rule_type = module_b_type if module_b_type != "UNKNOWN" else None
            rule_confidence = payload_b.classification.classification_confidence if rule_type else None

            # Step 3: Classify page
            yield _make_event(
                step_key="cls",
                label=f"Classify document for page {page_number}",
                state="running",
                detail="detecting category, type and language",
            )

            try:
                doc_info = self.resolve_page(
                    page_items[0][1],
                    payload_a.claimed_filename,
                    rule_type,
                    rule_confidence,
                    choices,
                    page_number,
                )
            except TYPED_DOMAIN_EXCEPTIONS:
                raise
            except Exception as exc:
                logger.error("Classification/routing error: %s", exc, exc_info=True)
                sanitized_msg = "Document classification and routing failed."
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    payload_b=payload_b,
                    error=_make_error("ROUTING_FAILED", sanitized_msg),
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "cls", str(exc))
                yield _make_event(
                    step_key="cls",
                    label=f"Classify document for page {page_number}",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            cls_detail = (
                f"{doc_info['category']} · {doc_info['document_type']} · {doc_info['language']} · {doc_info['category_source']}"
                + (f" · filled by {doc_info['filled_by']}" if doc_info.get("filled_by") else "")
                + (f" · {doc_info['router_model'].split('/')[-1]}" if doc_info.get("router_model") else "")
            )

            rotation = doc_info.get("rotation", 0)
            read_items = page_items
            if rotation:
                try:
                    read_items = [
                        (page_number, workspace.rotate_page(page_items[0][1], rotation, page_no=page_number), page_items[0][2])
                    ]
                except TYPED_DOMAIN_EXCEPTIONS:
                    raise
                except Exception as exc:
                    logger.error("Page rotation error: %s", exc, exc_info=True)
                    sanitized_msg = "Document page rotation failed."
                    res = DocumentProcessingResult(
                        document_id=payload_a.document_id,
                        tenant_id=payload_a.tenant_id,
                        status="failed",
                        payload_a=payload_a,
                        payload_b=payload_b,
                        error=_make_error("ROTATION_FAILED", sanitized_msg),
                        execution_time_seconds=time.perf_counter() - t_start,
                    )
                    _safe_diagnostic_sink(diagnostic_sink, "cls", str(exc))
                    yield _make_event(
                        step_key="cls",
                        label=f"Classify document for page {page_number}",
                        state="failed",
                        detail=sanitized_msg,
                        data={"error": sanitized_msg},
                        result=res,
                    )
                    return
                cls_detail += f" · rotated {rotation}°"

            yield _make_event(
                step_key="cls",
                label=f"Classify document for page {page_number}",
                state="done",
                detail=cls_detail,
                data={"doc_info": doc_info, "rotation": rotation},
            )

            category = DocumentCategory(doc_info["category"])
            schema_cls = SCHEMA_MAP.get(doc_info["schema_id"]) if doc_info["schema_id"] else None
            doc_type = doc_info["schema_id"]

            if doc_type and payload_b.classification.document_type.value != doc_type:
                try:
                    payload_b = exec_module_b.apply_reviewer_override(
                        payload_b,
                        ReviewerOverrideRequest(
                            document_id=payload_b.document_id,
                            tenant_id=payload_b.tenant_id,
                            reviewer_id=effective_actor,
                            new_document_type=MVPDocumentType(doc_type),
                            override_reason="Document type explicitly selected in the extraction UI",
                        ),
                    )
                except TYPED_DOMAIN_EXCEPTIONS:
                    raise
                except Exception as exc:
                    logger.error("Reviewer override error: %s", exc, exc_info=True)
                    sanitized_msg = "Module B reviewer override failed."
                    res = DocumentProcessingResult(
                        document_id=payload_a.document_id,
                        tenant_id=payload_a.tenant_id,
                        status="failed",
                        payload_a=payload_a,
                        payload_b=payload_b,
                        error=_make_error("MODULE_B_ERROR", sanitized_msg),
                        execution_time_seconds=time.perf_counter() - t_start,
                    )
                    _safe_diagnostic_sink(diagnostic_sink, "cls", str(exc))
                    yield _make_event(
                        step_key="cls",
                        label=f"Classify document for page {page_number}",
                        state="failed",
                        detail=sanitized_msg,
                        data={"error": sanitized_msg},
                        result=res,
                    )
                    return

            # Step 4: Per-page extraction with specialist model
            pages_info: List[Dict[str, Any]] = []
            page_results: List[RoutedPageResult] = []
            routes: List[Dict[str, Any]] = []

            for page_no, img_path, _orig in read_items:
                info = dict(doc_info, page=page_no)
                pages_info.append(info)

                yield _make_event(
                    step_key=f"ext{page_no}",
                    label=f"Page {page_no}: extract fields",
                    state="running",
                    detail=f"routing to the {category.value} model",
                    data={
                        "page_no": page_no,
                        "read_count": len(read_items),
                        "category": category.value,
                        "language": info["language"],
                        "pages_info": list(pages_info),
                    },
                )

                try:
                    if not (img_path and img_path.exists()):
                        raise FileNotFoundError(f"Page {page_no} image not found.")
                    raw_dict, prov, route = self.routed_extractor.extract(
                        str(img_path), category, info["language"], schema_cls, info["subtype"], engine=engine
                    )
                    route_dict = route if isinstance(route, dict) else {}
                    model_used = route_dict.get("model_used") or "unknown_model"
                    fallback_occurred = bool(route_dict.get("fallback_occurred", False))
                except Exception as exc:
                    logger.error("Page %d extraction error: %s", page_no, exc, exc_info=True)
                    sanitized_page_err = f"{type(exc).__name__}: Extraction failed for page {page_no}."
                    page_results.append(RoutedPageResult(
                        page_number=page_no, rotation=rotation, error=sanitized_page_err
                    ))
                    info["model_used"] = "-"
                    _safe_diagnostic_sink(diagnostic_sink, f"ext{page_no}", str(exc))
                    yield _make_event(
                        step_key=f"ext{page_no}",
                        label=f"Page {page_no}: extract fields",
                        state="failed",
                        detail=f"Page {page_no} extraction failed",
                        data={"error": sanitized_page_err, "page_no": page_no},
                    )
                    continue

                info["model_used"] = model_used
                info["fallback_occurred"] = fallback_occurred
                routes.append({"page": page_no, **route_dict})
                page_results.append(RoutedPageResult(
                    page_number=page_no, raw=raw_dict, provenance=prov, route=route_dict, rotation=rotation
                ))

                yield _make_event(
                    step_key=f"ext{page_no}",
                    label=f"Page {page_no}: extract fields",
                    state="done",
                    detail=f"{model_used}{' (fallback)' if fallback_occurred else ''} · {len(raw_dict or {})} fields",
                    data={"route": route_dict, "raw_fields": raw_dict},
                )

            # Step 5: Canonical processing and envelope assembly (Module C)
            yield _make_event(
                step_key="merge",
                label="Validate and build JSON",
                state="running",
            )

            try:
                refs = [page_refs[n] for n, _, _ in page_items if n in page_refs]
                env: CanonicalExtractionEnvelope = self.module_c.process_routed_pages(
                    payload_b,
                    page_results,
                    refs,
                    system_prompt=build_generic_prompt(category, doc_info["subtype"], doc_info["language"]),
                    document_type=doc_type or "generic",
                    schema_bound=schema_cls is not None,
                    force_review=category == DocumentCategory.HANDWRITTEN,
                    expected_page_numbers=[page_number],
                    total_document_pages=len(payload_a.pages),
                )

                first = pages_info[0]
                specialists = sorted({p.get("model_used") for p in pages_info if p.get("model_used") not in (None, "-")})
                md = env.document_metadata
                md.document_category = first["category"]
                md.document_subtype = first["subtype"]
                md.language = first["language"]
                md.pages_processed = len(page_items)
                md.pages = [
                    {k: p.get(k) for k in ("page", "category", "document_type", "subtype", "language", "model_used")}
                    for p in pages_info
                ]
                env.run.routing = routes
                env.run.confidence_calibrated = False
                env.run.model_versions.update({
                    k: v for k, v in {
                        "router": first.get("router_model") or first.get("method"),
                        "specialist": ", ".join(specialists),
                        "normalizer": "deterministic-v1",
                    }.items() if v
                })
            except TYPED_DOMAIN_EXCEPTIONS:
                raise
            except Exception as exc:
                logger.error("Module C canonical assembly error: %s", exc, exc_info=True)
                sanitized_msg = "Canonical output assembly failed."
                res = DocumentProcessingResult(
                    document_id=payload_a.document_id,
                    tenant_id=payload_a.tenant_id,
                    status="failed",
                    payload_a=payload_a,
                    payload_b=payload_b,
                    page_results=page_results,
                    pages_info=pages_info,
                    page_items=page_items,
                    routes=routes,
                    error=_make_error("CANONICAL_ASSEMBLY_FAILED", sanitized_msg),
                    execution_time_seconds=time.perf_counter() - t_start,
                )
                _safe_diagnostic_sink(diagnostic_sink, "merge", str(exc))
                yield _make_event(
                    step_key="merge",
                    label="Validate and build JSON",
                    state="failed",
                    detail=sanitized_msg,
                    data={"error": sanitized_msg},
                    result=res,
                )
                return

            final_res = DocumentProcessingResult(
                document_id=payload_a.document_id,
                tenant_id=payload_a.tenant_id,
                status=env.document_metadata.status,
                envelope=env,
                payload_a=payload_a,
                payload_b=payload_b,
                page_results=page_results,
                pages_info=pages_info,
                page_items=page_items,
                routes=routes,
                execution_time_seconds=time.perf_counter() - t_start,
            )

            yield _make_event(
                step_key="merge",
                label="Validate and build JSON",
                state="done",
                detail=f"{len(env.fields)} fields · {len(env.flags)} flag(s)",
                data={"rotation": rotation, "total_pages": len(payload_a.pages), "page_number": page_number},
                result=final_res,
            )
        finally:
            workspace.cleanup()

    def process_stored_page(
        self,
        document_id: str,
        page_number: int,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        category_choice: Optional[str] = None,
        doc_type_choice: Optional[str] = None,
        subtype_choice: Optional[str] = None,
        language_choice: Optional[str] = None,
        engine_choice: Optional[str] = None,
        progress_callback: Optional[Callable[[ProcessingProgressEvent], None]] = None,
        diagnostic_sink: Optional[Callable[[str, str], None]] = None,
        request_id: Optional[str] = None,
    ) -> DocumentProcessingResult:
        """
        Synchronous entry point returning a complete DocumentProcessingResult for an individual stored page.
        """
        if not isinstance(document_id, str) or isinstance(document_id, bool) or not document_id.strip():
            raise ValueError("Document ID is required and must be a non-empty string.")

        last_result: Optional[DocumentProcessingResult] = None
        for event in self.process_stored_page_stream(
            document_id=document_id,
            page_number=page_number,
            tenant_id=tenant_id,
            actor_id=actor_id,
            category_choice=category_choice,
            doc_type_choice=doc_type_choice,
            subtype_choice=subtype_choice,
            language_choice=language_choice,
            engine_choice=engine_choice,
            diagnostic_sink=diagnostic_sink,
            request_id=request_id,
        ):
            if progress_callback is not None:
                progress_callback(event)
            if event.result is not None:
                last_result = event.result

        if last_result is None:
            raise DocumentProcessingError(
                message="Stored page processing finished without producing a result.",
                details={"document_id": document_id, "page_number": page_number},
            )
        return last_result
