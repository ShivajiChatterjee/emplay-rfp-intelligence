
from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

OperationalFieldStatus = Literal[
    "passed", "not_found_evidence", "provider_error", "retrieval_error",
    "reconciliation_incomplete", "validation_failed",
]


FieldStatus = Literal[
    "found",
    "not_found",
    "conflict",
    "error",
]

ExpectedValueType = Literal[
    "string",
    "datetime",
    "list",
    "object",
    "summary",
]

SpecialistGroup = Literal[
    "dates_logistics",
    "commercial_legal",
    "product_specifications",
    "qa_report",
]

ExtractionSpecialist = Literal[
    "dates_logistics",
    "commercial_legal",
    "product_specifications",
]

WorkflowMode = Literal["extract", "ask", "compare"]
PlanAction = Literal[
    "ensure_index",
    "retrieve",
    "extract",
    "reconcile",
    "validate",
    "answer",
    "report",
]
AllowedTool = Literal[
    "ingestion.ensure_index",
    "retrieval.hybrid_cohere",
    "gemini.extract",
    "gemini.reconcile",
    "validator.check",
    "gemini.answer",
    "report.finalize",
]


def merge_state_mappings(left: dict, right: dict) -> dict:
    """Merge independent per-field node updates for future LangGraph use."""

    return {**left, **right}


class FieldSpec(BaseModel):
    """Bid-agnostic instructions for extracting one official field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: str = Field(min_length=1)
    questions: tuple[str, ...] = Field(min_length=1)
    expected_type: ExpectedValueType
    specialist_group: SpecialistGroup
    search_filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_questions(self) -> "FieldSpec":
        if any(not question.strip() for question in self.questions):
            raise ValueError("Field questions cannot be blank")
        return self


class Evidence(BaseModel):
    file_name: str
    page_number: int = Field(ge=1)
    quote: str
    chunk_id: str
    doc_type: str | None = None
    addendum_number: int | None = None
    document_date: str | None = None
    # Internal retrieval structure used for completeness checks.  Final output
    # still publishes only file/page SourceCitation records.
    section: str | None = None
    chunk_type: str | None = None
    table_id: str | None = None


class FieldCandidate(BaseModel):
    value: Any
    evidence: Evidence
    confidence: float = Field(ge=0, le=1)


class ExtractedField(BaseModel):
    field_name: str
    status: FieldStatus = "not_found"
    value: Any = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    citations: list[Evidence] = Field(default_factory=list)
    candidates: list[FieldCandidate] = Field(default_factory=list)
    reasoning: str = ""


class ValidationResult(BaseModel):
    field_name: str
    valid: bool
    errors: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    retry_required: bool = False


class ConflictRecord(BaseModel):
    field_name: str
    original_value: Any = None
    updated_value: Any = None
    resolution: str
    supporting_evidence: list[Evidence] = Field(
        default_factory=list
    )


class AgentTrace(BaseModel):
    agent: str
    action: str
    status: Literal["success", "error", "skipped"]
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    latency_ms: float = Field(default=0, ge=0)
    tool_calls: list[str] = Field(default_factory=list)
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    tokens_used: int | None = None
    error: str | None = None
    run_id: str | None = None
    field_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FieldExtractionRequest(BaseModel):
    """Evidence package sent to one extraction specialist."""

    bid_id: str = Field(min_length=1)
    field_spec: FieldSpec
    evidence: list[Evidence] = Field(default_factory=list)


class EvidenceSearchRequest(BaseModel):
    """Provider-neutral search request accepted by the retrieval adapter."""

    bid_id: str
    field_name: str
    question: str
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=20, ge=1)
    doc_type: str | None = None
    addendum_number: int | None = None
    file_name: str | None = None


class ExtractionRequest(BaseModel):
    """Workflow input containing caller-selected, reusable field specs."""

    bid_id: str = Field(min_length=1)
    field_specs: tuple[FieldSpec, ...] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def validate_request(self) -> "ExtractionRequest":
        names = [spec.field_name for spec in self.field_specs]
        if len(names) != len(set(names)):
            raise ValueError("field_specs must contain unique field names")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class AgentTask(BaseModel):
    task_id: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    field_names: tuple[str, ...] = Field(default_factory=tuple)
    depends_on: tuple[str, ...] = Field(default_factory=tuple)


class AgentPlan(BaseModel):
    bid_id: str = Field(min_length=1)
    tasks: tuple[AgentTask, ...] = Field(min_length=1)
    rationale: str = ""


class ExecutionTask(BaseModel):
    task_id: str = Field(min_length=1)
    action: PlanAction
    agent: str = Field(min_length=1)
    tool: AllowedTool
    bid_ids: tuple[str, ...] = Field(min_length=1)
    field_names: tuple[str, ...] = Field(default_factory=tuple)
    question: str | None = None
    depends_on: tuple[str, ...] = Field(default_factory=tuple)
    completion_criteria: str = Field(min_length=1)


class ExecutionPlan(BaseModel):
    mode: WorkflowMode
    goal: str = Field(min_length=1)
    tasks: tuple[ExecutionTask, ...] = Field(min_length=1)
    selected_fields: tuple[str, ...] = Field(default_factory=tuple)
    stop_condition: str = Field(min_length=1)
    rationale: str = ""


class CompactExtractionPlan(BaseModel):
    """Concise Gemini decision that Python expands into field-level work."""

    mode: Literal["extract"]
    workflow: Literal["full_bid_extraction"]
    specialist_groups: tuple[ExtractionSpecialist, ...] = Field(
        min_length=3,
        max_length=3,
    )
    actions: tuple[PlanAction, ...] = Field(min_length=6, max_length=6)
    tools: tuple[AllowedTool, ...] = Field(min_length=6, max_length=6)
    evidence_requirements: tuple[str, ...] = Field(min_length=1, max_length=4)
    completion_criteria: tuple[str, ...] = Field(min_length=1, max_length=4)
    stop_condition: str = Field(min_length=1, max_length=240)
    rationale: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def validate_compact_plan(self) -> "CompactExtractionPlan":
        required_specialists = {
            "dates_logistics",
            "commercial_legal",
            "product_specifications",
        }
        if set(self.specialist_groups) != required_specialists:
            raise ValueError("Full extraction requires all three specialist groups")
        if len(set(self.specialist_groups)) != 3:
            raise ValueError("Specialist groups must be unique")
        required_actions = {
            "ensure_index",
            "retrieve",
            "extract",
            "reconcile",
            "validate",
            "report",
        }
        if set(self.actions) != required_actions or len(set(self.actions)) != 6:
            raise ValueError("Compact extraction actions are incomplete or duplicated")
        required_tools = {
            "ingestion.ensure_index",
            "retrieval.hybrid_cohere",
            "gemini.extract",
            "gemini.reconcile",
            "validator.check",
            "report.finalize",
        }
        if set(self.tools) != required_tools or len(set(self.tools)) != 6:
            raise ValueError("Compact extraction tools are incomplete or duplicated")
        return self


class PlanningOutcome(BaseModel):
    plan: ExecutionPlan
    trace: AgentTrace
    compact_plan: CompactExtractionPlan | None = None


class QueryRewrite(BaseModel):
    question: str = Field(min_length=1)
    doc_type: str | None = None
    addendum_number: int | None = Field(default=None, ge=1)
    file_name: str | None = None
    rationale: str = ""


class CriticDecision(BaseModel):
    field_name: str
    status: Literal[
        "passed", "failed", "not_found", "not_found_evidence",
        "provider_error", "retrieval_error", "reconciliation_incomplete",
        "validation_failed",
    ]
    reasons: list[str] = Field(default_factory=list)
    corrective_instruction: str | None = None
    revised_question: str | None = None
    retry_required: bool = False
    confidence: float = Field(ge=0, le=1)


class ReconciliationDecision(BaseModel):
    field_name: str
    amended: bool = False
    selected_value: Any = None
    original_value: Any = None
    updated_value: Any = None
    original_evidence_rank: int | None = Field(default=None, ge=1)
    updated_evidence_rank: int | None = Field(default=None, ge=1)
    addendum_number: int | None = Field(default=None, ge=1)
    addendum_date: str | None = None
    explanation: str = ""
    confidence: float = Field(ge=0, le=1)
    change_mode: Literal["append", "replace"] = "append"


class ProductSpecificationPatch(BaseModel):
    """One evidence-backed amendment to a product specification object."""

    operation: Literal["set_attribute", "add_category", "remove_category"]
    category: str = Field(min_length=1)
    attribute: str | None = None
    value: Any = None
    evidence_rank: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_patch_shape(self) -> "ProductSpecificationPatch":
        if self.operation == "set_attribute" and not self.attribute:
            raise ValueError("set_attribute requires attribute")
        if self.operation == "set_attribute" and self.value is None:
            raise ValueError("set_attribute requires a non-null value")
        if self.operation == "add_category" and not isinstance(self.value, dict):
            raise ValueError("add_category requires an object value")
        if self.operation == "remove_category" and self.value is not None:
            raise ValueError("remove_category value must be null")
        return self


class ProductSpecificationReconciliationDecision(BaseModel):
    field_name: Literal["Product Specification"] = "Product Specification"
    amended: bool = False
    patches: list[ProductSpecificationPatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_amended_patches(self) -> "ProductSpecificationReconciliationDecision":
        if self.amended != bool(self.patches):
            raise ValueError("amended must match whether patches are present")
        return self


class CitedAnswer(BaseModel):
    answer: str
    sources: list["AnswerCitation"] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    notes: str = ""


class AnswerCitation(BaseModel):
    bid_id: str
    file: str
    page: int = Field(ge=1)
    chunk_id: str


class SourceCitation(BaseModel):
    """Compact citation format required by the final extraction JSON."""

    file: str = Field(min_length=1)
    page: int = Field(ge=1)


class AddendumChange(BaseModel):
    field_name: str = Field(min_length=1)
    original_value: Any = None
    updated_value: Any = None
    addendum_number: int | None = Field(default=None, ge=1)
    addendum_date: str | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    notes: str = ""


class FinalFieldRecord(BaseModel):
    value: Any = None
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    notes: str = ""


class ValidationSummary(BaseModel):
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    not_found: int = Field(default=0, ge=0)


class FinalExtractionOutput(BaseModel):
    bid_id: str = Field(min_length=1)
    fields: dict[str, FinalFieldRecord]
    addendum_changes: list[AddendumChange] = Field(default_factory=list)
    validation: ValidationSummary
    trace: list[AgentTrace] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_official_fields(self) -> "FinalExtractionOutput":
        # Local import keeps the registry as the single source of truth while
        # avoiding a module-import cycle during FieldSpec construction.
        from agents.field_specs import OFFICIAL_FIELD_LABELS

        actual = set(self.fields)
        expected = set(OFFICIAL_FIELD_LABELS)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "fields must contain exactly the official labels; "
                f"missing={missing}, extra={extra}"
            )

        validated_count = (
            self.validation.passed
            + self.validation.failed
            + self.validation.not_found
        )
        if validated_count != len(OFFICIAL_FIELD_LABELS):
            raise ValueError(
                "validation counts must total the number of official fields"
            )
        return self


class FieldRunRecord(BaseModel):
    status: OperationalFieldStatus
    terminal: bool = False
    reusable: bool = False
    reasons: list[str] = Field(default_factory=list)
    reconciliation_verified: bool = False


class RunManifest(BaseModel):
    artifact_type: Literal["production_extraction_run_manifest"] = "production_extraction_run_manifest"
    run_id: str
    bid_id: str
    run_status: Literal["complete", "partial", "incomplete"]
    final_submission_published: bool = False
    checkpoint_fingerprint: str
    field_statuses: dict[str, FieldRunRecord]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    output_path: str | None = None
    status_counts: dict[OperationalFieldStatus, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def complete_only_with_terminal_evidence(self) -> "RunManifest":
        calculated = {
            status: sum(item.status == status for item in self.field_statuses.values())
            for status in (
                "passed", "not_found_evidence", "provider_error", "retrieval_error",
                "reconciliation_incomplete", "validation_failed",
            )
        }
        self.status_counts = {key: value for key, value in calculated.items() if value}
        if self.run_status == "complete":
            from agents.field_specs import OFFICIAL_FIELD_LABELS
            if set(self.field_statuses) != set(OFFICIAL_FIELD_LABELS):
                raise ValueError("complete run must contain exactly all official fields")
            if not all(
                item.terminal and item.status in {"passed", "not_found_evidence"}
                for item in self.field_statuses.values()
            ):
                raise ValueError("complete run contains unresolved field statuses")
        if self.final_submission_published != (self.run_status == "complete"):
            raise ValueError("final submission publication must match complete run status")
        return self


class PartialFieldRecord(FinalFieldRecord):
    validation_status: Literal["passed", "failed", "not_found"]
    validation_reasons: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)


class PartialUsage(BaseModel):
    gemini_calls: int = Field(default=0, ge=0)
    qdrant_cohere_queries: int = Field(default=0, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    calls_without_token_metadata: int = Field(default=0, ge=0)


class PartialExtractionOutput(BaseModel):
    """Explicitly non-final artifact for a bounded subset smoke test."""

    artifact_type: Literal["partial_extraction_smoke"] = "partial_extraction_smoke"
    partial: Literal[True] = True
    final_submission: Literal[False] = False
    bid_id: str = Field(min_length=1)
    selected_fields: tuple[str, ...] = Field(min_length=1, max_length=3)
    api_limits: dict[str, int]
    index_status: dict[str, Any]
    fields: dict[str, PartialFieldRecord]
    addendum_changes: list[AddendumChange] = Field(default_factory=list)
    validation: ValidationSummary
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    usage: PartialUsage = Field(default_factory=PartialUsage)
    latency_ms: float = Field(ge=0)
    trace: list[AgentTrace] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_partial_fields(self) -> "PartialExtractionOutput":
        from agents.field_specs import OFFICIAL_FIELD_LABELS

        selected = set(self.selected_fields)
        if len(selected) != len(self.selected_fields):
            raise ValueError("selected_fields must be unique")
        if not selected.issubset(set(OFFICIAL_FIELD_LABELS)):
            raise ValueError("selected_fields must use official field labels")
        if set(self.fields) != selected:
            raise ValueError("partial fields must exactly match selected_fields")
        total = (
            self.validation.passed
            + self.validation.failed
            + self.validation.not_found
        )
        if total != len(self.selected_fields):
            raise ValueError("partial validation counts must total selected_fields")
        return self


class WorkflowState(BaseModel):
    bid_id: str
    goal: str = "extract_bid"
    plan: list[str] = Field(default_factory=list)
    field_specs: tuple[FieldSpec, ...] = Field(default_factory=tuple)
    agent_plan: AgentPlan | None = None

    # Retrieved evidence, grouped by extraction field.
    evidence: Annotated[
        dict[str, list[Evidence]],
        merge_state_mappings,
    ] = Field(
        default_factory=dict
    )

    # Draft and final extracted fields.
    draft_fields: Annotated[
        dict[str, ExtractedField],
        merge_state_mappings,
    ] = Field(
        default_factory=dict
    )
    final_fields: Annotated[
        dict[str, ExtractedField],
        merge_state_mappings,
    ] = Field(
        default_factory=dict
    )

    # Validation and conflict information.
    validation_results: Annotated[
        dict[str, ValidationResult],
        merge_state_mappings,
    ] = Field(
        default_factory=dict
    )
    conflicts: Annotated[
        list[ConflictRecord],
        operator.add,
    ] = Field(
        default_factory=list
    )
    addendum_changes: Annotated[
        list[AddendumChange],
        operator.add,
    ] = Field(default_factory=list)

    # Retry count for each failed field.
    retries: Annotated[
        dict[str, int],
        merge_state_mappings,
    ] = Field(default_factory=dict)
    max_retries: int = Field(default=2, ge=0)

    # Observability and error reporting.
    agent_trace: Annotated[
        list[AgentTrace],
        operator.add,
    ] = Field(default_factory=list)
    warnings: Annotated[
        list[str],
        operator.add,
    ] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    bid_id: str
    fields: dict[str, ExtractedField]
    conflicts: list[ConflictRecord] = Field(
        default_factory=list
    )
    agent_trace: list[AgentTrace] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(default_factory=list)
