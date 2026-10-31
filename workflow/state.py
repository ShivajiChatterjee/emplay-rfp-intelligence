"""Typed graph input and concurrency-safe shared state."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

from agents.schemas import (
    AddendumChange,
    CitedAnswer,
    CriticDecision,
    Evidence,
    ExecutionPlan,
    ExtractedField,
    FinalExtractionOutput,
    FieldRunRecord,
    AgentTrace,
)


def merge_mappings(left: dict, right: dict) -> dict:
    return {**left, **right}


class WorkflowRequest(BaseModel):
    mode: Literal["extract", "ask", "compare"]
    bid_folders: tuple[Path, ...] = Field(min_length=1)
    goal: str | None = None
    question: str | None = None
    output_dir: Path = Path("outputs")
    max_retries: int = Field(default=2, ge=0, le=2)
    allow_query_rewrite: bool = True
    semantic_validation: bool = True
    resume: bool = False
    checkpoint_path: Path | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> "WorkflowRequest":
        if self.mode in {"ask", "compare"} and not (
            self.question and self.question.strip()
        ):
            raise ValueError(f"{self.mode} mode requires a question")
        if self.mode == "compare" and len(self.bid_folders) < 2:
            raise ValueError("compare mode requires at least two bid folders")
        if self.mode == "extract" and len(self.bid_folders) != 1:
            raise ValueError("extract mode processes exactly one bid at a time")
        return self

    @property
    def bid_ids(self) -> tuple[str, ...]:
        return tuple(path.name for path in self.bid_folders)

    @property
    def resolved_goal(self) -> str:
        if self.goal:
            return self.goal
        if self.mode == "extract":
            return "Extract and validate all official RFP fields"
        if self.mode == "compare":
            return f"Compare bids using the question: {self.question}"
        return f"Answer the RFP question: {self.question}"


class GraphState(TypedDict, total=False):
    request: dict[str, Any]
    run_id: str
    plan: ExecutionPlan
    planner_failed: bool
    index_status: Annotated[dict[str, Any], merge_mappings]
    evidence: Annotated[dict[str, list[Evidence]], merge_mappings]
    fields: Annotated[dict[str, ExtractedField], merge_mappings]
    validation: Annotated[dict[str, CriticDecision], merge_mappings]
    validation_attempts: Annotated[dict[str, int], merge_mappings]
    retry_counts: Annotated[dict[str, int], merge_mappings]
    addendum_changes: Annotated[list[AddendumChange], operator.add]
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]
    agent_traces: Annotated[list[AgentTrace], operator.add]
    qa_answer: CitedAnswer
    final_output: FinalExtractionOutput
    field_statuses: Annotated[dict[str, FieldRunRecord], merge_mappings]
    checkpoint_fingerprint: str

    # Isolated Send payload fields used by one specialist worker.
    worker_bid_id: str
    worker_field_name: str
    worker_question: str
    worker_attempt: int
