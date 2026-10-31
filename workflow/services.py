"""Dependency-injection boundary between LangGraph nodes and real agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class WorkflowServices:
    plan: Callable[..., Any]
    ensure_index: Callable[..., Any]
    retrieve: Callable[..., Any]
    extract: Callable[..., Any]
    reconcile: Callable[..., Any]
    validate: Callable[..., Any]
    answer: Callable[..., Any]
    summarize: Callable[..., Any]
    assemble: Callable[..., Any]

    @classmethod
    def default(cls) -> "WorkflowServices":
        # Lazy imports keep graph unit tests independent of Qdrant/Cohere native
        # dependencies and avoid initializing external clients at import time.
        from agents.adaptive_retrieval import retrieve_adaptively
        from agents.extractor import extract_field
        from agents.ingestion_agent import ensure_bid_indexed
        from agents.planner import create_execution_plan_with_trace
        from agents.reconciler import reconcile_field
        from agents.report_agent import (
            answer_question,
            assemble_final_output,
            generate_bid_summary,
        )
        from agents.validator import validate_field

        return cls(
            plan=create_execution_plan_with_trace,
            ensure_index=ensure_bid_indexed,
            retrieve=retrieve_adaptively,
            extract=extract_field,
            reconcile=reconcile_field,
            validate=validate_field,
            answer=lambda **kwargs: answer_question(**kwargs, include_trace=True),
            summarize=lambda **kwargs: generate_bid_summary(**kwargs, include_trace=True),
            assemble=assemble_final_output,
        )
