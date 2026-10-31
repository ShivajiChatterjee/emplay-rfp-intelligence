"""Deterministic conditional routing for the LangGraph workflow."""

from __future__ import annotations

from langgraph.types import Send

from agents.field_specs import OFFICIAL_FIELD_LABELS, get_field_spec
from workflow.state import GraphState


def route_after_planner(state: GraphState) -> str:
    return "failure" if state.get("planner_failed") else "ingest"


def route_after_ingestion(state: GraphState):
    statuses = state.get("index_status", {})
    if not statuses or all(
        item.get("status") == "error" for item in statuses.values()
    ):
        return "failure"

    plan = state["plan"]
    if plan.mode in {"ask", "compare"}:
        return "qa"

    bid_id = state["request"]["bid_ids"][0]
    sends = []
    for field_name in plan.selected_fields:
        if field_name == "Bid Summary":
            continue
        existing = state.get("field_statuses", {}).get(field_name)
        if existing and existing.reusable and existing.terminal:
            continue
        spec = get_field_spec(field_name)
        sends.append(
            Send(
                "field_worker",
                {
                    "request": state["request"],
                    "run_id": state["run_id"],
                    "worker_bid_id": bid_id,
                    "worker_field_name": field_name,
                    "worker_question": spec.questions[0],
                    "worker_attempt": 0,
                },
            )
        )
    return sends or "report"


def route_after_validation(state: GraphState):
    sends = []
    for field_name, decision in state.get("validation", {}).items():
        attempt = state.get("retry_counts", {}).get(field_name, 0)
        validated_attempt = state.get("validation_attempts", {}).get(
            field_name, -1
        )
        if (
            decision.status == "failed"
            and decision.retry_required
            and attempt < state["request"]["max_retries"]
            and validated_attempt == attempt
        ):
            sends.append(
                Send(
                    "field_worker",
                    {
                        "request": state["request"],
                        "run_id": state["run_id"],
                        "worker_bid_id": state["request"]["bid_ids"][0],
                        "worker_field_name": field_name,
                        "worker_question": (
                            decision.revised_question
                            or get_field_spec(field_name).questions[0]
                        ),
                        "worker_attempt": attempt + 1,
                    },
                )
            )
    return sends or "report"
