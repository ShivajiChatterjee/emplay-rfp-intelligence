"""Safe normalization of historical and current trace schema variants."""

from __future__ import annotations

from typing import Any


SENSITIVE_KEYS = {
    "api_key", "apikey", "authorization", "credential", "credentials",
    "password", "secret", "access_token", "refresh_token", "cookie",
    "cookies", "environment", "env", "headers", "raw_payload",
}


def safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in SENSITIVE_KEYS or normalized.endswith("_api_key"):
                result[key] = "[redacted]"
            else:
                result[key] = safe_value(item)
        return result
    if isinstance(value, list):
        return [safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [safe_value(item) for item in value]
    return value


def _trace_step(item: dict[str, Any], order: int, source: str) -> dict[str, Any]:
    details = {
        key: item.get(key)
        for key in (
            "input_summary", "output_summary", "metadata", "tool_calls",
            "summary", "error", "field_name", "tokens_used",
        )
        if item.get(key) not in (None, {}, [])
    }
    return {
        "order": order,
        "source": source,
        "agent": item.get("agent") or "workflow",
        "operation": item.get("action") or item.get("operation") or "recorded_step",
        "status": item.get("status") or "recorded",
        "timestamp": item.get("started_at") or item.get("timestamp"),
        "duration_ms": item.get("latency_ms") or item.get("duration_ms"),
        "details": safe_value(details),
    }


def normalize_trace(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    traces = [item for item in payload.get("agent_traces", []) if isinstance(item, dict)]
    events = [item for item in payload.get("events", []) if isinstance(item, dict)]
    # Agent traces contain duration, timestamp and tool metadata.  Older traces
    # sometimes contain only a planner trace, in which case workflow events are
    # the more complete execution-order record.
    primary = traces if len(traces) > 1 or not events else events
    source = "agent_trace" if primary is traces else "workflow_event"
    steps = [_trace_step(item, index + 1, source) for index, item in enumerate(primary)]
    workflow_events = [
        _trace_step(item, index + 1, "workflow_event")
        for index, item in enumerate(events)
    ]
    agent_traces = [
        _trace_step(item, index + 1, "agent_trace")
        for index, item in enumerate(traces)
    ]
    context = safe_value({
        "retry_counts": payload.get("retry_counts", {}),
        "validation": payload.get("validation", {}),
        "field_statuses": payload.get("field_statuses", {}),
        "retrieval_sources": payload.get("retrieval_sources", {}),
        "errors": payload.get("errors", []),
        "warnings": payload.get("warnings", []),
        "operational_status_counts": payload.get("operational_status_counts", {}),
    })
    return {
        "run_id": run_id,
        "available": bool(steps),
        "step_count": len(steps),
        "steps": steps,
        "agent_traces": agent_traces,
        "workflow_events": workflow_events,
        "context": context,
        "reason": None if steps else "No execution trace was recorded for this run.",
    }
