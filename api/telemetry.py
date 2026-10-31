"""Honest local telemetry aggregation for workflow runs."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class TelemetryStore:
    def __init__(self, root: Path | None = None):
        self.root = root or Path(os.getenv("EMPLAY_TELEMETRY_DIR", "outputs/telemetry"))
        self._lock = threading.Lock()

    def record(
        self,
        run_id: str,
        *,
        mode: str,
        bid_ids: list[str],
        latency_ms: float,
        state: dict[str, Any],
        cache_hit: bool = False,
    ) -> dict[str, Any]:
        traces = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in state.get("agent_traces", [])]
        usages = [
            item.get("metadata", {}).get("token_usage", {})
            for item in traces
            if isinstance(item.get("metadata"), dict)
        ]

        def total(key: str) -> int | None:
            values = [usage.get(key) for usage in usages if usage.get(key) is not None]
            return sum(values) if values else None

        prompt_tokens = total("prompt_tokens")
        output_tokens = total("output_tokens")
        thought_tokens = total("thought_tokens")
        known_total = total("total_tokens")
        if known_total is None:
            trace_totals = [item.get("tokens_used") for item in traces if item.get("tokens_used") is not None]
            known_total = sum(trace_totals) if trace_totals else None
        input_rate = self._optional_price("GEMINI_INPUT_COST_PER_MILLION")
        output_rate = self._optional_price("GEMINI_OUTPUT_COST_PER_MILLION")
        thinking_rate = self._optional_price("GEMINI_THINKING_COST_PER_MILLION")
        can_estimate = (
            prompt_tokens is not None
            and output_tokens is not None
            and input_rate is not None
            and output_rate is not None
            and (thought_tokens in {None, 0} or thinking_rate is not None)
        )
        estimated_cost = None
        if can_estimate:
            estimated_cost = (
                prompt_tokens * input_rate
                + output_tokens * output_rate
                + (thought_tokens or 0) * (thinking_rate or 0)
            ) / 1_000_000
        provider_errors = [item for item in traces if item.get("status") == "error"]
        payload = {
            "run_id": run_id,
            "mode": mode,
            "bid_ids": bid_ids,
            "request_latency_ms": round(latency_ms, 2),
            "cache": {"hit": cache_hit},
            "usage": {
                "gemini": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "thinking_tokens": thought_tokens,
                    "total_tokens": known_total,
                    "note": None if prompt_tokens is not None else "Provider usage unavailable in recorded traces",
                },
                "cohere": {
                    "calls": sum(
                        "cohere" in call.casefold()
                        for item in traces
                        for call in item.get("tool_calls", [])
                    ),
                    "usage": None,
                },
                "embeddings": {"usage": None},
            },
            "estimated_cost": {
                "amount": round(estimated_cost, 8) if estimated_cost is not None else None,
                "currency": os.getenv("EMPLAY_PRICING_CURRENCY", "USD"),
                "pricing_date": os.getenv("EMPLAY_PRICING_DATE") or None,
                "is_estimate": True,
                "reason": (
                    None
                    if estimated_cost is not None
                    else "Token usage or configured per-million pricing is unavailable; cost was not invented"
                ),
            },
            "provider_errors": len(provider_errors),
            "retries": sum(state.get("retry_counts", {}).values()),
            "agents": [
                {
                    "agent": item.get("agent"),
                    "action": item.get("action"),
                    "latency_ms": item.get("latency_ms"),
                    "status": item.get("status"),
                    "tokens_used": item.get("tokens_used"),
                }
                for item in traces
            ],
        }
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{run_id}.json"
        with self._lock:
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    @staticmethod
    def _optional_price(name: str) -> float | None:
        text = os.getenv(name, "").strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if value >= 0 else None

    def get(self, run_id: str) -> dict[str, Any] | None:
        target = self.root / f"{run_id}.json"
        if not target.is_file():
            return None
        return json.loads(target.read_text(encoding="utf-8"))


telemetry = TelemetryStore()


class Timer:
    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.latency_ms = (time.perf_counter() - self.started) * 1000
