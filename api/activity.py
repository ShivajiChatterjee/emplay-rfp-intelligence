"""Persistent application-originated activity records.

This ledger is intentionally separate from workflow manifests.  It records
operations initiated through FastAPI without altering or deleting historical
CLI artifacts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActivityStore:
    def __init__(self, path: Path | None = None):
        default_path = Path(os.getenv("EMPLAY_OUTPUT_DIR", "outputs")) / "api_activity.sqlite3"
        self.path = path or Path(os.getenv("EMPLAY_ACTIVITY_PATH", str(default_path)))
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """CREATE TABLE IF NOT EXISTS activities (
                activity_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                bid_ids_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT,
                workflow_run_id TEXT,
                job_id TEXT,
                metrics_available INTEGER NOT NULL DEFAULT 0,
                trace_available INTEGER NOT NULL DEFAULT 0,
                cache_hit INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                details_json TEXT NOT NULL DEFAULT '{}'
            )"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_activity_updated ON activities(updated_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_activity_workflow ON activities(workflow_run_id)"
        )
        return connection

    def create(
        self,
        operation: str,
        bid_ids: list[str],
        *,
        status: str = "queued",
        workflow_run_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        activity_id = str(uuid.uuid4())
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO activities (
                    activity_id, operation, bid_ids_json, status, created_at,
                    updated_at, workflow_run_id, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    activity_id,
                    operation,
                    json.dumps(bid_ids),
                    status,
                    now,
                    now,
                    workflow_run_id,
                    json.dumps(details or {}, ensure_ascii=False),
                ),
            )
        return self.get(activity_id) or {}

    def update(self, activity_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {
            "status", "finished_at", "workflow_run_id", "job_id",
            "metrics_available", "trace_available", "cache_hit", "error",
            "details",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported activity fields: {sorted(unknown)}")
        assignments = ["updated_at = ?"]
        values: list[Any] = [_now()]
        for key, value in changes.items():
            column = "details_json" if key == "details" else key
            if key == "details":
                value = json.dumps(value or {}, ensure_ascii=False)
            if key in {"metrics_available", "trace_available", "cache_hit"}:
                value = int(bool(value))
            assignments.append(f"{column} = ?")
            values.append(value)
        values.append(activity_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE activities SET {', '.join(assignments)} WHERE activity_id = ?",
                values,
            )
        record = self.get(activity_id)
        if record is None:
            raise KeyError("Activity not found")
        return record

    def get(self, identifier: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM activities
                   WHERE activity_id = ? OR workflow_run_id = ?
                   ORDER BY updated_at DESC LIMIT 1""",
                (identifier, identifier),
            ).fetchone()
        return self._decode(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        bounded = max(1, min(limit, 200))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM activities ORDER BY updated_at DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def workflow_run_ids(self) -> set[str]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT workflow_run_id FROM activities WHERE workflow_run_id IS NOT NULL"
            ).fetchall()
        return {row[0] for row in rows}

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["bid_ids"] = json.loads(record.pop("bid_ids_json"))
        record["details"] = json.loads(record.pop("details_json") or "{}")
        for key in ("metrics_available", "trace_available", "cache_hit"):
            record[key] = bool(record[key])
        record["source"] = "application"
        record["inspection_id"] = record["activity_id"]
        return record


activity_store = ActivityStore()
