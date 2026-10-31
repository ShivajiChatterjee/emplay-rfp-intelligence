"""Bounded in-process background jobs for indexing and extraction."""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobQueueFull(RuntimeError):
    pass


class JobManager:
    def __init__(self, max_workers: int | None = None, max_jobs: int | None = None):
        workers = max_workers or max(1, min(int(os.getenv("EMPLAY_API_JOB_WORKERS", "2")), 4))
        capacity = max_jobs or max(workers, min(int(os.getenv("EMPLAY_API_MAX_JOBS", "8")), 32))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="emplay-job")
        self._slots = threading.BoundedSemaphore(capacity)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def submit(
        self,
        kind: str,
        bid_id: str,
        task: Callable[[], dict[str, Any]],
        *,
        activity_id: str | None = None,
    ) -> dict[str, Any]:
        if not self._slots.acquire(blocking=False):
            raise JobQueueFull("Background job capacity is full")
        job_id = str(uuid.uuid4())
        record = {
            "job_id": job_id,
            "kind": kind,
            "bid_id": bid_id,
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "progress": 0,
            "run_id": None,
            "result": None,
            "error": None,
            "activity_id": activity_id,
        }
        with self._lock:
            self._jobs[job_id] = record
        self._executor.submit(self._run, job_id, task)
        return dict(record)

    def _run(self, job_id: str, task: Callable[[], dict[str, Any]]) -> None:
        try:
            self._update(job_id, status="running", started_at=_now(), progress=10)
            result = task()
            self._update(
                job_id,
                status="succeeded",
                finished_at=_now(),
                progress=100,
                result=result,
                run_id=result.get("run_id"),
            )
        except Exception as exc:  # the safe API message is intentionally generic
            self._update(
                job_id,
                status="failed",
                finished_at=_now(),
                progress=100,
                error=f"{type(exc).__name__}: background operation failed",
            )
        finally:
            self._slots.release()

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record else None


jobs = JobManager()
