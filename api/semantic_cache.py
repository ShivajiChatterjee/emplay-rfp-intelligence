"""Conservative, bid-scoped cache for verified grounded Q&A answers."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable


NEGATIONS = {"no", "not", "never", "without", "except", "exclude", "excluding"}
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._/-]*", re.IGNORECASE)


def _tokens(question: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(question.casefold()))


def _qualifiers(question: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tokens = _tokens(question)
    numeric = tuple(sorted({token for token in tokens if any(char.isdigit() for char in token)}))
    negatives = tuple(sorted(set(tokens) & NEGATIONS))
    return numeric, negatives


def questions_match(left: str, right: str, threshold: float = 0.92) -> bool:
    """Require close semantics and exact material qualifiers before reuse."""

    if _qualifiers(left) != _qualifiers(right):
        return False
    left_tokens, right_tokens = set(_tokens(left)), set(_tokens(right))
    if not left_tokens or not right_tokens:
        return False
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    sequence = difflib.SequenceMatcher(None, " ".join(_tokens(left)), " ".join(_tokens(right))).ratio()
    return jaccard >= 0.80 and sequence >= threshold


class SemanticAnswerCache:
    def __init__(self, path: Path | None = None, ttl_seconds: int | None = None, max_entries: int | None = None):
        self.path = path or Path(os.getenv("EMPLAY_CACHE_PATH", "outputs/cache/qa_cache.sqlite3"))
        configured_ttl = ttl_seconds or int(os.getenv("EMPLAY_CACHE_TTL_SECONDS", "86400"))
        configured_entries = max_entries or int(os.getenv("EMPLAY_CACHE_MAX_ENTRIES", "500"))
        self.ttl_seconds = max(60, min(configured_ttl, 30 * 24 * 60 * 60))
        self.max_entries = max(10, min(configured_entries, 5_000))
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute(
            """CREATE TABLE IF NOT EXISTS answers (
                cache_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                question TEXT NOT NULL,
                answer_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                source_latency_ms REAL
            )"""
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_answers_scope ON answers(scope)")
        return connection

    @staticmethod
    def scope(bid_id: str, index_version: str, retrieval_config: dict[str, Any]) -> str:
        payload = json.dumps(
            {"bid_id": bid_id, "index_version": index_version, "retrieval": retrieval_config},
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"{bid_id}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def get(
        self,
        scope: str,
        question: str,
        *,
        evidence_is_fresh: Callable[[dict[str, Any]], bool],
    ) -> dict[str, Any] | None:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM answers WHERE expires_at <= ?", (now,))
            rows = connection.execute(
                "SELECT cache_id, question, answer_json, source_latency_ms FROM answers WHERE scope = ?",
                (scope,),
            ).fetchall()
            for cache_id, saved_question, answer_json, source_latency_ms in rows:
                if not questions_match(question, saved_question):
                    continue
                answer = json.loads(answer_json)
                if not evidence_is_fresh(answer):
                    connection.execute("DELETE FROM answers WHERE cache_id = ?", (cache_id,))
                    continue
                self._hits += 1
                return {
                    "answer": answer,
                    "cache": {
                        "hit": True,
                        "source_latency_ms": source_latency_ms,
                        "latency_saved_ms": source_latency_ms,
                    },
                }
            self._misses += 1
            return None

    def put(self, scope: str, question: str, answer: dict[str, Any], source_latency_ms: float) -> None:
        if not answer.get("answer") or not answer.get("sources"):
            return
        now = time.time()
        cache_id = hashlib.sha256(f"{scope}\0{question}\0{now}".encode("utf-8")).hexdigest()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO answers VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cache_id,
                    scope,
                    question,
                    json.dumps(answer, ensure_ascii=False),
                    now,
                    now + self.ttl_seconds,
                    source_latency_ms,
                ),
            )
            overflow = connection.execute("SELECT COUNT(*) FROM answers").fetchone()[0] - self.max_entries
            if overflow > 0:
                connection.execute(
                    "DELETE FROM answers WHERE cache_id IN (SELECT cache_id FROM answers ORDER BY created_at ASC LIMIT ?)",
                    (overflow,),
                )

    def invalidate_bid(self, bid_id: str) -> int:
        """Remove only one bid's entries after its index changes."""

        with self._lock, self._connect() as connection:
            pattern = f"{bid_id}:%"
            count = connection.execute(
                "SELECT COUNT(*) FROM answers WHERE scope LIKE ?", (pattern,)
            ).fetchone()[0]
            connection.execute("DELETE FROM answers WHERE scope LIKE ?", (pattern,))
            return count

    def stats(self) -> dict[str, int]:
        with self._lock, self._connect() as connection:
            entries = connection.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        return {"hits": self._hits, "misses": self._misses, "entries": entries}


answer_cache = SemanticAnswerCache()
