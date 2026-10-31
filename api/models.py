"""Validated HTTP request and response contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SearchQuery(BaseModel):
    bid_id: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)
    doc_type: str | None = Field(default=None, max_length=80)
    addendum_number: int | None = Field(default=None, ge=1, le=999)
    file_name: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def candidates_cover_results(self) -> "SearchQuery":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class AskRequest(BaseModel):
    bid_id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=2, max_length=2000)
    use_cache: bool = True


class ExtractRequest(BaseModel):
    bid_id: str = Field(min_length=1, max_length=80)
    resume: bool = True
    max_retries: int = Field(default=1, ge=0, le=2)
    allow_query_rewrite: bool = True
    semantic_validation: bool = True


class CompareRequest(BaseModel):
    bid_ids: list[str] = Field(min_length=2, max_length=10)
    fields: list[str] | None = Field(default=None, max_length=20)
    format: Literal["json", "markdown"] = "json"

    @field_validator("bid_ids")
    @classmethod
    def unique_bids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("bid_ids must be unique")
        return value


class JobView(BaseModel):
    job_id: str
    kind: Literal["index", "extract"]
    bid_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: int = Field(default=0, ge=0, le=100)
    run_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    activity_id: str | None = None
