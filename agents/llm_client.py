"""Native Google GenAI client for bounded, structured Gemini generation."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Generic, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, SecretStr, ValidationError
from providers.rate_limit import (
    gemini_limiter,
    is_daily_exhausted,
    retry_after_seconds,
)


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class GeminiConfigurationError(RuntimeError):
    """Raised when required Gemini configuration is absent or unsafe."""


class GeminiGenerationError(RuntimeError):
    """Sanitized provider/parse failure with actionable telemetry."""

    def __init__(
        self,
        message: str,
        *,
        classification: str,
        finish_reason: str | None = None,
        usage: "TokenUsage | None" = None,
        latency_ms: float = 0,
        response_text_characters: int = 0,
        retry_after: float | None = None,
        daily_quota_exhausted: bool = False,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.finish_reason = finish_reason
        self.usage = usage or TokenUsage()
        self.latency_ms = latency_ms
        self.response_text_characters = response_text_characters
        self.retry_after = retry_after
        self.daily_quota_exhausted = daily_quota_exhausted


class GeminiSettings(BaseModel):
    api_key: SecretStr
    model: str = Field(min_length=1)
    timeout_ms: int = Field(default=60_000, ge=1_000, le=300_000)

    @classmethod
    def from_environment(cls) -> "GeminiSettings":
        load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        model = os.getenv("GEMINI_MODEL", "").strip()

        if not api_key:
            raise GeminiConfigurationError(
                "GOOGLE_API_KEY must be configured in the environment"
            )
        if not model:
            raise GeminiConfigurationError(
                "GEMINI_MODEL must explicitly select a Gemini model"
            )

        timeout_text = os.getenv("GEMINI_TIMEOUT_MS", "60000").strip()
        try:
            timeout_ms = int(timeout_text)
        except ValueError as exc:
            raise GeminiConfigurationError(
                "GEMINI_TIMEOUT_MS must be an integer"
            ) from exc

        return cls(
            api_key=SecretStr(api_key),
            model=model,
            timeout_ms=timeout_ms,
        )


class GenerationSettings(BaseModel):
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_output_tokens: int = Field(default=1_024, ge=1, le=4_096)
    seed: int | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    thought_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class StructuredGenerationResult(BaseModel, Generic[StructuredModel]):
    output: StructuredModel
    model: str
    latency_ms: float = Field(ge=0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
    validation_status: str = "passed"


def _attribute(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _finish_reason(response: Any) -> str | None:
    candidates = _attribute(response, "candidates") or []
    if not candidates:
        return None
    reason = _attribute(candidates[0], "finish_reason")
    if reason is None:
        return None
    if hasattr(reason, "name"):
        return str(reason.name)
    text = str(reason)
    return text.rsplit(".", 1)[-1]


def _usage(response: Any) -> TokenUsage:
    metadata = _attribute(response, "usage_metadata")
    return TokenUsage(
        prompt_tokens=_attribute(metadata, "prompt_token_count"),
        output_tokens=_attribute(metadata, "candidates_token_count"),
        thought_tokens=_attribute(metadata, "thoughts_token_count"),
        total_tokens=_attribute(metadata, "total_token_count"),
    )


def _provider_failure_classification(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".casefold()
    if "timeout" in text or "deadline" in text:
        return "timeout"
    if any(item in text for item in ("quota", "rate limit", "429", "resource_exhausted")):
        return "quota_or_rate_limit"
    if any(item in text for item in ("503", "504", "unavailable", "gateway timeout")):
        return "transient_unavailable"
    if any(item in text for item in ("api key", "unauth", "permission_denied", "401", "403")):
        return "authentication"
    if "additionalproperties" in text or "response_schema" in text:
        return "schema_validation"
    return "transport"


def _parse_failure_classification(
    exc: Exception,
    finish_reason: str | None,
) -> str:
    if finish_reason and "MAX_TOKENS" in finish_reason.upper():
        return "truncated_output"
    if isinstance(exc, ValidationError):
        error_types = {item.get("type") for item in exc.errors()}
        if "json_invalid" in error_types:
            return "malformed_json"
        return "schema_validation"
    if isinstance(exc, ValueError):
        return "malformed_json"
    return "schema_validation"


class GeminiClient:
    """One provider-specific boundary with injectable SDK transport for tests."""

    def __init__(
        self,
        settings: GeminiSettings | None = None,
        sdk_client: Any | None = None,
        rate_limiter: Any | None = None,
    ) -> None:
        self.settings = settings or GeminiSettings.from_environment()

        self._client = sdk_client or genai.Client(
            api_key=self.settings.api_key.get_secret_value(),
            http_options=types.HttpOptions(
                timeout=self.settings.timeout_ms,
                # Application-level limiter owns retries so SDK retries cannot
                # multiply calls across concurrent field workers.
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._rate_limiter = rate_limiter or (gemini_limiter() if sdk_client is None else None)

    @property
    def model(self) -> str:
        return self.settings.model

    def generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredModel],
        *,
        system_instruction: str | None = None,
        generation: GenerationSettings | None = None,
    ) -> StructuredGenerationResult[StructuredModel]:
        """Generate and validate one Pydantic response without provider fallback."""

        if not prompt.strip():
            raise ValueError("prompt cannot be blank")
        if not isinstance(response_model, type) or not issubclass(
            response_model, BaseModel
        ):
            raise TypeError("response_model must be a Pydantic BaseModel class")

        settings = generation or GenerationSettings()
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_output_tokens=settings.max_output_tokens,
            seed=settings.seed,
            response_mime_type="application/json",
            response_schema=response_model,
        )

        started = time.perf_counter()
        trace_context = None
        trace_run = None
        if os.getenv("LANGSMITH_TRACING", "").casefold() == "true":
            try:
                from langsmith import trace

                trace_context = trace(
                    "gemini_structured_generation",
                    run_type="llm",
                    inputs={
                        "prompt_characters": len(prompt),
                        "system_instruction_characters": len(
                            system_instruction or ""
                        ),
                        "response_schema": response_model.__name__,
                    },
                    metadata={
                        "model": self.model,
                        "max_output_tokens": settings.max_output_tokens,
                    },
                    tags=["gemini", "structured-output"],
                )
                trace_run = trace_context.__enter__()
            except Exception:
                # Observability must never make the primary workflow fail.
                trace_context = None
                trace_run = None

        def fail(
            exc: Exception,
            *,
            classification: str,
            finish_reason: str | None = None,
            usage: TokenUsage | None = None,
            response_text_characters: int = 0,
        ) -> None:
            latency_ms = round((time.perf_counter() - started) * 1_000, 2)
            telemetry = {
                "model": self.model,
                "latency_ms": latency_ms,
                "finish_reason": finish_reason,
                "token_usage": (usage or TokenUsage()).model_dump(),
                "response_schema": response_model.__name__,
                "validation_status": "failed",
                "failure_classification": classification,
                "response_text_characters": response_text_characters,
            }
            if trace_run is not None:
                try:
                    trace_run.end(error=classification, outputs=telemetry)
                except Exception:
                    pass
            if trace_context is not None:
                try:
                    trace_context.__exit__(type(exc), exc, exc.__traceback__)
                except Exception:
                    pass
            raise GeminiGenerationError(
                f"Structured generation with {self.model} failed: "
                f"{classification}; finish_reason={finish_reason or 'unknown'}",
                classification=classification,
                finish_reason=finish_reason,
                usage=usage,
                latency_ms=latency_ms,
                response_text_characters=response_text_characters,
                retry_after=retry_after_seconds(exc),
                daily_quota_exhausted=is_daily_exhausted(exc),
            ) from exc

        try:
            operation = lambda: self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
            response = (
                self._rate_limiter.call(
                    "gemini", operation,
                    retryable=lambda exc: _provider_failure_classification(exc)
                    in {
                        "quota_or_rate_limit", "timeout", "transport",
                        "transient_unavailable",
                    },
                )
                if self._rate_limiter is not None
                else operation()
            )
        except Exception as exc:
            fail(
                exc,
                classification=_provider_failure_classification(exc),
            )

        finish_reason = _finish_reason(response)
        usage = _usage(response)
        response_text = _attribute(response, "text")
        response_text_characters = len(str(response_text or ""))
        try:
            parsed = _attribute(response, "parsed")
            if isinstance(parsed, response_model):
                output = parsed
            elif parsed is not None:
                output = response_model.model_validate(parsed)
            else:
                if not response_text or not str(response_text).strip():
                    raise ValueError("Gemini returned no structured response")
                output = response_model.model_validate_json(response_text)
        except Exception as exc:
            fail(
                exc,
                classification=_parse_failure_classification(
                    exc,
                    finish_reason,
                ),
                finish_reason=finish_reason,
                usage=usage,
                response_text_characters=response_text_characters,
            )

        latency_ms = (time.perf_counter() - started) * 1_000

        if trace_run is not None:
            try:
                trace_run.end(
                    outputs={
                        "model": self.model,
                        "latency_ms": round(latency_ms, 2),
                        "token_usage": usage.model_dump(),
                        "finish_reason": finish_reason,
                        "response_schema": response_model.__name__,
                        "validation_status": "passed",
                    }
                )
            except Exception:
                pass
        if trace_context is not None:
            try:
                trace_context.__exit__(None, None, None)
            except Exception:
                pass

        return StructuredGenerationResult[response_model](
            output=output,
            model=self.model,
            latency_ms=round(latency_ms, 2),
            usage=usage,
            finish_reason=finish_reason,
            validation_status="passed",
        )


@lru_cache(maxsize=1)
def get_gemini_client() -> GeminiClient:
    """Return the process-wide Gemini client configured from the environment."""

    return GeminiClient()


def configured_gemini_model() -> str:
    """Return the explicit model identity without reading or exposing the key."""
    load_dotenv()
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not model:
        raise GeminiConfigurationError(
            "GEMINI_MODEL must explicitly select a Gemini model"
        )
    return model
