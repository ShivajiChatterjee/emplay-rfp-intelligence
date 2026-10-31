import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from agents.llm_client import (
    GeminiClient,
    GeminiConfigurationError,
    GeminiGenerationError,
    GeminiSettings,
    GenerationSettings,
)

TEST_MODEL = "gemini-unit-test-model"


class SyntheticAnswer(BaseModel):
    value: str
    supported: bool


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeSdkClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response=response, error=error)


def settings():
    return GeminiSettings(api_key="unit-test-key", model=TEST_MODEL)


class GeminiClientTests(unittest.TestCase):
    def test_structured_generation_uses_required_model_and_usage(self):
        response = SimpleNamespace(
            parsed=SyntheticAnswer(value="Synthetic", supported=True),
            text=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=7,
                candidates_token_count=3,
                thoughts_token_count=2,
                total_token_count=10,
            ),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )
        sdk = FakeSdkClient(response=response)
        client = GeminiClient(settings=settings(), sdk_client=sdk)

        result = client.generate_structured(
            "Return the supplied synthetic value.",
            SyntheticAnswer,
            system_instruction="Use only supplied synthetic data.",
            generation=GenerationSettings(max_output_tokens=64),
        )

        self.assertEqual(result.output.value, "Synthetic")
        self.assertEqual(result.model, TEST_MODEL)
        self.assertEqual(result.usage.total_tokens, 10)
        self.assertEqual(result.usage.thought_tokens, 2)
        self.assertEqual(result.finish_reason, "STOP")
        self.assertGreaterEqual(result.latency_ms, 0)

        call = sdk.models.calls[0]
        self.assertEqual(call["model"], TEST_MODEL)
        self.assertEqual(call["config"].response_mime_type, "application/json")
        self.assertIs(call["config"].response_schema, SyntheticAnswer)
        self.assertEqual(call["config"].max_output_tokens, 64)

    def test_json_text_fallback_is_validated(self):
        response = SimpleNamespace(
            parsed=None,
            text='{"value":"Fallback","supported":true}',
            usage_metadata=None,
        )
        client = GeminiClient(
            settings=settings(),
            sdk_client=FakeSdkClient(response=response),
        )

        result = client.generate_structured("Synthetic prompt", SyntheticAnswer)
        self.assertEqual(result.output.value, "Fallback")
        self.assertIsNone(result.usage.total_tokens)

    def test_provider_errors_are_clear_and_do_not_change_models(self):
        sdk = FakeSdkClient(error=RuntimeError("quota unavailable"))
        client = GeminiClient(settings=settings(), sdk_client=sdk)

        with self.assertRaises(GeminiGenerationError) as raised:
            client.generate_structured("Synthetic prompt", SyntheticAnswer)

        self.assertEqual(raised.exception.classification, "quota_or_rate_limit")

        self.assertEqual(len(sdk.models.calls), 1)

    def test_truncated_json_is_classified_from_finish_reason(self):
        response = SimpleNamespace(
            parsed=None,
            text='{"value":"unfinished',
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
            usage_metadata=SimpleNamespace(
                prompt_token_count=20,
                candidates_token_count=8,
                thoughts_token_count=5,
                total_token_count=33,
            ),
        )
        client = GeminiClient(
            settings=settings(),
            sdk_client=FakeSdkClient(response=response),
        )
        with self.assertRaises(GeminiGenerationError) as raised:
            client.generate_structured("Synthetic prompt", SyntheticAnswer)
        error = raised.exception
        self.assertEqual(error.classification, "truncated_output")
        self.assertEqual(error.finish_reason, "MAX_TOKENS")
        self.assertEqual(error.usage.thought_tokens, 5)
        self.assertGreater(error.response_text_characters, 0)

    def test_malformed_json_and_schema_failure_are_distinct(self):
        malformed = SimpleNamespace(
            parsed=None,
            text='{"value":',
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=None,
        )
        client = GeminiClient(
            settings=settings(),
            sdk_client=FakeSdkClient(response=malformed),
        )
        with self.assertRaises(GeminiGenerationError) as malformed_error:
            client.generate_structured("Synthetic prompt", SyntheticAnswer)
        self.assertEqual(malformed_error.exception.classification, "malformed_json")

        invalid_schema = SimpleNamespace(
            parsed={"value": "missing supported"},
            text=None,
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=None,
        )
        client = GeminiClient(
            settings=settings(),
            sdk_client=FakeSdkClient(response=invalid_schema),
        )
        with self.assertRaises(GeminiGenerationError) as schema_error:
            client.generate_structured("Synthetic prompt", SyntheticAnswer)
        self.assertEqual(schema_error.exception.classification, "schema_validation")

    def test_environment_settings_require_key_and_explicit_model(self):
        with patch("agents.llm_client.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(GeminiConfigurationError):
                    GeminiSettings.from_environment()

            with patch.dict(
                os.environ,
                {
                    "GOOGLE_API_KEY": "secret",
                },
                clear=True,
            ):
                with self.assertRaises(GeminiConfigurationError):
                    GeminiSettings.from_environment()

            with patch.dict(
                os.environ,
                {
                    "GOOGLE_API_KEY": "secret",
                    "GEMINI_MODEL": "gemini-explicit-test-model",
                },
                clear=True,
            ):
                loaded = GeminiSettings.from_environment()
                self.assertEqual(loaded.model, "gemini-explicit-test-model")
                self.assertEqual(
                    loaded.api_key.get_secret_value(),
                    "secret",
                )

    def test_client_honors_non_default_configured_model_without_fallback(self):
        response = SimpleNamespace(
            parsed=SyntheticAnswer(value="Configured", supported=True),
            text=None,
            usage_metadata=None,
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )
        sdk = FakeSdkClient(response=response)
        client = GeminiClient(
            settings=GeminiSettings(
                api_key="unit-test-key", model="gemini-3.1-flash-lite"
            ),
            sdk_client=sdk,
        )
        result = client.generate_structured("Synthetic prompt", SyntheticAnswer)
        self.assertEqual(result.model, "gemini-3.1-flash-lite")
        self.assertEqual(sdk.models.calls[0]["model"], "gemini-3.1-flash-lite")


if __name__ == "__main__":
    unittest.main()
