import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class FakeGeminiClient:
    model = "gemini-2.5-flash-lite"

    def __init__(self):
        self.calls = []

    def generate_structured(
        self,
        prompt,
        response_model,
        **kwargs,
    ):
        self.calls.append((prompt, response_model, kwargs))
        return SimpleNamespace(
            output=response_model(
                value="Example Solicitation",
                evidence_rank=1,
                quote="Official title: Example Solicitation",
                reasoning="The title is directly stated.",
                confidence=0.91,
            ),
            usage=SimpleNamespace(total_tokens=12),
        )


class ExtractorTests(unittest.TestCase):
    def test_part_number_mapping_normalizes_to_list_without_losing_association(self):
        from agents.extractor import normalize_extracted_value

        normalized = normalize_extracted_value(
            "Part_no",
            {"Mobile Console": "ZX-4810", "Charging Cradle": "CR-22"},
        )
        self.assertEqual(normalized, [
            {"item": "Mobile Console", "part_number": "ZX-4810"},
            {"item": "Charging Cradle", "part_number": "CR-22"},
        ])

    def test_existing_extractor_uses_gemini_and_preserves_citation(self):
        # Isolate extractor behavior from Qdrant/Cohere imports. Those external
        # adapters have their own implementation and are not called here.
        retrieval_module = ModuleType("agents.retrieval_agent")
        retrieval_module.RetrievalRequest = type("RetrievalRequest", (), {})
        retrieval_module.RetrievalResult = type("RetrievalResult", (), {})
        retrieval_module.search_evidence = lambda request: None

        with patch.dict(
            sys.modules,
            {"agents.retrieval_agent": retrieval_module},
        ):
            sys.modules.pop("agents.extractor", None)
            extractor = importlib.import_module("agents.extractor")

        evidence = SimpleNamespace(
            file_name="solicitation.pdf",
            page_number=1,
            quote=(
                "Cover page. Official title: Example Solicitation. "
                "Additional content."
            ),
            chunk_id="chunk-title",
            doc_type="rfp",
            addendum_number=None,
        )
        retrieval = SimpleNamespace(
            bid_id="example",
            field_name="Title",
            question="What is the official title?",
            status="success",
            passages=[
                SimpleNamespace(rank=1, score=0.9, evidence=evidence)
            ],
        )
        client = FakeGeminiClient()

        field, trace = extractor.extract_field(
            retrieval,
            llm_client=client,
        )

        self.assertEqual(field.status, "found")
        self.assertEqual(field.value, "Example Solicitation")
        self.assertEqual(field.confidence, 0.91)
        self.assertEqual(field.citations[0].file_name, "solicitation.pdf")
        self.assertEqual(trace.tokens_used, 12)
        self.assertEqual(
            trace.tool_calls,
            ["google.genai.generate_content:gemini-2.5-flash-lite"],
        )
        self.assertIn("Official title", client.calls[0][0])
        self.assertIn(
            "Commercial and Legal Extraction Specialist",
            client.calls[0][2]["system_instruction"],
        )


if __name__ == "__main__":
    unittest.main()
