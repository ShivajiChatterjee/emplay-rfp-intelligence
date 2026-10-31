import json
import unittest

from pydantic import ValidationError

from agents.field_specs import FIELD_SPECS, OFFICIAL_FIELD_LABELS
from agents.schemas import (
    Evidence,
    ExtractedField,
    ExtractionRequest,
    FinalExtractionOutput,
    FinalFieldRecord,
    SourceCitation,
    ValidationSummary,
    WorkflowState,
    merge_state_mappings,
)


class SchemaTests(unittest.TestCase):
    def test_existing_evidence_and_extracted_field_contracts_remain_valid(self):
        evidence = Evidence(
            file_name="solicitation.pdf",
            page_number=2,
            quote="Responses are due at 2:00 PM.",
            chunk_id="chunk-1",
            doc_type="rfp",
        )
        field = ExtractedField(
            field_name="Due Date",
            status="found",
            value="2:00 PM",
            confidence=0.8,
            citations=[evidence],
        )
        self.assertEqual(field.citations[0].chunk_id, "chunk-1")

    def test_extraction_request_validates_limits_and_uniqueness(self):
        request = ExtractionRequest(
            bid_id="unseen-solicitation",
            field_specs=FIELD_SPECS,
            top_k=5,
            candidate_k=20,
        )
        self.assertEqual(len(request.field_specs), 20)

        with self.assertRaises(ValidationError):
            ExtractionRequest(
                bid_id="example",
                field_specs=(FIELD_SPECS[0], FIELD_SPECS[0]),
            )

        with self.assertRaises(ValidationError):
            ExtractionRequest(
                bid_id="example",
                field_specs=(FIELD_SPECS[0],),
                top_k=6,
                candidate_k=5,
            )

    def test_final_output_serializes_required_shape(self):
        records = {
            label: FinalFieldRecord(
                value=None,
                sources=[],
                confidence=0.0,
                notes="Not found in retrieved evidence.",
            )
            for label in OFFICIAL_FIELD_LABELS
        }
        records["Bid Number"] = FinalFieldRecord(
            value="RFP-123",
            sources=[SourceCitation(file="solicitation.pdf", page=1)],
            confidence=0.95,
            notes="Directly stated on the cover page.",
        )

        output = FinalExtractionOutput(
            bid_id="example",
            fields=records,
            validation=ValidationSummary(
                passed=1,
                failed=0,
                not_found=19,
            ),
        )
        payload = json.loads(output.model_dump_json())

        self.assertEqual(list(payload["fields"]), list(OFFICIAL_FIELD_LABELS))
        self.assertEqual(
            payload["fields"]["Bid Number"]["sources"],
            [{"file": "solicitation.pdf", "page": 1}],
        )
        self.assertIn("addendum_changes", payload)
        self.assertEqual(payload["validation"]["not_found"], 19)

    def test_final_output_rejects_missing_fields_or_bad_counts(self):
        with self.assertRaises(ValidationError):
            FinalExtractionOutput(
                bid_id="example",
                fields={"Bid Number": FinalFieldRecord()},
                validation=ValidationSummary(not_found=1),
            )

        records = {
            label: FinalFieldRecord() for label in OFFICIAL_FIELD_LABELS
        }
        with self.assertRaises(ValidationError):
            FinalExtractionOutput(
                bid_id="example",
                fields=records,
                validation=ValidationSummary(not_found=19),
            )

    def test_shared_state_is_serializable_and_has_merge_reducer(self):
        state = WorkflowState(
            bid_id="example",
            field_specs=FIELD_SPECS,
            evidence={"Title": []},
        )
        restored = WorkflowState.model_validate_json(state.model_dump_json())
        self.assertEqual(restored.bid_id, "example")
        self.assertEqual(len(restored.field_specs), 20)
        self.assertEqual(
            merge_state_mappings({"Title": 1}, {"Due Date": 2}),
            {"Title": 1, "Due Date": 2},
        )


if __name__ == "__main__":
    unittest.main()
