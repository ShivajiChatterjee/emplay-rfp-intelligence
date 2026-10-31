from types import SimpleNamespace
from unittest.mock import patch

from agents.field_specs import get_field_spec
from agents.reconciler import PRODUCT_PATCH_MAX_OUTPUT_TOKENS, reconcile_field
from agents.schemas import (
    Evidence,
    ExtractedField,
    ProductSpecificationPatch,
    ProductSpecificationReconciliationDecision,
)
from workflow.checkpoints import (
    load_checkpoint,
    reusable_checkpoint_state,
    save_field_checkpoint,
)
from agents.retrieval_agent import RetrievalRequest, search_evidence


class DecisionQueueClient:
    model = "offline-test-model"

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def generate_structured(self, prompt, response_model, **kwargs):
        self.calls.append((prompt, response_model, kwargs))
        decision = self.decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return SimpleNamespace(
            output=decision,
            usage=SimpleNamespace(
                total_tokens=25,
                model_dump=lambda: {"prompt_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            ),
            latency_ms=1,
        )


class TruncatedOutput(RuntimeError):
    classification = "truncated_output"


def base_field(category_names):
    base = Evidence(
        file_name="unseen-rfp.pdf",
        page_number=1,
        quote="Product specification schedule.",
        chunk_id="base",
        doc_type="rfp",
    )
    value = {
        name: {"Warranty": "1 year", "Interface": "Legacy"}
        for name in category_names
    }
    return (
        ExtractedField(
            field_name="Product Specification",
            status="found",
            value=value,
            confidence=0.9,
            citations=[base],
        ),
        base,
    )


def addendum(index, text, number=1):
    return Evidence(
        file_name=f"amendment-{number}.pdf",
        page_number=index,
        quote=text,
        chunk_id=f"a-{number}-{index}",
        doc_type="addendum",
        addendum_number=number,
    )


def decision(*patches):
    return ProductSpecificationReconciliationDecision(
        amended=bool(patches),
        patches=list(patches),
    )


def test_many_dynamic_categories_reconcile_in_small_batches_without_regeneration():
    categories = [f"Unseen Device Category {index}" for index in range(1, 13)]
    field, base = base_field(categories)
    amendments = [
        addendum(index, f"Product specifications are revised. Marker {index}.")
        for index in range(1, 8)
    ]
    amendments[0].quote += " Unseen Device Category 1 Warranty is revised to 3 years."
    amendments[3].quote += " Unseen Device Category 7 Interface is revised to USB-C."
    client = DecisionQueueClient([
        decision(ProductSpecificationPatch(
            operation="set_attribute", category=categories[0], attribute="Warranty",
            value="3 years", evidence_rank=1,
        )),
        decision(ProductSpecificationPatch(
            operation="set_attribute", category=categories[6], attribute="Interface",
            value="USB-C", evidence_rank=1,
        )),
        decision(),
    ])

    outcome = reconcile_field(
        bid_id="UnseenBid",
        field_spec=get_field_spec("Product Specification"),
        field=field,
        evidence=[base, *amendments],
        llm_client=client,
        search_addenda=False,
    )

    assert outcome.verified is True
    assert len(client.calls) == 3
    assert all("CURRENT VALUE" not in prompt for prompt, _, _ in client.calls)
    assert all(
        kwargs["generation"].max_output_tokens == PRODUCT_PATCH_MAX_OUTPUT_TOKENS
        for _, _, kwargs in client.calls
    )
    assert PRODUCT_PATCH_MAX_OUTPUT_TOKENS == 4_096
    assert list(outcome.field.value) == categories
    assert outcome.field.value[categories[0]]["Warranty"] == "3 years"
    assert outcome.field.value[categories[6]]["Interface"] == "USB-C"
    assert outcome.field.value[categories[1]] == field.value[categories[1]]


def test_wildcard_attribute_override_requires_explicit_every_category_language():
    categories = ["Field Sensor", "Control Gateway", "Portable Reader"]
    field, base = base_field(categories)
    amendment = addendum(
        1,
        "Product specifications are revised. Every product category Warranty must be OEM-backed.",
    )
    client = DecisionQueueClient([decision(ProductSpecificationPatch(
        operation="set_attribute", category="*", attribute="Warranty",
        value="OEM-backed", evidence_rank=1,
    ))])
    outcome = reconcile_field(
        bid_id="AnotherBid", field_spec=get_field_spec("Product Specification"),
        field=field, evidence=[base, amendment], llm_client=client, search_addenda=False,
    )
    assert outcome.verified is True
    assert all(value["Warranty"] == "OEM-backed" for value in outcome.field.value.values())


def test_truncated_patch_batch_fails_safely_and_preserves_original_structure():
    field, base = base_field(["Lab Analyzer", "Sample Processor"])
    amendment = addendum(1, "Product specifications are revised for the Lab Analyzer.")
    client = DecisionQueueClient([TruncatedOutput("finish_reason=MAX_TOKENS")])
    outcome = reconcile_field(
        bid_id="LabBid", field_spec=get_field_spec("Product Specification"),
        field=field, evidence=[base, amendment], llm_client=client, search_addenda=False,
    )
    assert outcome.verified is False
    assert outcome.failure_status == "reconciliation_incomplete"
    assert outcome.field.value == field.value
    assert "MAX_TOKENS" in outcome.warning
    assert len(client.calls) == 1
    assert outcome.traces[-1].metadata["will_subdivide"] is False


def test_second_batch_truncation_subdivides_and_merges_all_verified_patches():
    categories = ["Portable Scanner", "Docking Station", "Control Panel"]
    field, base = base_field(categories)
    amendments = [
        addendum(index, f"Product specification clarification {index}; no other change.")
        for index in range(1, 6)
    ]
    amendments[3].quote = (
        "Product specifications are revised. Portable Scanner Warranty is 4 years."
    )
    amendments[4].quote = (
        "Product specifications are revised. Docking Station Interface is USB4."
    )
    client = DecisionQueueClient([
        decision(),
        TruncatedOutput("finish_reason=MAX_TOKENS"),
        decision(ProductSpecificationPatch(
            operation="set_attribute", category="Portable Scanner", attribute="Warranty",
            value="4 years", evidence_rank=1,
        )),
        decision(ProductSpecificationPatch(
            operation="set_attribute", category="Docking Station", attribute="Interface",
            value="USB4", evidence_rank=1,
        )),
    ])

    outcome = reconcile_field(
        bid_id="UnseenHardwareBid",
        field_spec=get_field_spec("Product Specification"),
        field=field,
        evidence=[base, *amendments],
        llm_client=client,
        search_addenda=False,
    )

    assert outcome.verified is True
    assert len(client.calls) == 4
    assert outcome.field.value["Portable Scanner"]["Warranty"] == "4 years"
    assert outcome.field.value["Docking Station"]["Interface"] == "USB4"
    assert outcome.field.value["Control Panel"] == field.value["Control Panel"]
    assert list(outcome.field.value) == categories
    batch_labels = [trace.input_summary["batch"] for trace in outcome.traces]
    assert batch_labels == ["1", "2", "2.1", "2.2"]
    assert outcome.traces[1].metadata["will_subdivide"] is True


def test_subdivided_single_passage_truncation_is_bounded_and_unresolved():
    field, base = base_field(["Portable Scanner", "Docking Station"])
    amendments = [
        addendum(index, f"Product specifications are revised. Clarification {index}.")
        for index in range(1, 6)
    ]
    client = DecisionQueueClient([
        decision(),
        TruncatedOutput("batch 2 MAX_TOKENS"),
        TruncatedOutput("single passage MAX_TOKENS"),
    ])

    outcome = reconcile_field(
        bid_id="BoundedFailureBid",
        field_spec=get_field_spec("Product Specification"),
        field=field,
        evidence=[base, *amendments],
        llm_client=client,
        search_addenda=False,
    )

    assert outcome.verified is False
    assert outcome.failure_status == "reconciliation_incomplete"
    assert outcome.field.value == field.value
    assert len(client.calls) == 3
    assert "single-passage batch remains truncated" in outcome.warning
    assert outcome.traces[-1].input_summary["batch"] == "2.1"
    assert outcome.traces[-1].metadata["will_subdivide"] is False


def test_product_patch_schema_omits_repeated_explanations_and_confidence():
    schema = ProductSpecificationReconciliationDecision.model_json_schema()
    patch_properties = schema["$defs"]["ProductSpecificationPatch"]["properties"]
    assert set(patch_properties) == {
        "operation", "category", "attribute", "value", "evidence_rank",
    }
    assert "explanation" not in schema["properties"]


def test_unknown_category_patch_cannot_fabricate_or_drop_categories():
    field, base = base_field(["Vehicle Camera", "Dock Reader"])
    amendment = addendum(
        1,
        "Product specifications are revised. Imaginary Device Warranty is 9 years.",
    )
    client = DecisionQueueClient([decision(ProductSpecificationPatch(
        operation="set_attribute", category="Imaginary Device", attribute="Warranty",
        value="9 years", evidence_rank=1,
    ))])
    outcome = reconcile_field(
        bid_id="FleetBid", field_spec=get_field_spec("Product Specification"),
        field=field, evidence=[base, amendment], llm_client=client, search_addenda=False,
    )
    assert outcome.verified is False
    assert outcome.field.value == field.value
    assert "unknown category" in outcome.warning


def test_successful_product_checkpoint_preserves_unrelated_checkpoint(tmp_path):
    field, base = base_field(["Network Probe", "Edge Controller"])
    amendment = addendum(
        1,
        "Product specifications are revised. Network Probe Warranty is 2 years.",
    )
    client = DecisionQueueClient([decision(ProductSpecificationPatch(
        operation="set_attribute", category="Network Probe", attribute="Warranty",
        value="2 years", evidence_rank=1,
    ))])
    outcome = reconcile_field(
        bid_id="NetworkBid", field_spec=get_field_spec("Product Specification"),
        field=field, evidence=[base, amendment], llm_client=client, search_addenda=False,
    )
    path = tmp_path / "checkpoints.json"
    reusable = {"status": "passed", "terminal": True, "reusable": True, "reasons": [], "reconciliation_verified": True}
    decision_payload = {"field_name": "Title", "status": "passed", "reasons": [], "retry_required": False, "confidence": 1.0}
    title = ExtractedField(field_name="Title", status="found", value="Unseen Procurement", confidence=1.0, citations=[base])
    save_field_checkpoint(path, "fp", "Title", {
        "field": title.model_dump(mode="json"), "evidence": [base.model_dump(mode="json")],
        "decision": decision_payload, "operational": reusable, "retry_count": 0,
    })
    save_field_checkpoint(path, "fp", "Product Specification", {
        "field": outcome.field.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in outcome.evidence],
        "decision": {**decision_payload, "field_name": "Product Specification"},
        "operational": reusable, "retry_count": 0,
        "addendum_changes": [outcome.change.model_dump(mode="json")],
    })
    restored = reusable_checkpoint_state(load_checkpoint(path, "fp"))
    assert set(restored) == {"Title", "Product Specification"}
    assert restored["Title"]["field"]["value"] == "Unseen Procurement"
    assert restored["Product Specification"]["field"]["value"]["Network Probe"]["Warranty"] == "2 years"


def test_addendum_only_product_search_does_not_readd_full_base_inventory():
    raw = {
        "results": [{
            "chunk_id": "addendum-only",
            "file_name": "amendment.pdf",
            "page_number": 1,
            "text": "Product specifications are revised.",
            "doc_type": "addendum",
            "addendum_number": 1,
            "score": 0.9,
        }],
        "ranking_method": "cohere_rerank",
        "rerank_applied": True,
    }
    request = RetrievalRequest(
        bid_id="InventoryIsolationBid",
        field_name="Product Specification",
        question="Which addendum changes product specifications?",
        doc_type="addendum",
    )
    with patch(
        "agents.retrieval_agent.retrieve_and_rerank_cohere", return_value=raw
    ), patch(
        "agents.retrieval_agent.structured_product_inventory",
        side_effect=AssertionError("base inventory must not run for addendum-only retrieval"),
    ):
        result = search_evidence(request)
    assert result.status == "success"
    assert [item.evidence.chunk_id for item in result.passages] == ["addendum-only"]


def test_semantic_product_evidence_remains_usable_when_structure_inventory_is_empty():
    raw = {
        "results": [{
            "chunk_id": "prose-spec",
            "file_name": "unseen-request.pdf",
            "page_number": 6,
            "text": (
                "The field analyzer shall include 64 GB memory, an IP67 enclosure, "
                "and a five-year warranty."
            ),
            "doc_type": "rfp",
            "score": 0.91,
        }],
        "ranking_method": "cohere_rerank",
        "rerank_applied": True,
    }
    request = RetrievalRequest(
        bid_id="ProseOnlyBid",
        field_name="Product Specification",
        question="What specifications are required?",
    )
    with patch(
        "agents.retrieval_agent.retrieve_and_rerank_cohere", return_value=raw
    ), patch(
        "agents.retrieval_agent.structured_product_inventory", return_value=[]
    ):
        result = search_evidence(request)
    assert result.status == "success"
    assert [item.evidence.chunk_id for item in result.passages] == ["prose-spec"]
