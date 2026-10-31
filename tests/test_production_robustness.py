import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agents.schemas import (
    FieldRunRecord,
    RunManifest,
)
from providers.rate_limit import RateLimitSettings, SharedRateLimiter
from workflow.checkpoints import (
    invalidate_checkpoint_fields,
    load_checkpoint,
    reusable_checkpoint_state,
    repair_checkpoint_contact_values,
    repair_checkpoint_citations,
    normalize_checkpoint_values,
    revalidate_checkpoint_fields,
    save_field_checkpoint,
    workflow_fingerprint,
)
from agents.evidence_verification import (
    claim_evidence_errors,
    find_quote_evidence,
    is_unknown_attribute_marker,
    material_claims,
    normalize_unestablished_composite_claims,
    quote_matches,
    resolve_claim_evidence,
    resolve_value_evidence,
    value_support_errors,
)


def test_shared_rate_limiter_paces_starts():
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = SharedRateLimiter(
        RateLimitSettings(60, 1, 0, 10), clock=lambda: now[0], sleeper=sleep
    )
    limiter.acquire(); limiter.release()
    limiter.acquire(); limiter.release()
    assert sleeps == [1.0]


def test_cohere_429_uses_explicit_qdrant_rrf_fallback(monkeypatch):
    import rag.cohere_reranker as module

    candidate = {
        "rank": 1, "score": 0.8, "file_name": "rfp.pdf", "page_number": 2,
        "chunk_id": "c1", "text": "supported text", "doc_type": "rfp",
        "addendum_number": None, "section": "General",
    }
    monkeypatch.setattr(module, "search", lambda **kwargs: {"results": [candidate]})
    monkeypatch.setenv("COHERE_API_KEY", "test")
    client = Mock()
    client.rerank.side_effect = RuntimeError("HTTP 429 trial key 10 API calls/minute")
    monkeypatch.setattr(module.cohere, "ClientV2", lambda **kwargs: client)
    limiter = Mock()
    limiter.call.side_effect = RuntimeError("HTTP 429 trial key 10 API calls/minute")
    monkeypatch.setattr(module, "cohere_limiter", lambda: limiter)

    output = module.retrieve_and_rerank_cohere("q", "Bid1", top_k=1)
    assert output["ranking_method"] == "qdrant_hybrid_rrf_fallback"
    assert output["rerank_applied"] is False
    assert output["results"][0]["chunk_id"] == "c1"
    assert output["results"][0]["rerank_score"] is None


def test_gemini_429_is_provider_error_not_document_absence():
    from agents.schemas import ExtractedField
    from agents.field_specs import get_field_spec
    from agents.validator import validate_field

    decision = validate_field(
        field=ExtractedField(
            field_name="Due Date", status="error", value=None,
            reasoning="quota_or_rate_limit: HTTP 429 RESOURCE_EXHAUSTED",
        ),
        field_spec=get_field_spec("Due Date"), evidence=[], retry_count=0, max_retries=0,
    )
    assert decision.status == "provider_error"
    assert "Not found" not in " ".join(decision.reasons)


def test_incomplete_manifest_cannot_claim_final_submission():
    manifest = RunManifest(
        run_id="r", bid_id="Bid1", run_status="partial",
        final_submission_published=False, checkpoint_fingerprint="abc",
        field_statuses={
            "Due Date": FieldRunRecord(status="provider_error", reasons=["429"])
        },
    )
    assert manifest.run_status == "partial"
    assert manifest.status_counts == {"provider_error": 1}
    with pytest.raises(ValueError):
        RunManifest(
            run_id="r", bid_id="Bid1", run_status="complete",
            final_submission_published=True, checkpoint_fingerprint="abc",
            field_statuses={"Due Date": FieldRunRecord(status="provider_error")},
        )


def test_successful_finalization_requires_all_official_terminal_statuses():
    from agents.field_specs import OFFICIAL_FIELD_LABELS
    manifest = RunManifest(
        run_id="r", bid_id="Bid1", run_status="complete",
        final_submission_published=True, checkpoint_fingerprint="abc",
        field_statuses={
            name: FieldRunRecord(
                status="passed", terminal=True, reusable=True,
                reconciliation_verified=True,
            )
            for name in OFFICIAL_FIELD_LABELS
        },
    )
    assert manifest.run_status == "complete"
    assert manifest.status_counts == {"passed": len(OFFICIAL_FIELD_LABELS)}


def test_checkpoint_resume_reuses_only_terminal_records(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    save_field_checkpoint(path, "fp", "Bid Number", {
        "field": {}, "decision": {},
        "operational": FieldRunRecord(
            status="passed", terminal=True, reusable=True,
            reconciliation_verified=True,
        ).model_dump(mode="json"),
    })
    save_field_checkpoint(path, "fp", "Due Date", {
        "field": {}, "decision": {},
        "operational": FieldRunRecord(
            status="provider_error", terminal=False, reusable=False,
        ).model_dump(mode="json"),
    })
    reusable = reusable_checkpoint_state(load_checkpoint(path, "fp"))
    assert set(reusable) == {"Bid Number"}
    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(path, "stale")


def test_checkpoint_fingerprint_includes_configured_model(tmp_path: Path):
    (tmp_path / "bid.pdf").write_bytes(b"stable document")
    first = workflow_fingerprint(tmp_path, {"gemini_model": "gemini-2.5-flash-lite"})
    second = workflow_fingerprint(tmp_path, {"gemini_model": "gemini-3.1-flash-lite"})
    assert first != second


def test_reconciliation_failure_is_visible():
    from agents.reconciler import reconcile_field
    from agents.schemas import Evidence, ExtractedField
    from agents.field_specs import get_field_spec

    evidence = Evidence(file_name="rfp.pdf", page_number=1, quote="Due May 1", chunk_id="c1", doc_type="rfp")
    field = ExtractedField(field_name="Due Date", status="found", value="May 1", confidence=.8, citations=[evidence])
    failed = SimpleNamespace(status="error", passages=[], trace=None)
    result = reconcile_field(
        bid_id="Bid1", field_spec=get_field_spec("Due Date"), field=field,
        evidence=[evidence], retrieval_fn=lambda request: failed,
    )
    assert result.verified is False
    assert "incomplete" in result.warning.casefold()


def test_genuine_missing_evidence_is_terminal_after_successful_addendum_check():
    from agents.reconciler import reconcile_field
    from agents.schemas import ExtractedField
    from agents.field_specs import get_field_spec

    result = reconcile_field(
        bid_id="Bid1",
        field_spec=get_field_spec("Part_no"),
        field=ExtractedField(
            field_name="Part_no", status="not_found", value=None,
            reasoning="No part or SKU identifiers occur in the reviewed evidence.",
        ),
        evidence=[],
        retrieval_fn=lambda request: SimpleNamespace(
            status="success", passages=[], trace=None
        ),
    )
    assert result.verified is True
    assert "no addendum evidence" in result.verification_reason.casefold()


def test_quote_matching_normalizes_ocr_unicode_punctuation_and_line_breaks():
    assert quote_matches(
        "Vendor’s pre-bid con-\nference — June 10, 2024.",
        "Vendor's pre-bid conference - June 10, 2024!",
    )


def test_quote_can_match_sibling_chunk_on_same_source_page():
    from agents.schemas import Evidence
    selected = SimpleNamespace(
        rank=1,
        evidence=Evidence(
            file_name="bid.pdf", page_number=2, quote="First half of page.",
            chunk_id="first",
        ),
    )
    sibling = SimpleNamespace(
        rank=2,
        evidence=Evidence(
            file_name="bid.pdf", page_number=2,
            quote="Prebid Conference June 10, 2024", chunk_id="second",
        ),
    )
    match, error = find_quote_evidence(
        "Prebid Conference: June 10, 2024", [selected, sibling], 1
    )
    assert error is None
    assert match.evidence.chunk_id == "second"


def test_fabricated_quote_and_value_are_rejected():
    from agents.schemas import Evidence
    source = Evidence(
        file_name="bid.pdf", page_number=1,
        quote="Electronic responses are accepted through the iSupplier portal.",
        chunk_id="source",
    )
    assert not quote_matches("Bids must be emailed to invented@example.com", source.quote)
    errors = value_support_errors("Submit to invented@example.com", [source])
    assert errors and "invented@example.com" in errors[0]


@pytest.mark.parametrize(
    ("value", "file_name", "page", "source"),
    [
        (
            "Electronic through the iSupplier portal or manual sealed-envelope delivery; no fax or email submissions.",
            "JA-207652 Student and Staff Computing Devices FINAL.pdf", 9,
            "system. Please register with the iSupplier portal https://www.dallasisd.org/domain/143. "
            "Submissions may be submitted manually. Manual submissions must be delivered to the Procurement "
            "Services Department, in a sealed envelope by the date and time stated. No fax or email submissions "
            "will be accepted. Only one format is needed.",
        ),
        (
            "Any awarded vendor will need to coordinate a schedule of deliveries with the Dallas ISD ED TECH department.",
            "Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf", 4,
            "34. Deployment: Does Dallas ISD have a deployment schedule? Answer: Any awarded vendor will need "
            "to coordinate a schedule of deliveries with the Dallas ISD ED TECH department.",
        ),
        (
            "Prime vendors must pay all submitted invoices, including retainage, within 10 days of receiving payment from the district.",
            "JA-207652 Student and Staff Computing Devices FINAL.pdf", 51,
            "Subcontractor Payment. The Prime vendor shall submit an M/WBE Pay Activity Report. All Prime vendors "
            "must pay all submitted invoices, including retainage to subcontractors, suppliers, or entities within "
            "10 days of receiving payment from the district; No Prime vendor may withhold retainage greater than 5%.",
        ),
        (
            ["laptops", "desktops", "tablet devices", "display monitors"],
            "JA-207652 Student and Staff Computing Devices FINAL.pdf", 3,
            "This Request for Proposal (RFP) will be for student and staff computing devices, including laptops, "
            "desktops, tablet devices, and display monitors.",
        ),
    ],
)
def test_actual_bid1_failures_publish_only_authentic_indexed_quotes(
    value, file_name, page, source
):
    from agents.schemas import Evidence

    passage = SimpleNamespace(
        rank=1,
        evidence=Evidence(
            file_name=file_name, page_number=page, quote=source,
            chunk_id=f"actual-{page}",
        ),
    )
    match, authentic_quote, relocated, error = resolve_value_evidence(
        value=value,
        passages=[passage],
        selected_rank=1,
        evidence_id=passage.evidence.chunk_id,
    )
    assert error is None
    assert match is passage
    assert relocated is False
    assert authentic_quote in source
    assert not value_support_errors(value, [passage.evidence])


def test_wrong_page_is_relocated_only_when_retrieved_evidence_supports_value():
    from agents.schemas import Evidence

    wrong = SimpleNamespace(
        rank=1,
        evidence=Evidence(
            file_name="FINAL.pdf", page_number=17,
            quote="Delivery addresses will be provided after award.", chunk_id="wrong",
        ),
    )
    correct = SimpleNamespace(
        rank=2,
        evidence=Evidence(
            file_name="Addendum 1.pdf", page_number=4,
            quote="Any awarded vendor will need to coordinate a schedule of deliveries with Dallas ISD ED TECH department.",
            chunk_id="correct",
        ),
    )
    match, quote, relocated, error = resolve_value_evidence(
        value="Coordinate a schedule of deliveries with Dallas ISD ED TECH department",
        passages=[wrong, correct], selected_rank=1, evidence_id="wrong",
    )
    assert error is None
    assert match is correct
    assert relocated is True
    assert quote in correct.evidence.quote


def test_unsupported_value_is_rejected_even_when_model_supplies_a_quote():
    from agents.schemas import Evidence

    passage = SimpleNamespace(
        rank=1,
        evidence=Evidence(
            file_name="FINAL.pdf", page_number=9,
            quote="Responses may be submitted through the iSupplier portal.",
            chunk_id="submission",
        ),
    )
    match, quote, relocated, error = resolve_value_evidence(
        value="Email the bid to invented@example.com",
        passages=[passage], selected_rank=1, evidence_id="submission",
    )
    assert match is None and quote is None and relocated is False
    assert "not supported" in error


@pytest.mark.parametrize("status", [503, 504])
def test_gemini_transient_http_failures_are_bounded_and_retryable(status):
    from agents.llm_client import GeminiClient, GeminiSettings
    from pydantic import BaseModel, SecretStr

    class Result(BaseModel):
        answer: str

    class Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError(f"HTTP {status} transient unavailable")
            return SimpleNamespace(
                parsed=Result(answer="ok"), text='{"answer":"ok"}',
                candidates=[], usage_metadata=None,
            )

    models = Models()
    limiter = SharedRateLimiter(
        RateLimitSettings(1_000_000, 1, 2, .01),
        sleeper=lambda _: None,
    )
    client = GeminiClient(
        settings=GeminiSettings(
            api_key=SecretStr("test"), model="gemini-3.1-flash-lite",
        ),
        sdk_client=SimpleNamespace(models=models),
        rate_limiter=limiter,
    )
    assert client.generate_structured("prompt", Result).output.answer == "ok"
    assert models.calls == 3


def test_exhausted_504_remains_provider_error_and_resumable():
    from agents.field_specs import get_field_spec
    from agents.schemas import ExtractedField
    from agents.validator import validate_field

    decision = validate_field(
        field=ExtractedField(
            field_name="Part_no", status="error", value=None,
            reasoning="transient_unavailable: HTTP 504 after bounded retries",
        ),
        field_spec=get_field_spec("Part_no"), evidence=[],
        retry_count=0, max_retries=0,
    )
    assert decision.status == "provider_error"
    assert decision.retry_required is False


def test_list_items_use_multiple_real_bid1_passages_and_physical_pages():
    from agents.schemas import Evidence

    certification = SimpleNamespace(
        rank=1,
        evidence=Evidence(
            file_name="JA-207652 Student and Staff Computing Devices FINAL.pdf",
            page_number=3,
            chunk_id="8084d7b1-279a-5ac0-8d1e-58f21aed91e6",
            quote="For proposing vendors who are resellers, a copy of all factory-authorized repair and maintenance certifications must be submitted with each proposal.",
        ),
    )
    form_1295 = SimpleNamespace(
        rank=2,
        evidence=Evidence(
            file_name="Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf",
            page_number=4,
            chunk_id="6cb5d311-3359-564a-99e7-2fe96b93c483",
            quote="Answer: The Form 1295 should be submitted with the RFP response.",
            doc_type="addendum", addendum_number=1,
        ),
    )
    value = [
        "Copy of all factory-authorized repair and maintenance certifications",
        "Form 1295 should be submitted with the RFP response",
    ]
    diagnostics, matches, error = resolve_claim_evidence(
        value=value, passages=[certification, form_1295],
        selected_rank=1, evidence_id=certification.evidence.chunk_id,
    )
    assert error is None
    assert all(item["supported"] for item in diagnostics)
    assert {(item.evidence.file_name, item.evidence.page_number) for item, _ in matches} == {
        (certification.evidence.file_name, 3),
        (form_1295.evidence.file_name, 4),
    }


def test_signed_addendum_claim_uses_matching_indexed_addendum_metadata():
    from agents.schemas import Evidence

    passage = SimpleNamespace(
        rank=1,
        evidence=Evidence(
            file_name="Addendum 1.pdf", page_number=5, chunk_id="addendum-1-signature",
            doc_type="addendum", addendum_number=1,
            quote=(
                "Please sign this addendum and submit along with your copies of the "
                "proposal. Bidders are required to acknowledge and return/submit a "
                "copy of this addendum with their proposal."
            ),
        ),
    )
    diagnostics, matches, error = resolve_claim_evidence(
        value=["Signed copy of Addendum 1"], passages=[passage], selected_rank=1,
    )
    assert error is None
    assert diagnostics[0]["supported"] is True
    assert matches[0][0].evidence.addendum_number == 1
    assert "sign this addendum" in matches[0][1].casefold()

    _, _, wrong_addendum_error = resolve_claim_evidence(
        value=["Signed copy of Addendum 2"], passages=[passage], selected_rank=1,
    )
    assert "Signed copy of Addendum 2" in wrong_addendum_error


def test_one_correct_citation_cannot_justify_unrelated_composite_claim():
    from agents.schemas import Evidence

    citation = Evidence(
        file_name="FINAL.pdf", page_number=51, chunk_id="payment",
        quote="Prime vendors must pay subcontractor invoices within 10 days of receiving payment from the district.",
    )
    value = (
        "Prime vendors must pay subcontractor invoices within 10 days of receiving payment from the district. "
        "The District must pay the prime vendor within 30 days."
    )
    diagnostics, errors = claim_evidence_errors(value, [citation])
    assert diagnostics[0]["supported"] is True
    assert diagnostics[1]["supported"] is False
    assert errors and "30 days" in errors[0]


def test_product_items_are_independently_supported_and_invention_fails():
    from agents.schemas import Evidence

    passage = SimpleNamespace(
        rank=1,
        evidence=Evidence(
            file_name="JA-207652 Student and Staff Computing Devices FINAL.pdf",
            page_number=3,
            chunk_id="8853fa3d-5432-5cf3-9af2-ad256f03417c",
            quote=(
                "This Request for Proposal (RFP) will be for student and staff computing devices, "
                "including laptops, desktops, tablet devices, and display monitors. Purchases may "
                "also include device accessories including sleeves, cases, mice, keyboards, and cables."
            ),
        ),
    )
    supported = ["laptops", "desktops", "tablet devices", "display monitors", "device accessories", "sleeves", "cases", "mice", "keyboards", "cables"]
    diagnostics, matches, error = resolve_claim_evidence(
        value=supported, passages=[passage], selected_rank=1,
        evidence_id=passage.evidence.chunk_id,
    )
    assert error is None and len(matches) == len(supported)
    diagnostics, matches, error = resolve_claim_evidence(
        value=[*supported, "smartphones"], passages=[passage], selected_rank=1,
        evidence_id=passage.evidence.chunk_id,
    )
    assert "smartphones" in error
    assert diagnostics[-1]["supported"] is False


def test_composite_delivery_claim_uses_separate_real_pages_and_rejects_absence_claim():
    from agents.schemas import Evidence

    activity = SimpleNamespace(
        rank=1,
        evidence=Evidence(
            file_name="Addendum 1.pdf", page_number=1, chunk_id="activity",
            quote="Dallas ISD is anticipating requesting quotes for different projects from awarded vendors starting in September of 2024.",
        ),
    )
    schedule = SimpleNamespace(
        rank=2,
        evidence=Evidence(
            file_name="Addendum 1.pdf", page_number=4, chunk_id="schedule",
            quote="Any awarded vendor will need to coordinate a schedule of deliveries with the Dallas ISD ED TECH department.",
        ),
    )
    value = (
        "Dallas ISD anticipates requesting project quotes starting in September 2024; "
        "awarded vendors must coordinate delivery schedules with Dallas ISD ED TECH."
    )
    diagnostics, matches, error = resolve_claim_evidence(
        value=value, passages=[activity, schedule], selected_rank=1,
    )
    assert error is None
    assert {item.evidence.page_number for item, _ in matches} == {1, 4}
    _, _, error = resolve_claim_evidence(
        value="There is no fixed delivery date; " + value,
        passages=[activity, schedule], selected_rank=1,
    )
    assert "no fixed delivery date" in error


def test_delivery_passive_paraphrase_uses_authentic_ocr_split_source_span():
    from agents.schemas import Evidence

    source = (
        "34. Deployment: Does the agency have a deployment schedule? "
        "Answer: Any awarded vendor will need to coordinate a schedule of deliveries with the\n"
        "6. Agency technology department."
    )
    passage = SimpleNamespace(
        rank=5,
        evidence=Evidence(
            file_name="Addendum.pdf", page_number=4,
            chunk_id="delivery-coordination", quote=source,
        ),
    )
    value = "Delivery schedules are to be coordinated with the agency technology department."
    diagnostics, matches, error = resolve_claim_evidence(
        value=value, passages=[passage], selected_rank=5,
        evidence_id=passage.evidence.chunk_id,
    )
    assert error is None
    assert diagnostics[0]["supported"] is True
    assert matches[0][1] in source
    assert "\n6. " in matches[0][1]

    wrong_page = SimpleNamespace(
        rank=1,
        evidence=Evidence(
            file_name="Terms.pdf", page_number=19, chunk_id="wrong-page",
            quote="Invoices are due after accepted delivery.",
        ),
    )
    _, _, wrong_error = resolve_claim_evidence(
        value=value, passages=[wrong_page], selected_rank=1,
        evidence_id=wrong_page.evidence.chunk_id,
    )
    assert wrong_error is not None


def test_unseen_bid3_fixture_uses_same_extraction_validation_and_contract_logic():
    from agents.extractor import extract_field
    from agents.field_specs import OFFICIAL_FIELD_LABELS, get_field_spec
    from agents.schemas import (
        Evidence, FinalExtractionOutput, FinalFieldRecord, ValidationSummary,
    )
    from agents.validator import validate_field

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "Bid3_chunks.json").read_text(encoding="utf-8")
    )[0]
    evidence = Evidence(
        file_name=fixture["file_name"], page_number=fixture["page_number"],
        quote=fixture["text"], chunk_id=fixture["chunk_id"],
        doc_type=fixture["doc_type"], addendum_number=fixture["addendum_number"],
    )
    retrieval = SimpleNamespace(
        bid_id="Bid3", field_name="Delivery Date",
        question=get_field_spec("Delivery Date").questions[0], status="success",
        passages=[SimpleNamespace(rank=1, score=.9, evidence=evidence)],
    )

    class OfflineClient:
        model = "offline-test-model"

        def generate_structured(self, prompt, response_model, **kwargs):
            return SimpleNamespace(
                output=response_model(
                    value="Released orders must be delivered within 14 calendar days after receipt of the applicable purchase order.",
                    evidence_rank=1, evidence_id=fixture["chunk_id"], quote="",
                    reasoning="Direct delivery window.", confidence=.95,
                ),
                usage=SimpleNamespace(total_tokens=0),
            )

    field, _ = extract_field(retrieval, llm_client=OfflineClient())
    assert field.status == "found"
    assert field.citations[0].file_name == "Transit_Safety_Tools_Addendum_A.pdf"
    assert field.citations[0].page_number == 7
    decision = validate_field(
        field=field, field_spec=get_field_spec("Delivery Date"), evidence=[evidence],
        retry_count=0, max_retries=0, semantic_validation=False,
    )
    assert decision.status == "passed"

    missing_retrieval = SimpleNamespace(
        bid_id="Bid3", field_name="Model_no",
        question=get_field_spec("Model_no").questions[0], status="empty", passages=[],
    )
    missing, _ = extract_field(missing_retrieval, llm_client=OfflineClient())
    missing_decision = validate_field(
        field=missing, field_spec=get_field_spec("Model_no"), evidence=[],
        retry_count=0, max_retries=0, semantic_validation=False,
    )
    assert missing_decision.status == "not_found_evidence"

    records = {name: FinalFieldRecord(value="fixture") for name in OFFICIAL_FIELD_LABELS}
    records["Model_no"] = FinalFieldRecord(value=None, notes="Not found in fixture evidence")
    output = FinalExtractionOutput(
        bid_id="Bid3", fields=records,
        validation=ValidationSummary(passed=19, not_found=1),
    )
    assert len(output.fields) == 20


def _summary_source_fields():
    from agents.schemas import CriticDecision, Evidence, ExtractedField

    values = {
        "Title": "Transit Safety Tools",
        "Term of Bid": "The initial contract term is two years",
        "Delivery Date": "Orders must be delivered within 14 calendar days",
    }
    fields = {}
    decisions = {}
    evidence = []
    for index, (name, value) in enumerate(values.items(), start=1):
        citation = Evidence(
            file_name="Unseen_RFP.pdf", page_number=index,
            chunk_id=f"summary-{index}", quote=value + ".",
        )
        evidence.append(citation)
        fields[name] = ExtractedField(
            field_name=name, status="found", value=value,
            confidence=.9, citations=[citation],
        )
        decisions[name] = CriticDecision(
            field_name=name, status="passed", confidence=.9,
        )
    return fields, decisions, evidence


def test_summary_claims_are_attributed_across_all_validated_fields_not_model_subset():
    from agents.report_agent import generate_bid_summary

    fields, decisions, _ = _summary_source_fields()

    class SummaryClient:
        model = "offline-test-model"

        def generate_structured(self, prompt, response_model, **kwargs):
            return SimpleNamespace(output=response_model(
                value=(
                    "Transit Safety Tools. The initial contract term is two years. "
                    "Orders must be delivered within 14 calendar days."
                ),
                source_fields=["Title"], confidence=.9,
            ))

    summary = generate_bid_summary(
        fields=fields, decisions=decisions, llm_client=SummaryClient()
    )
    assert summary.status == "found"
    assert {item.chunk_id for item in summary.citations} == {
        "summary-1", "summary-2", "summary-3"
    }


def test_unsupported_generated_summary_uses_revalidated_exact_value_fallback():
    from agents.field_specs import get_field_spec
    from agents.report_agent import generate_bid_summary
    from agents.validator import validate_field

    fields, decisions, evidence = _summary_source_fields()

    class UnsupportedSummaryClient:
        model = "offline-test-model"

        def generate_structured(self, prompt, response_model, **kwargs):
            return SimpleNamespace(output=response_model(
                value=(
                    "The agency seeks revolutionary equipment. Vendors receive a "
                    "guaranteed award. Unlimited extensions are available."
                ),
                source_fields=list(fields), confidence=.99,
            ))

    summary = generate_bid_summary(
        fields=fields, decisions=decisions, llm_client=UnsupportedSummaryClient()
    )
    assert summary.status == "found"
    assert "Deterministic fallback" in summary.reasoning
    assert "revolutionary" not in summary.value
    decision = validate_field(
        field=summary, field_spec=get_field_spec("Bid Summary"), evidence=evidence,
        retry_count=0, max_retries=0, semantic_validation=False,
    )
    assert decision.status == "passed"


def test_transient_summary_provider_failure_uses_same_validated_fallback():
    from agents.field_specs import get_field_spec
    from agents.llm_client import GeminiGenerationError
    from agents.report_agent import generate_bid_summary
    from agents.validator import validate_field

    fields, decisions, evidence = _summary_source_fields()

    class UnavailableSummaryClient:
        def generate_structured(self, *args, **kwargs):
            raise GeminiGenerationError(
                "service unavailable",
                classification="transient_unavailable",
                finish_reason="503",
            )

    summary = generate_bid_summary(
        fields=fields, decisions=decisions, llm_client=UnavailableSummaryClient()
    )
    assert summary.status == "found"
    assert "transient_unavailable" in summary.reasoning
    decision = validate_field(
        field=summary, field_spec=get_field_spec("Bid Summary"), evidence=evidence,
        retry_count=0, max_retries=0, semantic_validation=False,
    )
    assert decision.status == "passed"


def test_summary_preserves_distinct_supporting_spans_from_same_chunk():
    from agents.report_agent import _claim_attributed_citations
    from agents.schemas import Evidence

    common = {
        "file_name": "Addendum.pdf", "page_number": 4, "chunk_id": "shared-chunk",
    }
    installation = Evidence(
        **common, quote="Software installation is required for deployed devices."
    )
    delivery = Evidence(
        **common, quote="Vendors must coordinate delivery schedules with the agency."
    )
    value = (
        "Software installation is required for deployed devices. "
        "Vendors must coordinate delivery schedules with the agency."
    )
    citations, errors = _claim_attributed_citations(
        value, [installation, delivery]
    )
    assert errors == []
    assert [item.quote for item in citations] == [
        installation.quote, delivery.quote
    ]


def test_official_output_does_not_publish_unvalidated_model_reasoning():
    from agents.field_specs import OFFICIAL_FIELD_LABELS
    from agents.report_agent import assemble_final_output
    from agents.schemas import CriticDecision, Evidence, ExtractedField

    citation = Evidence(
        file_name="RFP.pdf", page_number=1, chunk_id="supported",
        quote="Supported value.",
    )
    fields = {
        name: ExtractedField(
            field_name=name, status="found", value="Supported value",
            confidence=.9, citations=[citation],
            reasoning="Unsupported global absence conclusion.",
        )
        for name in OFFICIAL_FIELD_LABELS
    }
    decisions = {
        name: CriticDecision(
            field_name=name, status="passed",
            reasons=["Deterministic validation passed"], confidence=.9,
        )
        for name in OFFICIAL_FIELD_LABELS
    }
    output = assemble_final_output(
        bid_id="UnseenBid", fields=fields, decisions=decisions, changes=[],
    )
    assert output.fields["Delivery Date"].notes == "Deterministic validation passed"
    assert "Unsupported global" not in output.model_dump_json()


@pytest.mark.parametrize(
    ("field_name", "value", "evidence_text", "unknown_label"),
    [
        (
            "Pre Bid Meeting",
            "The meeting is June 10, 2024 at 2:00 PM CST via TEAMS. "
            "Participation is highly encouraged but not mandatory.",
            "A pre-proposal meeting will be held at 2:00 PM CST via TEAMS video "
            "conference on June 10, 2024. It is highly encouraged that all "
            "potential proposers participate and attend the preproposal.",
            "Attendance requirement: unspecified",
        ),
        (
            "Delivery Date",
            "There is no fixed delivery date; awarded vendors must coordinate a "
            "schedule of deliveries with Dallas ISD ED TECH.",
            "Any awarded vendor will need to coordinate a schedule of deliveries "
            "with the Dallas ISD ED TECH department.",
            "Exact delivery date: unspecified",
        ),
        (
            "Payment Terms",
            "The provided documents do not specify the payment terms for the "
            "District to the Prime vendor; prime vendors must pay submitted "
            "subcontractor invoices within 10 days of receiving payment from the District.",
            "Prime vendors must pay submitted invoices from subcontractors within "
            "10 days of receiving payment from the District.",
            "District-to-prime payment terms: unspecified",
        ),
    ],
)
def test_real_unsupported_negative_clause_becomes_explicit_unknown_without_losing_support(
    field_name, value, evidence_text, unknown_label,
):
    from agents.schemas import Evidence

    passage = SimpleNamespace(
        rank=1,
        evidence=Evidence(
            file_name="Bid.pdf", page_number=1, chunk_id="real-evidence",
            quote=evidence_text,
        ),
    )
    diagnostics, _, original_error = resolve_claim_evidence(
        value=value, passages=[passage], selected_rank=1,
    )
    assert original_error is not None
    normalized, annotations = normalize_unestablished_composite_claims(
        field_name=field_name,
        value=value,
        diagnostics=diagnostics,
        passages=[passage],
    )
    assert annotations == [unknown_label]
    assert unknown_label in normalized
    assert "not mandatory" not in normalized.casefold()
    assert "no fixed delivery date" not in normalized.casefold()
    assert "documents do not specify" not in normalized.casefold()

    normalized_diagnostics, matches, normalized_error = resolve_claim_evidence(
        value=normalized, passages=[passage], selected_rank=1,
    )
    assert normalized_error is None
    assert matches
    assert any(item.get("evidence_status") == "unknown_attribute" for item in normalized_diagnostics)
    cited = [item.evidence.model_copy(update={"quote": quote}) for item, quote in matches]
    _, validation_errors = claim_evidence_errors(normalized, cited)
    assert validation_errors == []


def test_unknown_marker_is_not_documentary_absence_or_technical_failure():
    from agents.schemas import Evidence

    assert is_unknown_attribute_marker("Exact delivery date: unspecified")
    assert not is_unknown_attribute_marker("There is no fixed delivery date")
    citation = Evidence(
        file_name="Bid.pdf", page_number=4, chunk_id="schedule",
        quote="The vendor must coordinate a delivery schedule with the District.",
    )
    diagnostics, errors = claim_evidence_errors(
        "The vendor must coordinate a delivery schedule with the District. "
        "Exact delivery date: unspecified.",
        [citation],
    )
    assert errors == []
    assert diagnostics[-1]["supported"] is None
    assert diagnostics[-1]["evidence_status"] == "unknown_attribute"


def test_technical_configuration_commas_preserve_one_material_claim():
    value = {
        "Portable Workstation": {
            "Technical Requirements": (
                "Core Ultra 7 165H processor (24 MB cache, 16 cores, 22 threads, "
                "up to 5.0 GHz), 32 GB: 2 x 16 GB, DDR5, 5600 MT/s, 1 TB, "
                "M.2 2280, TLC, Gen 4 PCIe NVMe SSD, FHD camera (IR, shutter, microphone)"
            )
        }
    }
    claims = material_claims(value)
    assert len(claims) == 1
    assert "24 MB cache, 16 cores, 22 threads" in claims[0]["claim"]
    assert "32 GB: 2 x 16 GB, DDR5" in claims[0]["claim"]


def test_comma_rich_technical_configuration_is_supported_but_fabrication_is_rejected():
    from agents.schemas import Evidence

    source = Evidence(
        file_name="unseen-specification.pdf", page_number=4, chunk_id="technical-config",
        quote=(
            "Core Ultra 7 165H processor (24 MB cache, 16 cores, 22 threads, "
            "up to 5.0 GHz), 32 GB: 2 x 16 GB, DDR5, 5600 MT/s, 1 TB, "
            "M.2 2280, TLC, Gen 4 PCIe NVMe SSD, FHD camera (IR, shutter, microphone)."
        ),
    )
    supported = {
        "Portable Workstation": {
            "Technical Requirements": source.quote.rstrip(".")
        }
    }
    diagnostics, errors = claim_evidence_errors(supported, [source])
    assert not errors
    assert all(item["supported"] is True for item in diagnostics)

    fabricated = {
        "Portable Workstation": {
            "Technical Requirements": (
                "Core Ultra 9 999Z processor, 96 GB DDR5, 4 TB NVMe SSD"
            )
        }
    }
    _, errors = claim_evidence_errors(fabricated, [source])
    assert errors


def test_nested_unspecified_attributes_are_uncertainty_markers_only():
    from agents.schemas import Evidence

    source = Evidence(
        file_name="unseen-rfp.pdf", page_number=2, chunk_id="invoice",
        quote="Invoices must be emailed within ten days after delivery.",
    )
    value = {
        "Invoicing Instructions": "Invoices must be emailed within ten days after delivery.",
        "Retainage": "unspecified",
        "Payment Schedule": "unknown",
    }
    diagnostics, errors = claim_evidence_errors(value, [source])
    assert not errors
    unknown = [item for item in diagnostics if item["supported"] is None]
    assert {item["claim"] for item in unknown} == {
        "Retainage: unspecified", "Payment Schedule: unknown",
    }

    _, errors = claim_evidence_errors(
        {"Retainage": "The solicitation states that no retainage applies."}, [source]
    )
    assert errors


def test_payment_object_normalization_preserves_labels_and_unknown_markers():
    from agents.extractor import normalize_extracted_value

    normalized = normalize_extracted_value(
        "Payment Terms",
        {
            "Invoicing Instructions": "Invoices are due within ten days",
            "Retainage": "unspecified",
            "Payment Schedule": "unknown",
        },
    )
    assert normalized == (
        "Invoicing Instructions: Invoices are due within ten days. "
        "Retainage: unspecified. Payment Schedule: unknown."
    )
    claims = material_claims(normalized)
    assert [item["claim"] for item in claims[-2:]] == [
        "Retainage: unspecified.", "Payment Schedule: unknown.",
    ]


def test_iso_datetime_is_deterministically_supported_by_natural_language_date():
    from agents.schemas import Evidence

    citation = Evidence(
        file_name="Addendum 2.pdf", page_number=1, chunk_id="due-date",
        quote="The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.",
    )
    _, errors = claim_evidence_errors("2024-07-09T14:00:00 CST", [citation])
    assert errors == []


def test_bid_bond_cross_references_finish_as_evidence_based_not_found_without_gemini():
    from agents.field_specs import get_field_spec
    from agents.reconciler import reconcile_field
    from agents.schemas import Evidence, ExtractedField

    base = Evidence(
        file_name="FINAL.pdf", page_number=9, chunk_id="cross-reference",
        quote="The Offeror must comply with any bid bond requirements as noted elsewhere in this solicitation document.",
        doc_type="rfp",
    )
    addendum = Evidence(
        file_name="Addendum 1.pdf", page_number=4, chunk_id="unchanged",
        quote="All other provisions and terms and conditions remain unchanged.",
        doc_type="addendum", addendum_number=1,
    )
    outcome = reconcile_field(
        bid_id="Bid1", field_spec=get_field_spec("Bid Bond Requirement"),
        field=ExtractedField(
            field_name="Bid Bond Requirement", status="not_found",
            value="Not found in documents", reasoning="No affirmative requirement found.",
        ),
        evidence=[base],
        retrieval_fn=lambda request: SimpleNamespace(
            status="success", passages=[SimpleNamespace(evidence=addendum)], trace=None,
        ),
        llm_client=Mock(),
    )
    assert outcome.verified is True
    assert outcome.field.status == "not_found"
    assert "No passage affirmatively states" in outcome.verification_reason


def test_checkpoint_citation_repair_is_selective_and_preserves_fingerprint(tmp_path: Path):
    from agents.schemas import CriticDecision, Evidence, ExtractedField

    path = tmp_path / "checkpoint.json"
    electronic = Evidence(
        file_name="FINAL.pdf", page_number=9, chunk_id="electronic",
        quote="Dallas ISD prefers electronic responses through the iSupplier portal.",
    )
    manual = Evidence(
        file_name="FINAL.pdf", page_number=9, chunk_id="manual",
        quote="Manual responses may be delivered in a sealed envelope by mail or FedEx.",
    )
    field = ExtractedField(
        field_name="Bid Submission Type", status="found",
        value="Electronic responses use the iSupplier portal; manual responses may be delivered by mail or FedEx.",
        confidence=.9, citations=[manual],
    )
    save_field_checkpoint(path, "fp", field.field_name, {
        "field": field.model_dump(mode="json"),
        "evidence": [electronic.model_dump(mode="json"), manual.model_dump(mode="json")],
        "decision": CriticDecision(
            field_name=field.field_name, status="passed", confidence=.9,
        ).model_dump(mode="json"),
        "operational": FieldRunRecord(
            status="passed", terminal=True, reusable=True,
            reconciliation_verified=True,
        ).model_dump(mode="json"),
    })
    repaired = repair_checkpoint_citations(path, "fp", {field.field_name, "Title"})
    payload = load_checkpoint(path, "fp")
    assert repaired == {field.field_name}
    assert payload["fingerprint"] == "fp"
    repaired_field = payload["fields"][field.field_name]["field"]
    assert {item["chunk_id"] for item in repaired_field["citations"]} == {
        "electronic", "manual"
    }
    assert all(item["supported"] for item in payload["fields"][field.field_name]["claim_evidence"])


def test_repaired_checkpoint_is_reusable_only_after_deterministic_revalidation(tmp_path: Path):
    from agents.schemas import CriticDecision, Evidence, ExtractedField

    path = tmp_path / "checkpoint.json"
    evidence = Evidence(
        file_name="FINAL.pdf", page_number=3, chunk_id="products",
        quote=(
            "The products include laptops. Vendors may bid one or more categories. "
            "The products also include desktops."
        ),
    )
    field = ExtractedField(
        field_name="Product", status="found", value=["laptops", "desktops"],
        confidence=.9,
        citations=[evidence.model_copy(update={"quote": "The products include laptops."})],
    )
    save_field_checkpoint(path, "fp", field.field_name, {
        "field": field.model_dump(mode="json"),
        "evidence": [evidence.model_dump(mode="json")],
        "decision": CriticDecision(
            field_name=field.field_name, status="validation_failed", confidence=0.0,
        ).model_dump(mode="json"),
        "operational": FieldRunRecord(
            status="validation_failed", terminal=False, reusable=False,
            reconciliation_verified=True,
        ).model_dump(mode="json"),
    })
    assert repair_checkpoint_citations(path, "fp", {field.field_name}) == {field.field_name}
    assert revalidate_checkpoint_fields(path, "fp", {field.field_name}) == {field.field_name}
    payload = load_checkpoint(path, "fp")
    assert set(reusable_checkpoint_state(payload)) == {field.field_name}
    assert payload["fields"][field.field_name]["field"]["citations"][0]["quote"] == evidence.quote

    payload["fields"][field.field_name]["operational"]["reconciliation_verified"] = False
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    assert revalidate_checkpoint_fields(path, "fp", {field.field_name}) == set()


def test_contact_checkpoint_offline_repair_preserves_unrelated_success(tmp_path: Path):
    from agents.schemas import CriticDecision, Evidence, ExtractedField

    path = tmp_path / "checkpoint.json"
    contact = Evidence(
        file_name="unseen.pdf", page_number=2, chunk_id="contact",
        quote=(
            "Contact Information Procurement Office 555-204-7788 bids@unseen.gov. "
            "Questions must be submitted in writing."
        ),
    )
    address = Evidence(
        file_name="unseen.pdf", page_number=7, chunk_id="manual-address",
        quote=(
            "Manual paper responses must be delivered to the following address: "
            "Unseen Procurement Office, 42 Harbor Way, Suite 8, Bayview, CA 90001. "
            "Questions received after the deadline will not be answered."
        ),
    )
    field = ExtractedField(
        field_name="contact_info", status="found", confidence=.9,
        value={"office": {
            "email": "bids@unseen.gov", "phone": "555-204-7788",
            "communication_instruction": "Questions must be submitted in writing.",
        }},
        citations=[contact],
    )
    save_field_checkpoint(path, "fp", "Bid Number", {
        "field": ExtractedField(
            field_name="Bid Number", status="found", value="RFP-42",
            confidence=.9,
        ).model_dump(mode="json"),
        "evidence": [], "decision": {},
        "operational": FieldRunRecord(
            status="passed", terminal=True, reusable=True,
            reconciliation_verified=True,
        ).model_dump(mode="json"),
    })
    save_field_checkpoint(path, "fp", "contact_info", {
        "field": field.model_dump(mode="json"),
        "evidence": [contact.model_dump(mode="json"), address.model_dump(mode="json")],
        "decision": CriticDecision(
            field_name="contact_info", status="validation_failed", confidence=0,
        ).model_dump(mode="json"),
        "operational": FieldRunRecord(
            status="validation_failed", terminal=False, reusable=False,
            reconciliation_verified=True,
        ).model_dump(mode="json"),
    })

    assert repair_checkpoint_contact_values(path, "fp", {"contact_info"}) == {"contact_info"}
    assert repair_checkpoint_citations(path, "fp", {"contact_info"}) == {"contact_info"}
    assert revalidate_checkpoint_fields(path, "fp", {"contact_info"}) == {"contact_info"}
    payload = load_checkpoint(path, "fp")
    assert set(reusable_checkpoint_state(payload)) == {"Bid Number", "contact_info"}
    repaired = payload["fields"]["contact_info"]
    assert repaired["field"]["value"]["submission_delivery"]["mailing_address"].endswith(
        "Bayview, CA 90001"
    )
    assert {item["chunk_id"] for item in repaired["field"]["citations"]} == {
        "contact", "manual-address"
    }


class _DecisionClient:
    model = "test-model"

    def __init__(self, decision):
        self.decision = decision

    def generate_structured(self, *args, **kwargs):
        return SimpleNamespace(
            output=self.decision,
            usage=SimpleNamespace(total_tokens=1, model_dump=lambda: {"total_tokens": 1}),
            latency_ms=1,
        )


def test_addendum_augments_list_unless_explicitly_replaced():
    from agents.field_specs import get_field_spec
    from agents.reconciler import reconcile_field
    from agents.schemas import Evidence, ExtractedField, ReconciliationDecision

    base = Evidence(
        file_name="rfp.pdf", page_number=3,
        quote="Submit factory authorization certificates.", chunk_id="base", doc_type="rfp",
    )
    addendum = Evidence(
        file_name="addendum.pdf", page_number=1,
        quote="All other requirements remain unchanged. Bidders must submit signed Addendum 1.",
        chunk_id="add", doc_type="addendum", addendum_number=1,
    )
    field = ExtractedField(
        field_name="Any Additional Documentation Required", status="found",
        value=["Factory authorization certificates"], confidence=.8, citations=[base],
    )
    decision = ReconciliationDecision(
        field_name=field.field_name, amended=True,
        original_value=field.value, updated_value=["Signed Addendum 1"],
        original_evidence_rank=1, updated_evidence_rank=2,
        addendum_number=1, explanation="Adds an acknowledgement.", confidence=.9,
        change_mode="replace",
    )
    outcome = reconcile_field(
        bid_id="Bid1", field_spec=get_field_spec(field.field_name), field=field,
        evidence=[base], llm_client=_DecisionClient(decision),
        retrieval_fn=lambda request: SimpleNamespace(
            status="success", passages=[SimpleNamespace(evidence=addendum)], trace=None
        ),
    )
    assert outcome.field.value == [
        "Factory authorization certificates", "Signed Addendum 1"
    ]
    assert {item.chunk_id for item in outcome.field.citations} == {"base", "add"}


def test_citation_reattribution_adds_the_actual_addendum_source_only_when_supported():
    from agents.evidence_verification import reattribute_supported_citations
    from agents.schemas import Evidence, ExtractedField

    unrelated = Evidence(
        file_name="base.pdf", page_number=4,
        quote="Submit the technical response form.", chunk_id="base", doc_type="rfp",
    )
    addendum = Evidence(
        file_name="addendum-2.pdf", page_number=1,
        quote=(
            "Please sign this addendum and submit it with the proposal. "
            "Bidders are required to return a copy of this addendum."
        ),
        chunk_id="add-2", doc_type="addendum", addendum_number=2,
    )
    field = ExtractedField(
        field_name="Any Additional Documentation Required", status="found",
        value=["Technical response form", "Signed copy of Addendum 2"],
        confidence=.9, citations=[unrelated],
    )
    repaired, changed = reattribute_supported_citations(field, [unrelated, addendum])
    assert changed is True
    assert {item.chunk_id for item in repaired.citations} == {"base", "add-2"}
    assert next(item for item in repaired.citations if item.chunk_id == "add-2").page_number == 1

    unsupported, changed = reattribute_supported_citations(field, [unrelated])
    assert changed is False
    assert unsupported == field


def test_installation_accepts_positive_requirements_and_rejects_invented_absence():
    from agents.field_specs import get_field_spec
    from agents.schemas import Evidence, ExtractedField
    from agents.validator import validate_field

    source = Evidence(
        file_name="deployment.pdf", page_number=7,
        quote=(
            "The awarded supplier must configure each controller at the agency facility "
            "and enroll it in the device-management platform before acceptance."
        ),
        chunk_id="install", doc_type="rfp",
    )
    supported = ExtractedField(
        field_name="Installation", status="found",
        value=(
            "The awarded supplier must configure each controller at the agency facility "
            "and enroll it in the device-management platform before acceptance."
        ),
        confidence=.95, citations=[source],
    )
    assert validate_field(
        field=supported, field_spec=get_field_spec("Installation"), evidence=[source],
        retry_count=0, max_retries=0, semantic_validation=False,
    ).status == "passed"

    unsupported = supported.model_copy(update={
        "value": "The documents do not define who performs installation.",
    })
    decision = validate_field(
        field=unsupported, field_spec=get_field_spec("Installation"), evidence=[source],
        retry_count=0, max_retries=0, semantic_validation=False,
    )
    assert decision.status == "validation_failed"
    assert any("unsupported" in reason.casefold() for reason in decision.reasons)


def test_addendum_explicit_replacement_can_replace_list():
    from agents.field_specs import get_field_spec
    from agents.reconciler import reconcile_field
    from agents.schemas import Evidence, ExtractedField, ReconciliationDecision

    base = Evidence(
        file_name="rfp.pdf", page_number=3, quote="Submit Form A.",
        chunk_id="base", doc_type="rfp",
    )
    addendum = Evidence(
        file_name="addendum.pdf", page_number=1,
        quote="This replaces the requirement to submit Form A. Submit Form B instead.",
        chunk_id="add", doc_type="addendum", addendum_number=1,
    )
    field = ExtractedField(
        field_name="Any Additional Documentation Required", status="found",
        value=["Form A"], confidence=.8, citations=[base],
    )
    decision = ReconciliationDecision(
        field_name=field.field_name, amended=True,
        original_value=field.value, updated_value=["Form B"],
        updated_evidence_rank=2, addendum_number=1,
        explanation="Form B explicitly replaces Form A.", confidence=.9,
        change_mode="replace",
    )
    outcome = reconcile_field(
        bid_id="Bid1", field_spec=get_field_spec(field.field_name), field=field,
        evidence=[base], llm_client=_DecisionClient(decision),
        retrieval_fn=lambda request: SimpleNamespace(
            status="success", passages=[SimpleNamespace(evidence=addendum)], trace=None
        ),
    )
    assert outcome.field.value == ["Form B"]
    assert [item.chunk_id for item in outcome.field.citations] == ["add"]


def test_selective_checkpoint_invalidation_preserves_other_fields(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    for name in ("Bid Number", "Due Date", "Any Additional Documentation Required"):
        save_field_checkpoint(path, "fp", name, {
            "operational": FieldRunRecord(
                status="passed", terminal=True, reusable=True,
                reconciliation_verified=True,
            ).model_dump(mode="json")
        })
    removed = invalidate_checkpoint_fields(
        path, "fp", {"Any Additional Documentation Required"},
        reason="List reconciliation must append rather than replace.",
    )
    payload = load_checkpoint(path, "fp")
    assert removed == {"Any Additional Documentation Required"}
    assert set(payload["fields"]) == {"Bid Number", "Due Date"}
    assert payload["invalidations"][-1]["fields"] == ["Any Additional Documentation Required"]
    assert set(reusable_checkpoint_state(payload)) == {"Bid Number", "Due Date"}


def test_part_number_checkpoint_normalization_is_lossless_and_reusable(tmp_path: Path):
    from agents.schemas import Evidence, ExtractedField, FieldRunRecord

    path = tmp_path / "checkpoint.json"
    evidence = Evidence(
        file_name="items.pdf", page_number=2,
        quote="Mobile Console model ZX-4810 and Charging Cradle part CR-22.",
        chunk_id="parts", doc_type="rfp",
    )
    field = ExtractedField(
        field_name="Part_no", status="found",
        value={"Mobile Console": "ZX-4810", "Charging Cradle": "CR-22"},
        confidence=1.0, citations=[evidence],
    )
    save_field_checkpoint(path, "fp", "Part_no", {
        "field": field.model_dump(mode="json"),
        "evidence": [evidence.model_dump(mode="json")],
        "decision": {
            "field_name": "Part_no", "status": "validation_failed",
            "reasons": ["Expected list, got dict"], "retry_required": False,
            "confidence": 1.0,
        },
        "operational": FieldRunRecord(
            status="validation_failed", terminal=False, reusable=False,
            reconciliation_verified=True,
        ).model_dump(mode="json"),
    })
    assert normalize_checkpoint_values(path, "fp", {"Part_no"}) == {"Part_no"}
    assert revalidate_checkpoint_fields(path, "fp", {"Part_no"}) == {"Part_no"}
    restored = reusable_checkpoint_state(load_checkpoint(path, "fp"))["Part_no"]
    assert restored["field"]["value"] == [
        {"item": "Mobile Console", "part_number": "ZX-4810"},
        {"item": "Charging Cradle", "part_number": "CR-22"},
    ]
