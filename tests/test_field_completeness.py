import json
from pathlib import Path
from types import SimpleNamespace

from agents.field_completeness import (
    available_contact_components,
    contact_completeness_errors,
    discover_product_categories,
    documentation_structure_errors,
    enrich_contact_value_from_evidence,
    product_coverage_errors,
)
from agents.schemas import AddendumChange, Evidence, SourceCitation
from rag.structured_inventory import structured_contact_inventory, structured_product_inventory
from workflow.checkpoints import (
    checkpoint_addendum_changes,
    recover_addendum_changes_from_traces,
    reusable_checkpoint_state,
    save_field_checkpoint,
)


def evidence(text, chunk_id, *, section=None, chunk_type=None, page=1):
    return Evidence(
        file_name="unseen-procurement.pdf",
        page_number=page,
        quote=text,
        chunk_id=chunk_id,
        doc_type="rfp",
        section=section,
        chunk_type=chunk_type,
    )


def test_unseen_rfp_discovers_section_and_table_product_categories():
    items = [
        evidence(
            "Minimum efficiency: 96%",
            "section-category",
            section="Category Alpha Solar Inverter Minimum Requirements",
        ),
        evidence(
            "| Product Category | Required Configuration |\n"
            "|---|---|\n"
            "| Outdoor Sensor | IP67 enclosure |\n"
            "| Indoor Gateway | Dual Ethernet |",
            "table-categories",
            chunk_type="table",
        ),
    ]
    assert set(discover_product_categories(items)) == {
        "Category Alpha Solar Inverter",
        "Outdoor Sensor",
        "Indoor Gateway",
    }


def test_product_coverage_rejects_partial_and_accepts_all_discovered_categories():
    items = [
        evidence("Requirement A", "a", section="Portable Scanner Minimum Requirements"),
        evidence("Requirement B", "b", section="Fixed Scanner Minimum Requirements"),
    ]
    assert "Fixed Scanner" in product_coverage_errors(
        {"Portable Scanner": {"speed": "supported"}}, items
    )[0]
    assert product_coverage_errors(
        {
            "Portable Scanner": {"speed": "supported"},
            "Fixed Scanner": {"speed": "supported"},
        },
        items,
    ) == []


def test_contact_completeness_uses_only_components_visible_in_evidence():
    items = [
        evidence(
            "Contact Information\nProcurement Office\n+1 (555) 201-4400\n"
            "bids@example.gov\nMail responses to 123 Civic Avenue, Suite 4.\n"
            "Questions must be submitted in writing.",
            "contact",
        )
    ]
    errors = contact_completeness_errors(
        {"contact_entity": "Procurement Office"}, items
    )
    assert errors and all(
        name in errors[0]
        for name in ("email", "phone", "mailing_address", "contact_instructions")
    )
    assert contact_completeness_errors(
        {
            "contact_entity": "Procurement Office",
            "email": "bids@example.gov",
            "phone": "+1 (555) 201-4400",
            "mailing_address": "123 Civic Avenue, Suite 4",
            "contact_instructions": "Questions must be submitted in writing.",
        },
        items,
    ) == []


def test_contact_name_requires_a_populated_individual_not_an_office_label():
    office = evidence(
        "Contact Information\nProcurement Services\nbids@example.gov",
        "office",
    )
    assert "name" not in available_contact_components([office])
    person = evidence(
        "Buyer ALVAREZ, MORGAN\nEmail: malvarez@example.gov",
        "person",
    )
    assert "name" in available_contact_components([person])


def test_contact_completeness_accepts_nested_role_preserving_objects():
    items = [evidence(
        "Buyer Morgan Alvarez\nmalvarez@example.gov\n(555) 204-7788\n"
        "Mail responses to 42 Harbor Way, Suite 8. Questions must be emailed.",
        "nested-contact",
    )]
    value = {
        "Procurement Officer": {
            "Name": "Morgan Alvarez",
            "Email": "malvarez@example.gov",
            "Phone": "(555) 204-7788",
            "Mailing Address": "42 Harbor Way, Suite 8",
            "Communication Instructions": "Questions must be emailed.",
        }
    }
    assert contact_completeness_errors(value, items) == []


def test_contact_repair_handles_saved_bid_shape_and_verbatim_submission_address():
    source = evidence(
        "The Offeror may submit a manual response by paper via mail/FedEx/UPS "
        "before the cutoff date and time at the following address:\n\n"
        "Example District\nProcurement Services\n9400 N Central Expwy, Ste 1510\n"
        "Attn: Staff Member and BID NUMBER\nExample City, TX 75231\n\n"
        "Does the vendor agree and understand?",
        "saved-contact-address",
        page=9,
    )
    saved = {
        "Procurement_Services_Department": {
            "Email": "procurement@example.org",
            "Phone": "972-555-0100",
            "Communication_Instruction": "Questions must be submitted in writing.",
        }
    }
    evidence_items = [
        evidence(
            "Contact Information Procurement Services 972-555-0100 "
            "procurement@example.org. Questions must be submitted in writing.",
            "saved-contact",
        ),
        source,
    ]
    repaired, changed = enrich_contact_value_from_evidence(saved, evidence_items)
    assert changed is True
    assert repaired["submission_delivery"]["mailing_address"].endswith(
        "Example City, TX 75231"
    )
    assert "manual response" in repaired["submission_delivery"]["submission_context"]
    assert contact_completeness_errors(repaired, evidence_items) == []


def test_contact_repair_supports_different_nested_contact_document():
    items = [evidence(
        "Buyer: Morgan Alvarez\nEmail: malvarez@harbor.gov\n"
        "Manual paper proposals must be mailed to the following address: "
        "Harbor Purchasing, 42 Harbor Way, Suite 8, Bayview, CA 90001. "
        "Questions must be emailed to the buyer.",
        "unseen-contact-layout",
    )]
    value = [{"buyer": {"name": "Morgan Alvarez", "email": "malvarez@harbor.gov", "communication_instruction": "Questions must be emailed to the buyer."}}]
    # The official object contract is a mapping; list traversal is nevertheless
    # supported for nested role arrays inside that mapping.
    wrapped = {"contacts": value}
    repaired, changed = enrich_contact_value_from_evidence(wrapped, items)
    assert changed is True
    assert repaired["submission_delivery"]["mailing_address"] == (
        "Harbor Purchasing, 42 Harbor Way, Suite 8, Bayview, CA 90001"
    )
    assert contact_completeness_errors(repaired, items) == []


def test_documentation_preserves_conditions_and_alternatives():
    items = [
        evidence(
            "If proposing a consortium, submit a Consortium Agreement or Lead Partner Authorization.",
            "conditional-docs",
        )
    ]
    flattened = documentation_structure_errors(
        ["Consortium Agreement", "Lead Partner Authorization"], items
    )
    assert flattened
    assert documentation_structure_errors(
        [
            "If proposing a consortium, submit a Consortium Agreement or Lead Partner Authorization"
        ],
        items,
    ) == []


class _FakeQdrant:
    def __init__(self, points):
        self.points = points

    def collection_exists(self, _name):
        return True

    def scroll(self, **_kwargs):
        return self.points, None


def test_structured_inventory_scans_all_unseen_category_sections():
    points = [
        SimpleNamespace(
            id="one",
            score=None,
            payload={
                "bid_id": "UnseenBid",
                "file_name": "new.pdf",
                "page_number": 7,
                "doc_type": "specs",
                "section": "Portable Analyzer Minimum Requirements",
                "chunk_type": "text",
                "text": "Battery runtime: 12 hours",
            },
        ),
        SimpleNamespace(
            id="generic",
            score=None,
            payload={
                "bid_id": "UnseenBid",
                "file_name": "new.pdf",
                "page_number": 2,
                "doc_type": "rfp",
                "section": "Scope and Specifications",
                "chunk_type": "text",
                "text": "General terms",
            },
        ),
        SimpleNamespace(
            id="continuation",
            score=None,
            payload={
                "bid_id": "UnseenBid",
                "file_name": "new.pdf",
                "page_number": 8,
                "chunk_index": 0,
                "doc_type": "specs",
                "section": "Specifications",
                "chunk_type": "text",
                "text": "- Operating temperature: -10 C\n- Warranty: five years",
            },
        ),
    ]
    results = structured_product_inventory("UnseenBid", client=_FakeQdrant(points))
    assert [item["chunk_id"] for item in results] == ["one", "continuation"]


def test_product_inventory_discovers_sku_prose_and_late_table_headers():
    points = [
        SimpleNamespace(id="sku", score=None, payload={
            "bid_id": "UnseenBid", "file_name": "quote.pdf", "page_number": 2,
            "doc_type": "quote", "section": "SKU Description", "chunk_type": "text",
            "text": (
                "ZX-4810 Mobile Inspection Console Base; 32 GB memory; 1 TB storage; "
                "3.2 GHz processor; five-year warranty; operating system included."
            ),
        }),
        SimpleNamespace(id="mixed-table", score=None, payload={
            "bid_id": "UnseenBid", "file_name": "request.pdf", "page_number": 4,
            "doc_type": "rfp", "section": None, "chunk_type": "table",
            "text": (
                "| Business Need | Fleet inspection refresh |\n"
                "| Product Name | Product Description | Model # | Qty |\n"
                "| 1. Mobile Inspection Console | Rugged field console | ZX-4810 | 12 |"
            ),
        }),
    ]
    results = structured_product_inventory("UnseenBid", client=_FakeQdrant(points))
    assert {item["chunk_id"] for item in results} == {"sku", "mixed-table"}
    categories = discover_product_categories([
        evidence(points[1].payload["text"], "mixed-table", chunk_type="table")
    ])
    assert categories == ("Mobile Inspection Console",)


def test_product_category_discovery_uses_explicit_sku_prose_without_headings():
    item = evidence(
        "ZX-4810 Mobile Inspection Console Base; 32 GB memory; 1 TB storage; "
        "five-year warranty.",
        "sku-only-prose",
    )
    assert discover_product_categories([item]) == ("Mobile Inspection Console Base",)


def test_product_category_discovery_keeps_unlabeled_service_rows():
    item = evidence(
        "| SV II - Extended Maintenance (Describe required coverage) | "
        "SV II - Extended Maintenance (Describe required coverage) | Deliverables |\n"
        "| Coverage Requirements | Response | Start Date |",
        "service-row",
        chunk_type="table",
    )
    assert discover_product_categories([item]) == ("SV II - Extended Maintenance",)


def test_product_category_discovery_rejects_repeated_functional_area_form_header():
    item = evidence(
        "| FA V - Manufacturer's Extended Warranty (Provide a detailed description) | "
        "FA V - Manufacturer's Extended Warranty (Provide a detailed description) | "
        "FA V - Manufacturer's Extended Warranty (Provide a detailed description) |\n"
        "| Warranty Requirements | Deliverables | Start Date | End Date |",
        "functional-area-header",
        chunk_type="table",
    )
    assert discover_product_categories([item]) == ()


def test_structured_contact_inventory_ignores_blank_forms_and_keeps_roles():
    points = [
        SimpleNamespace(id="buyer", score=None, payload={
            "bid_id": "UnseenBid", "file_name": "request.pdf", "page_number": 1,
            "doc_type": "rfp", "text": "Buyer Morgan Alvarez\nmalvarez@example.gov",
        }),
        SimpleNamespace(id="address", score=None, payload={
            "bid_id": "UnseenBid", "file_name": "request.pdf", "page_number": 5,
            "doc_type": "rfp", "text": (
                "Manual submission address: Procurement Office, 42 Harbor Way, Suite 8, Bay City 10001"
            ),
        }),
        SimpleNamespace(id="blank", score=None, payload={
            "bid_id": "UnseenBid", "file_name": "form.pdf", "page_number": 8,
            "doc_type": "form", "text": "Contact Person:\nEmail:\nPhone:\nAddress:",
        }),
    ]
    results = structured_contact_inventory("UnseenBid", client=_FakeQdrant(points))
    assert [item["chunk_id"] for item in results] == ["buyer", "address"]


def _checkpoint_record(value, change=None):
    record = {
        "field": {
            "field_name": "Due Date",
            "status": "found",
            "value": value,
            "confidence": 1.0,
            "citations": [],
            "candidates": [],
            "reasoning": "supported",
        },
        "evidence": [
            {
                "file_name": "base.pdf",
                "page_number": 1,
                "quote": "Original date",
                "chunk_id": "base",
            },
            {
                "file_name": "amendment.pdf",
                "page_number": 2,
                "quote": "Updated date",
                "chunk_id": "amendment",
                "doc_type": "addendum",
                "addendum_number": 3,
            },
        ],
        "decision": {
            "field_name": "Due Date",
            "status": "passed",
            "reasons": ["supported"],
            "retry_required": False,
            "confidence": 1.0,
        },
        "operational": {
            "status": "passed",
            "terminal": True,
            "reusable": True,
            "reasons": [],
            "reconciliation_verified": True,
        },
        "retry_count": 0,
    }
    if change is not None:
        record["addendum_changes"] = [change.model_dump(mode="json")]
    return record


def test_checkpoint_persists_and_restores_addendum_change(tmp_path):
    change = AddendumChange(
        field_name="Due Date",
        original_value="2030-04-01",
        updated_value="2030-04-15",
        addendum_number=3,
        sources=[
            SourceCitation(file="base.pdf", page=1),
            SourceCitation(file="amendment.pdf", page=2),
        ],
    )
    path = tmp_path / "fields.json"
    save_field_checkpoint(
        path, "same-fingerprint", "Due Date", _checkpoint_record("2030-04-15", change)
    )
    records = reusable_checkpoint_state(json.loads(path.read_text(encoding="utf-8")))
    restored = checkpoint_addendum_changes(records)
    assert restored == [change]


def test_legacy_change_recovery_requires_same_fingerprint_value_and_sources(tmp_path):
    record = _checkpoint_record("2030-04-15")
    change = AddendumChange(
        field_name="Due Date",
        original_value="2030-04-01",
        updated_value="2030-04-15",
        addendum_number=3,
        sources=[
            SourceCitation(file="base.pdf", page=1),
            SourceCitation(file="amendment.pdf", page=2),
        ],
    )
    trace = {
        "checkpoint_fingerprint": "same-fingerprint",
        "final_output": {"addendum_changes": [change.model_dump(mode="json")]},
    }
    (tmp_path / "workflow_trace_old.json").write_text(json.dumps(trace), encoding="utf-8")
    restored = recover_addendum_changes_from_traces(
        tmp_path, "same-fingerprint", {"Due Date": record}
    )
    assert restored == [change]
    assert recover_addendum_changes_from_traces(
        tmp_path, "different-fingerprint", {"Due Date": record}
    ) == []
