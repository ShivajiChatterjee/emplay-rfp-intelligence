"""Canonical, bid-agnostic field registry for RFP extraction.

The labels in this module are the official Part D output contract.  Search
questions and filter hints are deliberately reusable across solicitations;
sample bid identifiers and answers do not belong here.
"""

from __future__ import annotations

from agents.schemas import FieldSpec


OFFICIAL_FIELD_LABELS: tuple[str, ...] = (
    "Bid Number",
    "Title",
    "Due Date",
    "Bid Submission Type",
    "Term of Bid",
    "Pre Bid Meeting",
    "Installation",
    "Bid Bond Requirement",
    "Delivery Date",
    "Payment Terms",
    "Any Additional Documentation Required",
    "MFG for Registration",
    "Contract or Cooperative to use",
    "Model_no",
    "Part_no",
    "Product",
    "contact_info",
    "company_name",
    "Bid Summary",
    "Product Specification",
)


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        field_name="Bid Number",
        questions=(
            "What is the solicitation, bid, RFP, RFQ, or procurement number?",
            "Which identifier does the issuing organization use for this bid?",
        ),
        expected_type="string",
        specialist_group="commercial_legal",
        search_filters={"preferred_doc_types": ["rfp", "bid_page"]},
    ),
    FieldSpec(
        field_name="Title",
        questions=(
            "What is the official title of the solicitation or bid?",
            "What project or procurement name appears with the bid number?",
        ),
        expected_type="string",
        specialist_group="commercial_legal",
        search_filters={"preferred_doc_types": ["rfp", "bid_page"]},
    ),
    FieldSpec(
        field_name="Due Date",
        questions=(
            "What is the final current bid submission deadline, including date, time, and time zone?",
            "Did an addendum revise the proposal due date or closing time?",
        ),
        expected_type="datetime",
        specialist_group="dates_logistics",
        search_filters={
            "preferred_doc_types": ["addendum", "bid_page", "rfp"],
            "include_addenda": True,
        },
    ),
    FieldSpec(
        field_name="Bid Submission Type",
        questions=(
            "How must the bid be submitted: electronically, by portal, email, mail, or physical delivery?",
            "What submission method and destination are required?",
        ),
        expected_type="string",
        specialist_group="dates_logistics",
        search_filters={"preferred_doc_types": ["rfp", "bid_page", "addendum"]},
    ),
    FieldSpec(
        field_name="Term of Bid",
        questions=(
            "What is the initial contract or bid term and are renewal periods stated?",
            "What period of performance or contract duration is required?",
        ),
        expected_type="string",
        specialist_group="commercial_legal",
        search_filters={"preferred_doc_types": ["rfp", "addendum"]},
    ),
    FieldSpec(
        field_name="Pre Bid Meeting",
        questions=(
            "Is there a pre-bid or pre-proposal meeting, and what are its date, time, location, and attendance requirements?",
            "Is the pre-bid conference mandatory or optional?",
        ),
        expected_type="string",
        specialist_group="dates_logistics",
        search_filters={"preferred_doc_types": ["rfp", "bid_page", "addendum"]},
    ),
    FieldSpec(
        field_name="Installation",
        questions=(
            "What installation, setup, deployment, configuration, or removal work is required?",
            "Who is responsible for installation and where must it occur?",
        ),
        expected_type="string",
        specialist_group="dates_logistics",
        search_filters={"preferred_doc_types": ["rfp", "specs", "addendum"]},
    ),
    FieldSpec(
        field_name="Bid Bond Requirement",
        questions=(
            "Is a bid bond, proposal security, or other bid guarantee required?",
            "What amount, percentage, form, and submission conditions apply to the bid bond?",
        ),
        expected_type="string",
        specialist_group="commercial_legal",
        search_filters={"preferred_doc_types": ["rfp", "addendum"]},
    ),
    FieldSpec(
        field_name="Delivery Date",
        questions=(
            "What delivery date, delivery window, lead time, or days-after-award requirement applies?",
            "Are there milestones or separate delivery requirements for any items?",
        ),
        expected_type="string",
        specialist_group="dates_logistics",
        search_filters={"preferred_doc_types": ["rfp", "specs", "addendum"]},
    ),
    FieldSpec(
        field_name="Payment Terms",
        questions=(
            "What payment terms, invoicing conditions, retainage, or payment schedule are stated?",
            "When will valid invoices be paid and what documentation is needed for payment?",
        ),
        expected_type="string",
        specialist_group="commercial_legal",
        search_filters={"preferred_doc_types": ["rfp", "addendum"]},
    ),
    FieldSpec(
        field_name="Any Additional Documentation Required",
        questions=(
            "Which affidavits, certifications, forms, licenses, references, or supporting documents must a bidder submit?",
            "What additional documentation is mandatory with the response or after award?",
        ),
        expected_type="list",
        specialist_group="commercial_legal",
        search_filters={
            "preferred_doc_types": ["rfp", "affidavit", "addendum"],
            "include_addenda": True,
        },
    ),
    FieldSpec(
        field_name="MFG for Registration",
        questions=(
            "Is manufacturer registration, authorization, certification, or reseller status required?",
            "Which manufacturer must the bidder be registered or authorized with?",
        ),
        expected_type="string",
        specialist_group="commercial_legal",
        search_filters={"preferred_doc_types": ["rfp", "specs", "addendum"]},
    ),
    FieldSpec(
        field_name="Contract or Cooperative to use",
        questions=(
            "Does the solicitation identify a contract, master agreement, cooperative purchasing vehicle, or cooperative contract to use?",
            "What contract or cooperative name and number govern this procurement?",
        ),
        expected_type="string",
        specialist_group="commercial_legal",
        search_filters={"preferred_doc_types": ["rfp", "bid_page", "addendum"]},
    ),
    FieldSpec(
        field_name="Model_no",
        questions=(
            "What manufacturer model number or model identifier is requested for each product?",
            "Which exact model designation appears in the specifications or item schedule?",
        ),
        expected_type="list",
        specialist_group="product_specifications",
        search_filters={"preferred_doc_types": ["specs", "rfp", "addendum"]},
    ),
    FieldSpec(
        field_name="Part_no",
        questions=(
            "What part number, SKU, catalog number, or manufacturer part identifier is requested for each item?",
            "Which exact part identifiers appear in the specifications or pricing schedule?",
        ),
        expected_type="list",
        specialist_group="product_specifications",
        search_filters={"preferred_doc_types": ["specs", "rfp", "addendum"]},
    ),
    FieldSpec(
        field_name="Product",
        questions=(
            "Which products, equipment, materials, or services are being procured?",
            "What item names and requested quantities define the procurement scope?",
        ),
        expected_type="list",
        specialist_group="product_specifications",
        search_filters={"preferred_doc_types": ["specs", "rfp", "bid_page"]},
    ),
    FieldSpec(
        field_name="contact_info",
        questions=(
            "What official procurement contact name, email address, phone number, and mailing address are provided?",
            "Who should receive bidder questions or communications about this solicitation?",
        ),
        expected_type="object",
        specialist_group="commercial_legal",
        search_filters={"preferred_doc_types": ["rfp", "bid_page", "addendum"]},
    ),
    FieldSpec(
        field_name="company_name",
        questions=(
            "What organization, agency, district, or company issued the solicitation?",
            "Which legal entity is the buyer or contracting authority?",
        ),
        expected_type="string",
        specialist_group="commercial_legal",
        search_filters={"preferred_doc_types": ["rfp", "bid_page"]},
    ),
    FieldSpec(
        field_name="Bid Summary",
        questions=(
            "What is the evidence-supported purpose, scope, key deliverables, schedule, and important bidder obligations of this solicitation?",
            "Summarize the bid in three to six factual sentences without adding unsupported conclusions.",
        ),
        expected_type="summary",
        specialist_group="qa_report",
        search_filters={
            "preferred_doc_types": ["rfp", "bid_page", "specs", "addendum"],
            "include_addenda": True,
        },
    ),
    FieldSpec(
        field_name="Product Specification",
        questions=(
            "What technical, functional, performance, compatibility, warranty, and configuration requirements apply to each requested product?",
            "Which minimum product specifications and accepted alternatives are explicitly stated?",
        ),
        expected_type="object",
        specialist_group="product_specifications",
        search_filters={
            "preferred_doc_types": ["specs", "rfp", "addendum"],
            "include_addenda": True,
        },
    ),
)


_FIELD_SPEC_BY_NAME = {spec.field_name: spec for spec in FIELD_SPECS}


def get_field_specs() -> tuple[FieldSpec, ...]:
    """Return the immutable canonical field registry."""

    return FIELD_SPECS


def get_field_spec(field_name: str) -> FieldSpec:
    """Return one official field specification by its exact label."""

    try:
        return _FIELD_SPEC_BY_NAME[field_name]
    except KeyError as exc:
        raise KeyError(f"Unknown official field: {field_name}") from exc
