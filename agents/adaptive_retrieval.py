"""Adaptive wrapper around the existing Qdrant/Cohere retrieval agent."""

from __future__ import annotations

import json

from agents.llm_client import GenerationSettings, GeminiClient, get_gemini_client
from agents.prompts import ORCHESTRATOR_PLANNER_PROMPT
from agents.schemas import EvidenceSearchRequest, QueryRewrite


def propose_query_rewrite(
    request: EvidenceSearchRequest,
    warnings: list[str],
    llm_client: GeminiClient | None = None,
) -> QueryRewrite:
    client = llm_client or get_gemini_client()
    prompt = f"""
Reformulate one unsuccessful RFP evidence search. Keep the same intent and never
invent bid-specific facts. You may choose one supported metadata filter only when
the question clearly warrants it.

{json.dumps({
    "field_name": request.field_name,
    "original_question": request.question,
    "doc_type": request.doc_type,
    "addendum_number": request.addendum_number,
    "file_name": request.file_name,
    "retrieval_warnings": warnings,
}, ensure_ascii=False)}

Allowed document types: rfp, addendum, specs, affidavit, bid_page.
Do not include or change a bid identifier in the rewritten question.
""".strip()
    result = client.generate_structured(
        prompt,
        QueryRewrite,
        system_instruction=ORCHESTRATOR_PLANNER_PROMPT,
        generation=GenerationSettings(temperature=0, max_output_tokens=300),
    )
    rewrite = result.output
    if rewrite.doc_type not in {
        None,
        "rfp",
        "addendum",
        "specs",
        "affidavit",
        "bid_page",
    }:
        raise ValueError(f"Unsupported rewritten doc_type: {rewrite.doc_type}")
    return rewrite


def retrieve_adaptively(
    request: EvidenceSearchRequest,
    *,
    llm_client: GeminiClient | None = None,
    search_fn=None,
    allow_rewrite: bool = True,
) -> RetrievalResult:
    """Retrieve once, then perform at most one validated semantic rewrite."""

    if search_fn is None:
        from agents.retrieval_agent import search_evidence

        active_search = search_evidence
    else:
        active_search = search_fn
    result = active_search(request)
    if result.passages or result.status == "error" or not allow_rewrite:
        return result

    rewrite = propose_query_rewrite(
        request,
        result.warnings,
        llm_client=llm_client,
    )
    rewritten_request = request.model_copy(
        update={
            "question": rewrite.question,
            "doc_type": rewrite.doc_type,
            "addendum_number": rewrite.addendum_number,
            "file_name": rewrite.file_name,
        }
    )
    retried = active_search(rewritten_request)
    retried.warnings.insert(
        0,
        f"Adaptive query rewrite used: {rewrite.rationale or rewrite.question}",
    )
    return retried
