
"""Query understanding for RFP hybrid retrieval.

Expands procurement terminology while preserving
the original question and exact identifiers.

No LLM or external API required.
"""

from __future__ import annotations

import argparse
import json
import re


# Canonical concepts and alternative RFP terminology.
# These are generic procurement terms, not bid-specific facts.

QUERY_TERMS = {

    "contract_term": [
        "initial contract term",
        "initial term",
        "length of contract",
        "contract duration",
        "period of performance",
        "contract term",
    ],

    "deadline": [
        "deadline",
        "due date",
        "submission deadline",
        "proposal due date",
        "closing date",
        "bid closing date",
    ],
    "warranty": [
        "warranty",
        "warranty coverage",
        "manufacturer warranty",
        "support period",
        "service coverage",
    ],
    "delivery": [
        "delivery",
        "delivery schedule",
        "shipping",
        "lead time",
        "delivery date",
    ],
    "specifications": [
        "specifications",
        "technical specifications",
        "specs",
        "minimum requirements",
        "technical requirements",
    ],
    "eligibility": [
        "eligibility",
        "qualification requirements",
        "bidder qualifications",
        "vendor eligibility",
    ],
    "pricing": [
        "pricing",
        "price",
        "cost",
        "unit price",
        "pricing schedule",
    ],
    "addendum": [
        "addendum",
        "amendment",
        "revised requirements",
        "solicitation amendment",
    ],
    "processor": [
        "processor",
        "CPU",
        "processor model",
    ],
    "memory": [
        "memory",
        "RAM",
        "system memory",
    ],
}


# Prefer longer matching phrases so that
# "submission deadline" is recognized as one phrase.

PATTERNS = {
    concept: re.compile(
        r"(?<!\w)(?:"
        + "|".join(
            re.escape(term)
            for term in sorted(
                alternatives,
                key=len,
                reverse=True,
            )
        )
        + r")(?!\w)",
        flags=re.IGNORECASE,
    )
    for concept, alternatives in QUERY_TERMS.items()
}


def find_identifiers(question: str) -> list[str]:
    """Identify technical identifiers and numeric codes.

    The original query and every rewritten variant
    retain these values unchanged.
    """

    matches = re.findall(
        r"\b[A-Za-z0-9][A-Za-z0-9/_-]*"
        r"\d[A-Za-z0-9/_-]*\b",
        question,
    )

    return list(dict.fromkeys(matches))


def build_query_plan(
    question: str,
    max_variants: int = 6,
) -> dict:
    """Generate a compact retrieval query plan.

    The original query is used for dense retrieval.

    Keyword retrieval can search multiple variants
    to capture different procurement vocabulary.
    """

    original = " ".join(question.split())

    if not original:
        raise ValueError("Question cannot be empty.")

    if max_variants < 1:
        raise ValueError(
            "max_variants must be at least 1."
        )

    keyword_queries = [original]
    matched_concepts = []
    applied_expansions = {}

    for concept, pattern in PATTERNS.items():

        match = pattern.search(original)

        if not match:
            continue

        matched_concepts.append(concept)

        matched_term = match.group(0)

        alternatives = [
            term
            for term in QUERY_TERMS[concept]
            if term.casefold() != matched_term.casefold()
        ]

        applied_expansions[concept] = alternatives

        for alternative in alternatives:

            # Replace only the recognized procurement
            # phrase. Leave the rest of the question,
            # including product IDs, untouched.
            rewritten = (
                original[:match.start()]
                + alternative
                + original[match.end():]
            )

            if rewritten.casefold() not in {
                query.casefold()
                for query in keyword_queries
            }:
                keyword_queries.append(rewritten)

            if len(keyword_queries) >= max_variants:
                break

        if len(keyword_queries) >= max_variants:
            break

    return {
        "original_query": original,
        "dense_query": original,
        "keyword_queries": keyword_queries,
        "matched_concepts": matched_concepts,
        "expansions": applied_expansions,
        "exact_identifiers": find_identifiers(original),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Preview RFP query expansion"
    )

    parser.add_argument(
        "--question",
        required=True,
    )

    parser.add_argument(
        "--max-variants",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    plan = build_query_plan(
        args.question,
        max_variants=args.max_variants,
    )

    print(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
