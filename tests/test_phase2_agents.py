import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from agents.field_specs import FIELD_SPECS, OFFICIAL_FIELD_LABELS, get_field_spec
from agents.adaptive_retrieval import retrieve_adaptively
from agents.planner import (
    create_execution_plan,
    create_execution_plan_with_trace,
    expand_compact_extraction_plan,
    validate_execution_plan,
)
from agents.llm_client import GeminiGenerationError, TokenUsage
from agents.reconciler import reconcile_field
from agents.report_agent import answer_question
from agents.report_agent import assemble_final_output
from agents.schemas import (
    AddendumChange,
    AgentTrace,
    AnswerCitation,
    CitedAnswer,
    CompactExtractionPlan,
    CriticDecision,
    Evidence,
    EvidenceSearchRequest,
    ExecutionPlan,
    ExecutionTask,
    ExtractedField,
    FieldCandidate,
    ReconciliationDecision,
    QueryRewrite,
)
from agents.validator import validate_field


def extraction_plan(goal="Extract everything"):
    bid_ids = ("ExampleBid",)
    actions = [
        ("index", "ensure_index", "ingestion", "ingestion.ensure_index"),
        ("retrieve", "retrieve", "retrieval_search", "retrieval.hybrid_cohere"),
        ("extract", "extract", "commercial_legal", "gemini.extract"),
        ("reconcile", "reconcile", "addendum_reconciliation", "gemini.reconcile"),
        ("validate", "validate", "validator_critic", "validator.check"),
        ("report", "report", "qa_report", "report.finalize"),
    ]
    tasks = []
    for index, (task_id, action, agent, tool) in enumerate(actions):
        tasks.append(
            ExecutionTask(
                task_id=task_id,
                action=action,
                agent=agent,
                tool=tool,
                bid_ids=bid_ids,
                field_names=OFFICIAL_FIELD_LABELS if action == "extract" else (),
                depends_on=(actions[index - 1][0],) if index else (),
                completion_criteria=f"{action} completed",
            )
        )
    return ExecutionPlan(
        mode="extract",
        goal=goal,
        tasks=tuple(tasks),
        selected_fields=OFFICIAL_FIELD_LABELS,
        stop_condition="Stop after bounded validation",
        rationale="Full extraction needs specialist work.",
    )


def ask_plan(goal, question):
    return ExecutionPlan(
        mode="ask",
        goal=goal,
        tasks=(
            ExecutionTask(
                task_id="index",
                action="ensure_index",
                agent="ingestion",
                tool="ingestion.ensure_index",
                bid_ids=("ExampleBid",),
                completion_criteria="Index ready",
            ),
            ExecutionTask(
                task_id="retrieve",
                action="retrieve",
                agent="retrieval_search",
                tool="retrieval.hybrid_cohere",
                bid_ids=("ExampleBid",),
                question=question,
                depends_on=("index",),
                completion_criteria="Cited passages retrieved",
            ),
            ExecutionTask(
                task_id="answer",
                action="answer",
                agent="qa_report",
                tool="gemini.answer",
                bid_ids=("ExampleBid",),
                question=question,
                depends_on=("retrieve",),
                completion_criteria="Cited answer returned",
            ),
        ),
        selected_fields=(),
        stop_condition="Stop after one grounded answer",
        rationale="A question does not require full extraction.",
    )


def compact_plan():
    return CompactExtractionPlan(
        mode="extract",
        workflow="full_bid_extraction",
        specialist_groups=(
            "commercial_legal",
            "dates_logistics",
            "product_specifications",
        ),
        actions=(
            "ensure_index",
            "retrieve",
            "extract",
            "reconcile",
            "validate",
            "report",
        ),
        tools=(
            "ingestion.ensure_index",
            "retrieval.hybrid_cohere",
            "gemini.extract",
            "gemini.reconcile",
            "validator.check",
            "report.finalize",
        ),
        evidence_requirements=("Use cited bid-isolated evidence",),
        completion_criteria=("Validate every official field",),
        stop_condition="Stop after bounded validation and final report",
        rationale="Use all registered specialists.",
    )


class PlannerClient:
    model = "gemini-2.5-flash-lite"

    def generate_structured(self, prompt, response_model, **kwargs):
        if response_model is CompactExtractionPlan:
            plan = compact_plan()
        elif '"mode": "ask"' in prompt:
            plan = ask_plan("placeholder", "What warranty applies?")
        else:
            plan = extraction_plan("placeholder")
        return SimpleNamespace(
            output=plan,
            usage=TokenUsage(),
            latency_ms=1,
            finish_reason="STOP",
            model="gemini-2.5-flash-lite",
        )


class RetryPlannerClient:
    model = "gemini-2.5-flash-lite"

    def __init__(self, second_error=None):
        self.calls = 0
        self.second_error = second_error

    def generate_structured(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise GeminiGenerationError(
                "truncated",
                classification="truncated_output",
                finish_reason="MAX_TOKENS",
                usage=TokenUsage(total_tokens=100),
                latency_ms=2,
            )
        if self.second_error:
            raise self.second_error
        return SimpleNamespace(
            output=ask_plan("placeholder", "What warranty applies?"),
            usage=TokenUsage(total_tokens=50),
            latency_ms=3,
            finish_reason="STOP",
            model=self.model,
        )


class DecisionClient:
    model = "gemini-2.5-flash-lite"

    def __init__(self, decision):
        self.decision = decision

    def generate_structured(self, *args, **kwargs):
        return SimpleNamespace(output=self.decision)


class Phase2AgentTests(unittest.TestCase):
    def test_compare_plan_preserves_bid_isolation_and_avoids_extraction(self):
        question = "Compare warranty requirements"
        plan = ExecutionPlan(
            mode="compare",
            goal="placeholder",
            tasks=(
                ExecutionTask(
                    task_id="index",
                    action="ensure_index",
                    agent="ingestion",
                    tool="ingestion.ensure_index",
                    bid_ids=("BidA", "BidB"),
                    completion_criteria="Both indexes ready",
                ),
                ExecutionTask(
                    task_id="retrieve_a",
                    action="retrieve",
                    agent="retrieval_search",
                    tool="retrieval.hybrid_cohere",
                    bid_ids=("BidA",),
                    question=question,
                    depends_on=("index",),
                    completion_criteria="BidA evidence retrieved",
                ),
                ExecutionTask(
                    task_id="retrieve_b",
                    action="retrieve",
                    agent="retrieval_search",
                    tool="retrieval.hybrid_cohere",
                    bid_ids=("BidB",),
                    question=question,
                    depends_on=("retrieve_a",),
                    completion_criteria="BidB evidence retrieved",
                ),
                ExecutionTask(
                    task_id="answer",
                    action="answer",
                    agent="qa_report",
                    tool="gemini.answer",
                    bid_ids=("BidA", "BidB"),
                    question=question,
                    depends_on=("retrieve_b",),
                    completion_criteria="Cited comparison returned",
                ),
            ),
            selected_fields=(),
            stop_condition="Stop after one comparison",
        )

        class CompareClient:
            model = "gemini-2.5-flash-lite"

            def generate_structured(self, *args, **kwargs):
                return SimpleNamespace(
                    output=plan,
                    usage=TokenUsage(total_tokens=20),
                    latency_ms=1,
                    finish_reason="STOP",
                    model=self.model,
                )

        result = create_execution_plan(
            {
                "mode": "compare",
                "goal": "Compare bids",
                "bid_ids": ["BidA", "BidB"],
                "question": question,
            },
            llm_client=CompareClient(),
        )
        self.assertEqual(result.selected_fields, ())
        self.assertNotIn("extract", {task.action for task in result.tasks})
        self.assertEqual(
            {bid for task in result.tasks for bid in task.bid_ids},
            {"BidA", "BidB"},
        )
    def test_compact_plan_expands_exact_field_coverage(self):
        plan = expand_compact_extraction_plan(
            compact_plan(),
            goal="Extract all official fields",
            bid_ids=("ExampleBid",),
        )
        extraction_tasks = [task for task in plan.tasks if task.action == "extract"]
        self.assertEqual(len(extraction_tasks), 19)
        self.assertEqual(
            [field for task in extraction_tasks for field in task.field_names],
            [name for name in OFFICIAL_FIELD_LABELS if name != "Bid Summary"],
        )
        for task in extraction_tasks:
            self.assertEqual(
                task.agent,
                get_field_spec(task.field_names[0]).specialist_group,
            )
        self.assertEqual(plan.selected_fields, OFFICIAL_FIELD_LABELS)
        self.assertEqual(
            [task.field_names for task in plan.tasks if task.action == "report"],
            [("Bid Summary",)],
        )

    def test_compact_plan_rejects_unknown_specialist_and_actions(self):
        payload = compact_plan().model_dump()
        payload["specialist_groups"] = (
            "commercial_legal",
            "dates_logistics",
            "unknown_specialist",
        )
        with self.assertRaises(ValidationError):
            CompactExtractionPlan.model_validate(payload)
        payload = compact_plan().model_dump()
        payload["actions"] = (
            "ensure_index",
            "retrieve",
            "extract",
            "reconcile",
            "validate",
            "answer",
        )
        with self.assertRaises(ValidationError):
            CompactExtractionPlan.model_validate(payload)

    def test_planner_retries_truncation_only_once(self):
        client = RetryPlannerClient()
        outcome = create_execution_plan_with_trace(
            {
                "mode": "ask",
                "goal": "Answer warranty",
                "bid_ids": ["ExampleBid"],
                "question": "What warranty applies?",
            },
            llm_client=client,
        )
        self.assertEqual(client.calls, 2)
        self.assertEqual(outcome.trace.metadata["retry_attempts"], 1)
        self.assertEqual(outcome.trace.tokens_used, 150)

        terminal = GeminiGenerationError(
            "still truncated",
            classification="truncated_output",
            finish_reason="MAX_TOKENS",
        )
        client = RetryPlannerClient(second_error=terminal)
        with self.assertRaises(GeminiGenerationError) as raised:
            create_execution_plan_with_trace(
                {
                    "mode": "ask",
                    "goal": "Answer warranty",
                    "bid_ids": ["ExampleBid"],
                    "question": "What warranty applies?",
                },
                llm_client=client,
            )
        self.assertEqual(client.calls, 2)
        self.assertEqual(raised.exception.retry_count, 1)

    def test_planner_does_not_retry_malformed_json(self):
        class MalformedClient:
            model = "gemini-2.5-flash-lite"
            calls = 0

            def generate_structured(self, *args, **kwargs):
                self.calls += 1
                raise GeminiGenerationError(
                    "malformed",
                    classification="malformed_json",
                    finish_reason="STOP",
                )

        client = MalformedClient()
        with self.assertRaises(GeminiGenerationError):
            create_execution_plan(
                {
                    "mode": "ask",
                    "goal": "Answer warranty",
                    "bid_ids": ["ExampleBid"],
                    "question": "What warranty applies?",
                },
                llm_client=client,
            )
        self.assertEqual(client.calls, 1)

    def test_extract_planning_uses_no_gemini_for_known_or_unseen_bids(self):
        class ExplodingClient:
            model = "must-not-be-used"

            def generate_structured(self, *args, **kwargs):
                raise AssertionError("extract planning called Gemini")

        with patch(
            "agents.planner.get_gemini_client",
            side_effect=AssertionError("extract planning initialized Gemini"),
        ):
            for bid_id in ("Bid1", "Bid2", "UnseenProcurementFolder"):
                with self.subTest(bid_id=bid_id):
                    outcome = create_execution_plan_with_trace(
                        {
                            "mode": "extract",
                            "goal": "Extract and validate all official RFP fields",
                            "bid_ids": [bid_id],
                            "question": None,
                        },
                        llm_client=ExplodingClient(),
                    )
                    self.assertEqual(outcome.plan.selected_fields, OFFICIAL_FIELD_LABELS)
                    self.assertEqual(outcome.trace.tool_calls, [])
                    self.assertEqual(outcome.trace.tokens_used, 0)
                    validate_execution_plan(outcome.plan, allowed_bid_ids={bid_id})

    def test_unvalidated_addendum_change_is_excluded_from_final_output(self):
        field = ExtractedField(
            field_name="Due Date",
            status="found",
            value="revised",
            confidence=0.9,
            citations=[Evidence(
                file_name="addendum.pdf",
                page_number=1,
                quote="The date is revised.",
                chunk_id="change",
                doc_type="addendum",
                addendum_number=1,
            )],
        )
        decision = CriticDecision(
            field_name="Due Date",
            status="not_found",
            reasons=["Unsupported final value"],
            confidence=0.0,
        )
        change = AddendumChange(
            field_name="Due Date",
            original_value="old",
            updated_value="revised",
            addendum_number=1,
        )
        output = assemble_final_output(
            bid_id="ExampleBid",
            fields={"Due Date": field},
            decisions={"Due Date": decision},
            changes=[change],
        )
        self.assertEqual(output.addendum_changes, [])

    def test_bid_bond_cross_reference_is_not_treated_as_requirement(self):
        evidence = Evidence(
            file_name="base.pdf",
            page_number=9,
            quote=(
                "The Offeror must comply with any insurance, bid bond, or "
                "liability requirements as noted elsewhere."
            ),
            chunk_id="bond-cross-reference",
            doc_type="rfp",
        )
        field = ExtractedField(
            field_name="Bid Bond Requirement",
            status="found",
            value="A bid bond is required.",
            confidence=0.9,
            citations=[evidence],
        )
        decision = validate_field(
            field=field,
            field_spec=get_field_spec("Bid Bond Requirement"),
            evidence=[evidence],
            retry_count=0,
            max_retries=0,
            semantic_validation=False,
        )
        self.assertEqual(decision.status, "validation_failed")
        self.assertIn("cross-references", decision.reasons[0])

    def test_planner_creates_different_valid_plans_for_different_goals(self):
        client = PlannerClient()
        extract = create_execution_plan(
            {
                "mode": "extract",
                "goal": "Extract everything",
                "bid_ids": ["ExampleBid"],
                "question": None,
            },
            llm_client=client,
        )
        ask = create_execution_plan(
            {
                "mode": "ask",
                "goal": "Answer warranty",
                "bid_ids": ["ExampleBid"],
                "question": "What warranty applies?",
            },
            llm_client=client,
        )
        self.assertEqual(len(extract.selected_fields), 20)
        self.assertEqual(ask.selected_fields, ())
        self.assertNotEqual(
            {task.action for task in extract.tasks},
            {task.action for task in ask.tasks},
        )

    def test_planner_rejects_action_tool_mismatch(self):
        plan = extraction_plan().model_copy(deep=True)
        tasks = list(plan.tasks)
        tasks[1] = tasks[1].model_copy(update={"tool": "gemini.answer"})
        plan = plan.model_copy(update={"tasks": tuple(tasks)})
        with self.assertRaisesRegex(ValueError, "not allowed"):
            validate_execution_plan(plan, allowed_bid_ids={"ExampleBid"})

    def test_addendum_explicit_amendment_wins_with_audit_log(self):
        base = Evidence(
            file_name="base.pdf",
            page_number=1,
            quote="Delivery is required in 60 days.",
            chunk_id="base",
            doc_type="rfp",
        )
        addendum = Evidence(
            file_name="addendum.pdf",
            page_number=2,
            quote="Delivery is revised to 30 days.",
            chunk_id="addendum",
            doc_type="addendum",
            addendum_number=2,
        )
        field = ExtractedField(
            field_name="Delivery Date",
            status="found",
            value="60 days",
            confidence=0.7,
            citations=[base],
            candidates=[FieldCandidate(value="60 days", evidence=base, confidence=0.7)],
        )
        retrieval = SimpleNamespace(
            passages=[SimpleNamespace(evidence=addendum)]
        )
        decision = ReconciliationDecision(
            field_name="Delivery Date",
            amended=True,
            selected_value="30 days",
            original_value="60 days",
            updated_value="30 days",
            original_evidence_rank=1,
            updated_evidence_rank=2,
            addendum_number=2,
            explanation="The addendum explicitly revises delivery.",
            confidence=0.95,
        )
        outcome = reconcile_field(
            bid_id="ExampleBid",
            field_spec=FIELD_SPECS[8],
            field=field,
            evidence=[base],
            llm_client=DecisionClient(decision),
            retrieval_fn=lambda request: retrieval,
        )
        self.assertEqual(outcome.field.value, "30 days")
        self.assertEqual(outcome.field.citations[0].chunk_id, "addendum")
        self.assertEqual(outcome.change.original_value, "60 days")
        self.assertEqual(outcome.change.addendum_number, 2)

    def test_non_amending_addendum_does_not_overwrite(self):
        base = Evidence(
            file_name="base.pdf",
            page_number=1,
            quote="Payment is Net 30.",
            chunk_id="base",
            doc_type="rfp",
        )
        addendum = Evidence(
            file_name="addendum.pdf",
            page_number=1,
            quote="The question deadline is extended.",
            chunk_id="addendum",
            doc_type="addendum",
            addendum_number=1,
        )
        field = ExtractedField(
            field_name="Payment Terms",
            status="found",
            value="Net 30",
            confidence=0.8,
            citations=[base],
        )
        outcome = reconcile_field(
            bid_id="ExampleBid",
            field_spec=FIELD_SPECS[9],
            field=field,
            evidence=[base],
            llm_client=DecisionClient(
                ReconciliationDecision(
                    field_name="Payment Terms",
                    amended=False,
                    selected_value="Net 30",
                    explanation="The addendum concerns another requirement.",
                    confidence=0.9,
                )
            ),
            retrieval_fn=lambda request: SimpleNamespace(
                passages=[SimpleNamespace(evidence=addendum)]
            ),
        )
        self.assertEqual(outcome.field.value, "Net 30")
        self.assertIsNone(outcome.change)

    def test_validator_rejects_unsupported_citation_and_stops_at_limit(self):
        actual = Evidence(
            file_name="base.pdf",
            page_number=1,
            quote="Official title: Supported",
            chunk_id="actual",
            doc_type="rfp",
        )
        invented = actual.model_copy(update={"chunk_id": "invented"})
        field = ExtractedField(
            field_name="Title",
            status="found",
            value="Unsupported",
            confidence=0.7,
            citations=[invented],
        )
        retry = validate_field(
            field=field,
            field_spec=FIELD_SPECS[1],
            evidence=[actual],
            retry_count=0,
            max_retries=2,
            semantic_validation=False,
        )
        terminal = validate_field(
            field=field,
            field_spec=FIELD_SPECS[1],
            evidence=[actual],
            retry_count=2,
            max_retries=2,
            semantic_validation=False,
        )
        self.assertEqual(retry.status, "failed")
        self.assertTrue(retry.retry_required)
        self.assertEqual(terminal.status, "validation_failed")
        self.assertFalse(terminal.retry_required)

    def test_qa_keeps_cross_bid_citations_separate(self):
        one = Evidence(
            file_name="one.pdf",
            page_number=1,
            quote="Warranty is one year.",
            chunk_id="one",
        )
        two = Evidence(
            file_name="two.pdf",
            page_number=2,
            quote="Warranty is three years.",
            chunk_id="two",
        )
        answer = CitedAnswer(
            answer="BidA has one year; BidB has three years.",
            sources=[
                AnswerCitation(bid_id="BidA", file="one.pdf", page=1, chunk_id="one"),
                AnswerCitation(bid_id="BidB", file="two.pdf", page=2, chunk_id="two"),
            ],
            confidence=0.9,
        )
        result = answer_question(
            question="Compare warranties",
            evidence_by_bid={"BidA": [one], "BidB": [two]},
            llm_client=DecisionClient(answer),
        )
        self.assertEqual({source.bid_id for source in result.sources}, {"BidA", "BidB"})

    def test_empty_retrieval_uses_one_validated_query_rewrite(self):
        calls = []

        def search(request):
            calls.append(request.question)
            return SimpleNamespace(
                status="empty" if len(calls) == 1 else "success",
                passages=[] if len(calls) == 1 else ["evidence"],
                warnings=[],
            )

        result = retrieve_adaptively(
            EvidenceSearchRequest(
                bid_id="ExampleBid",
                field_name="Title",
                question="Original question",
            ),
            search_fn=search,
            llm_client=DecisionClient(
                QueryRewrite(
                    question="Improved title search",
                    doc_type="rfp",
                    rationale="Use solicitation terminology.",
                )
            ),
        )
        self.assertEqual(calls, ["Original question", "Improved title search"])
        self.assertEqual(result.status, "success")
        self.assertIn("Adaptive query rewrite", result.warnings[0])


if __name__ == "__main__":
    unittest.main()
