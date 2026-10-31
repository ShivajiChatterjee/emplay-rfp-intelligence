import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.field_specs import OFFICIAL_FIELD_LABELS, get_field_spec
from agents.report_agent import assemble_final_output
from agents.schemas import (
    AgentTrace,
    AddendumChange,
    CitedAnswer,
    CompactExtractionPlan,
    CriticDecision,
    Evidence,
    ExecutionPlan,
    ExecutionTask,
    ExtractedField,
    FieldRunRecord,
    PlanningOutcome,
    SourceCitation,
)
from agents.planner import expand_compact_extraction_plan
from workflow.graph import run_workflow
from workflow.routing import route_after_ingestion
from workflow.services import WorkflowServices
from workflow.state import WorkflowRequest


def make_plan(mode, fields=()):
    action = "report" if mode == "extract" else "answer"
    tool = "report.finalize" if mode == "extract" else "gemini.answer"
    return ExecutionPlan(
        mode=mode,
        goal="mock goal",
        tasks=(
            ExecutionTask(
                task_id="mock",
                action=action,
                agent="qa_report",
                tool=tool,
                bid_ids=("ExampleBid",),
                question="Mock question" if mode != "extract" else None,
                completion_criteria="Mock completion",
            ),
        ),
        selected_fields=tuple(fields),
        stop_condition="Bounded completion",
    )


class MockServices:
    def __init__(
        self,
        fields,
        reject_field=None,
        always_reject=False,
        addendum_field=None,
        incomplete_contact=False,
    ):
        self.fields = tuple(fields)
        self.reject_field = reject_field
        self.always_reject = always_reject
        self.addendum_field = addendum_field
        self.incomplete_contact = incomplete_contact
        self.retrieve_calls = []
        self.extract_calls = []
        self.validate_calls = []
        self.rewrite_flags = []
        self.semantic_flags = []

    def plan(self, request):
        return make_plan(request["mode"], self.fields)

    def ensure_index(self, path, output_dir):
        return {
            "bid_id": path.name,
            "status": "ready",
            "documents": 1,
            "chunks": 1,
            "warnings": [],
            "errors": [],
        }

    def retrieve(self, request, allow_rewrite=True):
        self.rewrite_flags.append(allow_rewrite)
        self.retrieve_calls.append(
            (request.bid_id, request.field_name, request.question)
        )
        value = f"Supported {request.field_name}"
        evidence = Evidence(
            file_name=f"{request.bid_id}.pdf",
            page_number=1,
            quote=f"Evidence for {request.field_name}: {value}",
            chunk_id=f"{request.bid_id}-{request.field_name}-{len(self.retrieve_calls)}",
            doc_type="rfp",
        )
        return SimpleNamespace(
            bid_id=request.bid_id,
            field_name=request.field_name,
            question=request.question,
            status="success",
            passages=[SimpleNamespace(rank=1, score=0.9, evidence=evidence)],
            warnings=[],
            trace=AgentTrace(
                agent="retrieval_agent",
                action="search",
                status="success",
            ),
        )

    def extract(self, retrieval):
        self.extract_calls.append(retrieval.field_name)
        evidence = retrieval.passages[0].evidence
        spec = get_field_spec(retrieval.field_name)
        value = {
            "list": [f"Supported {retrieval.field_name}"],
            "object": {"detail": f"Supported {retrieval.field_name}"},
        }.get(spec.expected_type, f"Supported {retrieval.field_name}")
        if retrieval.field_name == "contact_info" and self.incomplete_contact:
            value = (
                {
                    "name": None, "email": None, "phone": None,
                    "mailing_address": "9400 N Central Expwy",
                }
                if len(retrieval.passages) == 1
                else {
                    "name": "Procurement Services",
                    "email": "procurement@example.org",
                    "phone": "972-555-0100",
                    "mailing_address": "9400 N Central Expwy",
                }
            )
        field = ExtractedField(
            field_name=retrieval.field_name,
            status="found",
            value=value,
            confidence=0.8,
            citations=[evidence.model_copy(update={"quote": evidence.quote})],
            reasoning="Grounded mock extraction",
        )
        return field, AgentTrace(
            agent=spec.specialist_group,
            action="extract",
            status="success",
        )

    def reconcile(self, **kwargs):
        if kwargs["field"].field_name == self.addendum_field:
            base = kwargs["evidence"][0]
            addendum = base.model_copy(
                update={
                    "file_name": "amendment.pdf",
                    "page_number": 2,
                    "chunk_id": "explicit-amendment",
                    "quote": "This requirement is explicitly revised.",
                    "doc_type": "addendum",
                    "addendum_number": 2,
                }
            )
            field = kwargs["field"].model_copy(
                update={
                    "value": f"Revised {self.addendum_field}",
                    "citations": [addendum],
                    "reasoning": "An explicit later amendment applies.",
                }
            )
            return SimpleNamespace(
                field=field,
                evidence=[base, addendum],
                change=AddendumChange(
                    field_name=self.addendum_field,
                    original_value=f"Supported {self.addendum_field}",
                    updated_value=f"Revised {self.addendum_field}",
                    addendum_number=2,
                    sources=[
                        SourceCitation(file=base.file_name, page=base.page_number),
                        SourceCitation(file=addendum.file_name, page=addendum.page_number),
                    ],
                    notes="Explicit synthetic amendment.",
                ),
                warning=None,
                traces=[AgentTrace(
                    agent="addendum_reconciliation",
                    action="reconcile_field",
                    status="success",
                    tokens_used=7,
                )],
            )
        return SimpleNamespace(
            field=kwargs["field"],
            evidence=kwargs["evidence"],
            change=None,
            warning=None,
            traces=[AgentTrace(
                agent="addendum_reconciliation",
                action="reconcile_field",
                status="success",
                tokens_used=7,
            )],
        )

    def validate(self, **kwargs):
        field = kwargs["field"]
        attempt = kwargs["retry_count"]
        self.validate_calls.append((field.field_name, attempt))
        self.semantic_flags.append(kwargs.get("semantic_validation", True))
        reject = field.field_name == self.reject_field and (
            self.always_reject or attempt == 0
        )
        if reject:
            return CriticDecision(
                field_name=field.field_name,
                status="failed",
                reasons=["Targeted retry required"],
                revised_question=f"Revised search for {field.field_name}",
                retry_required=True,
                confidence=0.2,
            )
        return CriticDecision(
            field_name=field.field_name,
            status="passed",
            reasons=["Grounded"],
            retry_required=False,
            confidence=0.8,
        )

    def summarize(self, **kwargs):
        evidence = next(iter(kwargs["fields"].values())).citations[0]
        return ExtractedField(
            field_name="Bid Summary",
            status="found",
            value="Sentence one. Sentence two. Sentence three.",
            confidence=0.8,
            citations=[evidence],
            reasoning="Grounded summary",
        )

    def answer(self, **kwargs):
        return CitedAnswer(
            answer="Mock cited answer" if any(kwargs["evidence_by_bid"].values()) else "Not found in documents",
            confidence=0.8,
        )

    assemble = staticmethod(assemble_final_output)

    def bundle(self):
        return WorkflowServices(
            plan=self.plan,
            ensure_index=self.ensure_index,
            retrieve=self.retrieve,
            extract=self.extract,
            reconcile=self.reconcile,
            validate=self.validate,
            answer=self.answer,
            summarize=self.summarize,
            assemble=self.assemble,
        )


class WorkflowGraphTests(unittest.TestCase):
    def test_bid1_style_resume_with_19_reusable_sources_routes_only_to_summary(self):
        plan = expand_compact_extraction_plan(
            CompactExtractionPlan(
                mode="extract", workflow="full_bid_extraction",
                specialist_groups=(
                    "commercial_legal", "dates_logistics", "product_specifications",
                ),
                actions=("ensure_index", "retrieve", "extract", "reconcile", "validate", "report"),
                tools=(
                    "ingestion.ensure_index", "retrieval.hybrid_cohere", "gemini.extract",
                    "gemini.reconcile", "validator.check", "report.finalize",
                ),
                evidence_requirements=("Use cited evidence",),
                completion_criteria=("Validate all fields",),
                stop_condition="Stop after final validation",
            ),
            goal="Extract all official fields", bid_ids=("Bid1",),
        )
        source_fields = [name for name in OFFICIAL_FIELD_LABELS if name != "Bid Summary"]
        state = {
            "index_status": {"Bid1": {"status": "ready"}},
            "plan": plan,
            "request": {"bid_ids": ["Bid1"]},
            "field_statuses": {
                name: FieldRunRecord(
                    status="passed", terminal=True, reusable=True,
                    reconciliation_verified=True,
                )
                for name in source_fields
            },
        }
        self.assertEqual(route_after_ingestion(state), "report")

    def test_bid2_style_resume_routes_only_unresolved_source_fields(self):
        plan = expand_compact_extraction_plan(
            CompactExtractionPlan(
                mode="extract", workflow="full_bid_extraction",
                specialist_groups=(
                    "commercial_legal", "dates_logistics", "product_specifications",
                ),
                actions=("ensure_index", "retrieve", "extract", "reconcile", "validate", "report"),
                tools=(
                    "ingestion.ensure_index", "retrieval.hybrid_cohere", "gemini.extract",
                    "gemini.reconcile", "validator.check", "report.finalize",
                ),
                evidence_requirements=("Use cited evidence",),
                completion_criteria=("Validate all fields",),
                stop_condition="Stop after final validation",
            ),
            goal="Extract all official fields", bid_ids=("Bid2",),
        )
        reusable = {
            "Bid Bond Requirement", "Bid Number", "Bid Submission Type", "company_name",
            "contact_info", "Due Date", "Part_no", "Pre Bid Meeting", "Product",
            "Term of Bid", "Title",
        }
        expected = {
            name for name in OFFICIAL_FIELD_LABELS
            if name not in reusable and name != "Bid Summary"
        }
        state = {
            "run_id": "offline-bid2-resume",
            "index_status": {"Bid2": {"status": "ready"}},
            "plan": plan,
            "request": {"bid_ids": ["Bid2"], "max_retries": 0},
            "field_statuses": {
                name: FieldRunRecord(
                    status="passed", terminal=True, reusable=True,
                    reconciliation_verified=True,
                )
                for name in reusable
            },
        }
        sends = route_after_ingestion(state)
        self.assertEqual(
            {send.arg["worker_field_name"] for send in sends}, expected
        )

    def test_compact_planning_reaches_readiness_and_all_specialist_routes(self):
        compact = CompactExtractionPlan(
            mode="extract",
            workflow="full_bid_extraction",
            specialist_groups=(
                "commercial_legal",
                "dates_logistics",
                "product_specifications",
            ),
            actions=(
                "ensure_index", "retrieve", "extract",
                "reconcile", "validate", "report",
            ),
            tools=(
                "ingestion.ensure_index", "retrieval.hybrid_cohere",
                "gemini.extract", "gemini.reconcile",
                "validator.check", "report.finalize",
            ),
            evidence_requirements=("Use cited evidence",),
            completion_criteria=("Validate all fields",),
            stop_condition="Stop after final report",
        )
        plan = expand_compact_extraction_plan(
            compact,
            goal="Extract all official fields",
            bid_ids=("ExampleBid",),
        )
        services = MockServices(
            tuple(name for name in OFFICIAL_FIELD_LABELS if name != "Bid Summary")
        )
        bundle = services.bundle()
        bundle.plan = lambda request: PlanningOutcome(
            plan=plan,
            compact_plan=compact,
            trace=AgentTrace(
                agent="orchestrator_planner",
                action="create_execution_plan",
                status="success",
                output_summary={
                    "selected_specialist_groups": list(compact.specialist_groups)
                },
                metadata={
                    "model": "gemini-2.5-flash-lite",
                    "finish_reason": "STOP",
                    "token_usage": {"total_tokens": 20},
                    "retry_attempts": 0,
                    "structured_output_validation": "passed",
                },
            ),
        )
        request = WorkflowRequest(
            mode="extract",
            bid_folders=(Path("data/ExampleBid"),),
            max_retries=0,
        )
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = run_workflow(request, services=bundle, export=False)
        self.assertEqual(len(services.extract_calls), 19)
        self.assertEqual(result["index_status"]["ExampleBid"]["status"], "ready")
        self.assertEqual(len(result["final_output"].fields), 20)
        self.assertEqual(result["agent_traces"][0].agent, "orchestrator_planner")
    def run_extract(self, services, max_retries=2):
        request = WorkflowRequest(
            mode="extract",
            bid_folders=(Path("data/ExampleBid"),),
            output_dir=Path("outputs"),
            max_retries=max_retries,
        )
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            return run_workflow(
                request,
                services=services.bundle(),
                export=False,
            )

    def test_parallel_specialists_merge_all_fields_without_loss(self):
        fields = tuple(name for name in OFFICIAL_FIELD_LABELS if name != "Bid Summary")
        services = MockServices(fields)
        result = self.run_extract(services)
        self.assertEqual(set(result["fields"]), set(OFFICIAL_FIELD_LABELS))
        self.assertEqual(result["final_output"].validation.passed, 20)
        self.assertEqual(len(services.extract_calls), 19)
        worker_agents = {
            event["summary"]["field_name"]: event["agent"]
            for event in result["events"]
            if event["action"] == "retrieve_extract_reconcile"
        }
        for field_name in OFFICIAL_FIELD_LABELS:
            if field_name != "Bid Summary":
                self.assertEqual(
                    worker_agents[field_name],
                    get_field_spec(field_name).specialist_group,
                )

    def test_controlled_run_disables_rewrite_and_semantic_critic(self):
        services = MockServices(("Title",))
        request = WorkflowRequest(
            mode="extract",
            bid_folders=(Path("data/ExampleBid"),),
            max_retries=0,
            allow_query_rewrite=False,
            semantic_validation=False,
        )
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = run_workflow(
                request,
                services=services.bundle(),
                export=False,
            )
        self.assertEqual(services.rewrite_flags, [False])
        self.assertTrue(services.semantic_flags)
        self.assertFalse(any(services.semantic_flags))
        self.assertTrue(any(
            trace.agent == "addendum_reconciliation"
            for trace in result["agent_traces"]
        ))

    def test_incomplete_contact_object_triggers_targeted_supplemental_retrieval(self):
        services = MockServices(("contact_info",), incomplete_contact=True)
        result = self.run_extract(services, max_retries=0)
        self.assertEqual(services.extract_calls.count("contact_info"), 1)
        contact_queries = [item for item in services.retrieve_calls if item[1] == "contact_info"]
        self.assertEqual(len(contact_queries), 2)
        self.assertIn("Who should receive bidder questions", contact_queries[1][2])
        self.assertEqual(
            result["fields"]["contact_info"].value["email"],
            "procurement@example.org",
        )

    def test_validator_rejection_triggers_real_retrieval_and_extraction_retry(self):
        services = MockServices(("Title",), reject_field="Title")
        result = self.run_extract(services)
        title_calls = [call for call in services.retrieve_calls if call[1] == "Title"]
        self.assertEqual(len(title_calls), 2)
        self.assertEqual(services.extract_calls.count("Title"), 2)
        self.assertEqual(result["retry_counts"]["Title"], 1)
        self.assertIn("Revised search", title_calls[1][2])
        attempts = [
            event["summary"]["attempt"]
            for event in result["events"]
            if event["action"] == "retrieve_extract_reconcile"
            and event["summary"]["field_name"] == "Title"
        ]
        self.assertEqual(attempts, [0, 1])

    def test_retry_counter_stops_after_two_retries(self):
        services = MockServices(
            ("Title",),
            reject_field="Title",
            always_reject=True,
        )
        result = self.run_extract(services, max_retries=2)
        self.assertEqual(services.extract_calls.count("Title"), 3)
        self.assertEqual(result["retry_counts"]["Title"], 2)
        self.assertEqual(result["validation"]["Title"].status, "validation_failed")
        self.assertIsNone(result.get("final_output"))
        self.assertEqual(result["validation"]["Title"].reasons, ["Targeted retry required"])

    def test_summary_validation_failure_is_reported_and_blocks_publication(self):
        fields = tuple(name for name in OFFICIAL_FIELD_LABELS if name != "Bid Summary")
        services = MockServices(fields)
        bundle = services.bundle()
        regular_validate = bundle.validate

        def reject_summary(**kwargs):
            if kwargs["field"].field_name == "Bid Summary":
                return CriticDecision(
                    field_name="Bid Summary", status="validation_failed",
                    reasons=["Summary claim lacks supporting evidence"],
                    retry_required=False, confidence=0.0,
                )
            return regular_validate(**kwargs)

        bundle.validate = reject_summary
        request = WorkflowRequest(
            mode="extract", bid_folders=(Path("data/ExampleBid"),), max_retries=0,
        )
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = run_workflow(request, services=bundle, export=False)
        self.assertIsNone(result.get("final_output"))
        self.assertEqual(
            result["field_statuses"]["Bid Summary"].status,
            "validation_failed",
        )
        self.assertIn(
            "Bid Summary: Summary claim lacks supporting evidence",
            result["errors"],
        )

    def test_ask_route_skips_extraction_and_retrieves_per_bid(self):
        services = MockServices(())
        request = WorkflowRequest(
            mode="ask",
            bid_folders=(Path("data/ExampleBid"),),
            question="What is required?",
        )
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = run_workflow(
                request,
                services=services.bundle(),
                export=False,
            )
        self.assertEqual(len(services.retrieve_calls), 1)
        self.assertEqual(services.extract_calls, [])
        self.assertEqual(result["qa_answer"].answer, "Mock cited answer")

    def test_mocked_end_to_end_exports_trace_with_addendum_and_retry(self):
        services = MockServices(
            ("Title", "Due Date"),
            reject_field="Title",
            addendum_field="Due Date",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            request = WorkflowRequest(
                mode="extract",
                bid_folders=(Path("data/ExampleBid"),),
                output_dir=output_dir,
                max_retries=2,
            )
            with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
                result = run_workflow(
                    request,
                    services=services.bundle(),
                    export=True,
                )

            self.assertEqual(result["retry_counts"]["Title"], 1)
            self.assertEqual(len(result["addendum_changes"]), 1)
            self.assertEqual(
                result["addendum_changes"][0].field_name,
                "Due Date",
            )
            checkpoint = json.loads(
                (output_dir / "ExampleBid_field_checkpoints.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                checkpoint["fields"]["Due Date"]["addendum_changes"][0]["updated_value"],
                "Revised Due Date",
            )
            traces = list(output_dir.glob("workflow_trace_*.json"))
            self.assertEqual(len(traces), 1)
            trace = json.loads(traces[0].read_text(encoding="utf-8"))
            actions = [event["action"] for event in trace["events"]]
            self.assertIn("plan", actions)
            self.assertIn("retrieve_extract_reconcile", actions)
            self.assertIn("validate_field", actions)
            self.assertFalse((output_dir / "ExampleBid_extraction.json").exists())
            manifests = list(output_dir.glob("run_manifest_*.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_status"], "partial")
            self.assertFalse(manifest["final_submission_published"])

    def test_localized_empty_retrieval_does_not_crash_graph(self):
        services = MockServices(
            ("Title",),
            reject_field="Title",
            always_reject=True,
        )

        def empty_retrieve(request, allow_rewrite=True):
            return SimpleNamespace(
                bid_id=request.bid_id,
                field_name=request.field_name,
                question=request.question,
                status="empty",
                passages=[],
                warnings=["No evidence"],
                trace=None,
            )

        bundle = services.bundle()
        bundle.retrieve = empty_retrieve
        request = WorkflowRequest(
            mode="extract",
            bid_folders=(Path("data/ExampleBid"),),
            max_retries=2,
        )
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = run_workflow(request, services=bundle, export=False)
        self.assertIsNone(result.get("final_output"))
        self.assertEqual(result["retry_counts"]["Title"], 2)


if __name__ == "__main__":
    unittest.main()
