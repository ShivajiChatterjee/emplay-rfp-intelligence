"""Compact LLM planning with deterministic full-extraction expansion."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agents.field_specs import FIELD_SPECS, OFFICIAL_FIELD_LABELS
from agents.llm_client import (
    configured_gemini_model,
    GeminiClient,
    GeminiGenerationError,
    GenerationSettings,
    TokenUsage,
    get_gemini_client,
)
from agents.prompts import ORCHESTRATOR_PLANNER_PROMPT
from agents.schemas import (
    AgentTrace,
    CompactExtractionPlan,
    ExecutionPlan,
    ExecutionTask,
    PlanningOutcome,
)


ACTION_TOOL_MAP = {
    "ensure_index": "ingestion.ensure_index",
    "retrieve": "retrieval.hybrid_cohere",
    "extract": "gemini.extract",
    "reconcile": "gemini.reconcile",
    "validate": "validator.check",
    "answer": "gemini.answer",
    "report": "report.finalize",
}

ALLOWED_AGENTS = {
    "orchestrator_planner",
    "ingestion",
    "retrieval_search",
    "dates_logistics",
    "commercial_legal",
    "product_specifications",
    "addendum_reconciliation",
    "validator_critic",
    "qa_report",
}

EXTRACTION_ACTIONS = (
    "ensure_index",
    "retrieve",
    "extract",
    "reconcile",
    "validate",
    "report",
)


def _ensure_index_precondition(
    plan: ExecutionPlan,
    bid_ids: tuple[str, ...],
) -> ExecutionPlan:
    if any(task.action == "ensure_index" for task in plan.tasks):
        return plan
    task_id = "required_index_readiness"
    index_task = ExecutionTask(
        task_id=task_id,
        action="ensure_index",
        agent="ingestion",
        tool="ingestion.ensure_index",
        bid_ids=bid_ids,
        completion_criteria="Every requested bid is indexed or has an explicit error",
    )
    adjusted = []
    for task in plan.tasks:
        if task.action == "retrieve" and task_id not in task.depends_on:
            task = task.model_copy(update={"depends_on": (task_id, *task.depends_on)})
        adjusted.append(task)
    return plan.model_copy(update={"tasks": (index_task, *adjusted)})


def validate_execution_plan(
    plan: ExecutionPlan,
    *,
    allowed_bid_ids: set[str],
) -> ExecutionPlan:
    """Reject invented tools, specialists, fields, bids, and dependencies."""

    official = set(OFFICIAL_FIELD_LABELS)
    if not set(plan.selected_fields).issubset(official):
        unknown = sorted(set(plan.selected_fields) - official)
        raise ValueError(f"Planner selected unknown fields: {unknown}")

    task_ids: set[str] = set()
    extraction_fields: list[str] = []
    for task in plan.tasks:
        if task.task_id in task_ids:
            raise ValueError(f"Duplicate planner task ID: {task.task_id}")
        if task.agent not in ALLOWED_AGENTS:
            raise ValueError(f"Planner selected disallowed agent: {task.agent}")
        if ACTION_TOOL_MAP[task.action] != task.tool:
            raise ValueError(
                f"Tool {task.tool!r} is not allowed for action {task.action!r}"
            )
        if not set(task.bid_ids).issubset(allowed_bid_ids):
            raise ValueError(f"Planner selected unknown bid IDs: {task.bid_ids}")
        if not set(task.field_names).issubset(official):
            raise ValueError(f"Planner selected unknown fields: {task.field_names}")
        unknown_dependencies = set(task.depends_on) - task_ids
        if unknown_dependencies:
            raise ValueError(
                f"Task {task.task_id} has unknown or forward dependencies: "
                f"{sorted(unknown_dependencies)}"
            )
        if task.action == "extract":
            extraction_fields.extend(task.field_names)
        task_ids.add(task.task_id)

    required_actions = {
        "extract": set(EXTRACTION_ACTIONS),
        "ask": {"ensure_index", "retrieve", "answer"},
        "compare": {"ensure_index", "retrieve", "answer"},
    }[plan.mode]
    missing = required_actions - {task.action for task in plan.tasks}
    if missing:
        raise ValueError(f"Planner omitted required actions: {sorted(missing)}")

    if plan.mode == "extract":
        if set(plan.selected_fields) != official:
            raise ValueError("Extract mode must cover all 20 official fields")
        required_extraction = official - {"Bid Summary"}
        if set(extraction_fields) != required_extraction:
            raise ValueError("Extraction tasks must cover all 19 non-summary fields")
        if len(extraction_fields) != len(required_extraction):
            raise ValueError("Each extraction field must be scheduled exactly once")
    if plan.mode in {"ask", "compare"} and not any(
        task.question for task in plan.tasks if task.action == "retrieve"
    ):
        raise ValueError("Question modes require a retrieval question")
    return plan


def expand_compact_extraction_plan(
    compact: CompactExtractionPlan,
    *,
    goal: str,
    bid_ids: tuple[str, ...],
) -> ExecutionPlan:
    """Expand approved groups into exactly 19 field tasks and one report task."""

    extraction_specs = [spec for spec in FIELD_SPECS if spec.field_name != "Bid Summary"]
    if len(extraction_specs) != 19:
        raise ValueError("Official registry must contain 19 extraction fields")
    selected_groups = set(compact.specialist_groups)
    if any(spec.specialist_group not in selected_groups for spec in extraction_specs):
        raise ValueError("Compact plan omitted a registered extraction specialist")

    tasks: list[ExecutionTask] = [
        ExecutionTask(
            task_id="ensure_index",
            action="ensure_index",
            agent="ingestion",
            tool="ingestion.ensure_index",
            bid_ids=bid_ids,
            completion_criteria="Every bid is ready for isolated retrieval",
        ),
        ExecutionTask(
            task_id="retrieve_evidence",
            action="retrieve",
            agent="retrieval_search",
            tool="retrieval.hybrid_cohere",
            bid_ids=bid_ids,
            field_names=tuple(spec.field_name for spec in extraction_specs),
            depends_on=("ensure_index",),
            completion_criteria="Each field receives ranked cited evidence",
        ),
    ]
    extraction_task_ids = []
    for index, spec in enumerate(extraction_specs, start=1):
        task_id = f"extract_field_{index:02d}"
        extraction_task_ids.append(task_id)
        tasks.append(
            ExecutionTask(
                task_id=task_id,
                action="extract",
                agent=spec.specialist_group,
                tool="gemini.extract",
                bid_ids=bid_ids,
                field_names=(spec.field_name,),
                depends_on=("retrieve_evidence",),
                completion_criteria=(
                    f"{spec.field_name} has a cited candidate or explicit not-found result"
                ),
            )
        )
    tasks.extend(
        [
            ExecutionTask(
                task_id="reconcile_addenda",
                action="reconcile",
                agent="addendum_reconciliation",
                tool="gemini.reconcile",
                bid_ids=bid_ids,
                field_names=tuple(spec.field_name for spec in extraction_specs),
                depends_on=tuple(extraction_task_ids),
                completion_criteria="Explicit amendments are applied with an audit log",
            ),
            ExecutionTask(
                task_id="validate_fields",
                action="validate",
                agent="validator_critic",
                tool="validator.check",
                bid_ids=bid_ids,
                field_names=tuple(spec.field_name for spec in extraction_specs),
                depends_on=("reconcile_addenda",),
                completion_criteria="Every reconciled field has a terminal validation status",
            ),
            ExecutionTask(
                task_id="report_bid",
                action="report",
                agent="qa_report",
                tool="report.finalize",
                bid_ids=bid_ids,
                field_names=("Bid Summary",),
                depends_on=("validate_fields",),
                completion_criteria="Exactly 20 official fields validate as final JSON",
            ),
        ]
    )
    plan = ExecutionPlan(
        mode="extract",
        goal=goal,
        tasks=tuple(tasks),
        selected_fields=OFFICIAL_FIELD_LABELS,
        stop_condition=compact.stop_condition,
        rationale=compact.rationale,
    )
    return validate_execution_plan(plan, allowed_bid_ids=set(bid_ids))


def _sum_optional(first: int | None, second: int | None) -> int | None:
    if first is None and second is None:
        return None
    return (first or 0) + (second or 0)


def _combine_usage(first: TokenUsage, second: TokenUsage) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=_sum_optional(first.prompt_tokens, second.prompt_tokens),
        output_tokens=_sum_optional(first.output_tokens, second.output_tokens),
        thought_tokens=_sum_optional(first.thought_tokens, second.thought_tokens),
        total_tokens=_sum_optional(first.total_tokens, second.total_tokens),
    )


def _generate_with_one_truncation_retry(
    *,
    client: GeminiClient,
    prompt: str,
    retry_prompt: str,
    response_model,
    initial_tokens: int,
    retry_tokens: int,
):
    try:
        result = client.generate_structured(
            prompt,
            response_model,
            system_instruction=ORCHESTRATOR_PLANNER_PROMPT,
            generation=GenerationSettings(temperature=0, max_output_tokens=initial_tokens),
        )
        usage = getattr(result, "usage", TokenUsage())
        return result, 0, usage, getattr(result, "latency_ms", 0)
    except GeminiGenerationError as first:
        if first.classification != "truncated_output":
            first.retry_count = 0
            raise
        try:
            result = client.generate_structured(
                retry_prompt,
                response_model,
                system_instruction=ORCHESTRATOR_PLANNER_PROMPT,
                generation=GenerationSettings(temperature=0, max_output_tokens=retry_tokens),
            )
        except GeminiGenerationError as second:
            second.retry_count = 1
            second.usage = _combine_usage(first.usage, second.usage)
            second.latency_ms += first.latency_ms
            raise
        return (
            result,
            1,
            _combine_usage(first.usage, result.usage),
            first.latency_ms + result.latency_ms,
        )


def _compact_prompt(request: dict[str, Any], *, retry: bool = False) -> str:
    prefix = "Return the minimal compact plan only." if retry else "Create a compact full-extraction plan."
    return f"""
{prefix}
Mode: extract
Goal: {request['goal']}
Workflow must be: full_bid_extraction
Specialists exactly: commercial_legal, dates_logistics, product_specifications
Actions exactly: {', '.join(EXTRACTION_ACTIONS)}
Tools exactly: {', '.join(ACTION_TOOL_MAP[action] for action in EXTRACTION_ACTIONS)}
State concise evidence requirements, completion criteria, and a bounded stop condition.
Do not list field names or create field-level tasks; Python expands the official registry.
""".strip()


def _question_prompt(request: dict[str, Any], *, retry: bool = False) -> str:
    instruction = "Use the fewest valid tasks." if retry else "Create a goal-specific executable plan."
    return f"""
{instruction}
REQUEST:
{json.dumps({
    'mode': request['mode'],
    'goal': request['goal'],
    'bid_ids': request['bid_ids'],
    'question': request.get('question'),
}, ensure_ascii=False)}
Allowed action-to-tool mapping: {json.dumps(ACTION_TOOL_MAP)}
Allowed agents: {sorted(ALLOWED_AGENTS)}
Use only the supplied bid IDs. Retrieve independently per bid. Do not plan full-field extraction.
Dependencies may reference only earlier task IDs. Include a bounded stop condition.
""".strip()


def create_execution_plan_with_trace(
    request: dict[str, Any],
    llm_client: GeminiClient | None = None,
) -> PlanningOutcome:
    mode = request["mode"]
    bid_ids = tuple(request["bid_ids"])
    started = datetime.now(timezone.utc)

    if mode == "extract":
        # Full extraction has no planning ambiguity: the assignment fixes the
        # field registry, specialist routing and execution stages.  Construct
        # the already-validated compact contract locally so provider outages
        # cannot prevent checkpoint recovery or summary publication.
        compact = CompactExtractionPlan(
            mode="extract",
            workflow="full_bid_extraction",
            specialist_groups=(
                "commercial_legal",
                "dates_logistics",
                "product_specifications",
            ),
            actions=EXTRACTION_ACTIONS,
            tools=tuple(ACTION_TOOL_MAP[action] for action in EXTRACTION_ACTIONS),
            evidence_requirements=(
                "Every found value must be supported by indexed file/page evidence.",
                "Addenda must be checked before a field is finalized.",
            ),
            completion_criteria=(
                "All 19 source fields have terminal evidence-based statuses.",
                "Bid Summary and the final 20-field output pass validation.",
            ),
            stop_condition=(
                "Stop after bounded retries; preserve provider and validation failures "
                "as resumable nonterminal states."
            ),
            rationale=(
                "The fixed official extraction contract is expanded deterministically "
                "from the canonical field registry."
            ),
        )
        plan = expand_compact_extraction_plan(
            compact,
            goal=request["goal"],
            bid_ids=bid_ids,
        )
        specialists = list(compact.specialist_groups)
        retries = 0
        usage = TokenUsage(total_tokens=0)
        latency_ms = 0
        result_model = "deterministic-registry-plan"
        finish_reason = "LOCAL_VALIDATION"
        tool_calls: list[str] = []
    elif mode in {"ask", "compare"}:
        client = llm_client or get_gemini_client()
        result, retries, usage, latency_ms = _generate_with_one_truncation_retry(
            client=client,
            prompt=_question_prompt(request),
            retry_prompt=_question_prompt(request, retry=True),
            response_model=ExecutionPlan,
            initial_tokens=900,
            retry_tokens=1_200,
        )
        compact = None
        plan = result.output
        if plan.mode != mode:
            raise ValueError(
                f"Planner mode {plan.mode!r} does not match request mode {mode!r}"
            )
        if plan.goal != request["goal"]:
            plan = plan.model_copy(update={"goal": request["goal"]})
        plan = _ensure_index_precondition(plan, bid_ids)
        plan = validate_execution_plan(plan, allowed_bid_ids=set(bid_ids))
        specialists = sorted({
            task.agent
            for task in plan.tasks
            if task.agent in {
                "dates_logistics",
                "commercial_legal",
                "product_specifications",
            }
        })
        model_name = getattr(client, "model", None) or configured_gemini_model()
        result_model = getattr(result, "model", None) or model_name
        finish_reason = getattr(result, "finish_reason", None)
        tool_calls = [f"google.genai.generate_content:{model_name}"]
    else:
        raise ValueError(f"Unsupported planner mode: {mode!r}")

    trace = AgentTrace(
        agent="orchestrator_planner",
        action="create_execution_plan",
        status="success",
        started_at=started,
        latency_ms=round(latency_ms, 2),
        tool_calls=tool_calls,
        input_summary={
            "mode": mode,
            "bid_count": len(bid_ids),
            "goal_characters": len(request["goal"]),
        },
        output_summary={
            "selected_specialist_groups": specialists,
            "planned_tasks": len(plan.tasks),
            "selected_fields": len(plan.selected_fields),
            "validation_status": "passed",
        },
        tokens_used=usage.total_tokens,
        metadata={
            "model": result_model,
            "finish_reason": finish_reason,
            "token_usage": usage.model_dump(),
            "retry_attempts": retries,
            "failure_classification": None,
            "structured_output_validation": "passed",
        },
    )
    return PlanningOutcome(plan=plan, compact_plan=compact, trace=trace)


def create_execution_plan(
    request: dict[str, Any],
    llm_client: GeminiClient | None = None,
) -> ExecutionPlan:
    """Backward-compatible plan-only facade."""

    return create_execution_plan_with_trace(request, llm_client=llm_client).plan
