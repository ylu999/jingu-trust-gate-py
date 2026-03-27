"""
Research agent step proposal policy for jingu-trust-gate.

Use case: a research agent proposes the next steps it wants to take.
jingu-trust-gate gates which steps are admitted before the agent executes them.

Gate rules:
  R1  required_context not satisfied by any support ref   → MISSING_CONTEXT    → reject
  R2  grade=proven + no support at all                    → INSUFFICIENT_FINDINGS → reject
  R3  justification is empty or too short (<20 chars)     → WEAK_JUSTIFICATION → downgrade
  R4  everything else                                     → approve

Conflict patterns:
  REDUNDANT_STEP  informational — two steps with identical description (case-insensitive)

Run:
  python examples/agent_step_policy.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from jingu_trust_gate import (
    AdmittedUnit,
    AuditEntry,
    AuditWriter,
    ConflictAnnotation,
    GatePolicy,
    Proposal,
    RenderContext,
    RetryContext,
    SupportRef,
    StructureError,
    StructureValidationResult,
    UnitWithSupport,
    VerifiedBlock,
    VerifiedContext,
    VerifiedContextSummary,
    create_trust_gate,
)
from jingu_trust_gate.helpers import (
    approve,
    reject,
    downgrade,
    first_failing,
    empty_proposal_errors,
    missing_id_errors,
    missing_text_field_errors,
    hints_feedback,
)


# ── Domain types ───────────────────────────────────────────────────────────────

@dataclass
class AgentStepProposal:
    id: str
    description: str
    step_type: str          # e.g. "search", "read", "analyze", "summarize"
    justification: str
    required_context: list[str]   # context keys the step needs
    grade: str              # "proven" | "derived" | "speculative"


# ── Policy ─────────────────────────────────────────────────────────────────────

class AgentStepPolicy(GatePolicy[AgentStepProposal]):

    def validate_structure(self, proposal: Proposal[AgentStepProposal]) -> StructureValidationResult:
        errors: list[StructureError] = []
        errors.extend(empty_proposal_errors(proposal))
        if errors:
            return StructureValidationResult(valid=False, errors=errors)
        errors.extend(missing_id_errors(proposal.units))
        errors.extend(missing_text_field_errors(proposal.units, "description", reason_code="EMPTY_DESCRIPTION"))
        errors.extend(missing_text_field_errors(proposal.units, "step_type", reason_code="MISSING_STEP_TYPE"))
        return StructureValidationResult(valid=len(errors) == 0, errors=errors)

    def bind_support(self, unit: AgentStepProposal, pool: list[SupportRef]) -> UnitWithSupport[AgentStepProposal]:
        # A step's support refs are those whose source_id matches one of its required_context keys
        matched = [s for s in pool if s.source_id in unit.required_context]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[AgentStepProposal], ctx: dict):
        unit = uws.unit
        return first_failing([
            self._check_context(uws),
            self._check_findings(uws),
            self._check_justification(uws),
        ]) or approve(unit.id)

    def _check_context(self, uws: UnitWithSupport[AgentStepProposal]):
        unit = uws.unit
        if unit.required_context:
            satisfied = {s.source_id for s in uws.support_refs}
            missing = [k for k in unit.required_context if k not in satisfied]
            if missing:
                return reject(
                    unit.id, "MISSING_CONTEXT",
                    note=f"Required context not available: {', '.join(missing)}",
                    missingContext=missing,
                )
        return None

    def _check_findings(self, uws: UnitWithSupport[AgentStepProposal]):
        unit = uws.unit
        if unit.grade == "proven" and not uws.support_ids:
            return reject(
                unit.id, "INSUFFICIENT_FINDINGS",
                note="Step graded 'proven' but no supporting findings are bound",
            )
        return None

    def _check_justification(self, uws: UnitWithSupport[AgentStepProposal]):
        unit = uws.unit
        if len(unit.justification.strip()) < 20:
            return downgrade(
                unit.id, "WEAK_JUSTIFICATION", "speculative",
                note=f"Justification too vague ({len(unit.justification.strip())} chars). "
                     "Provide at least 20 characters explaining why this step is needed.",
            )
        return None

    def detect_conflicts(
        self, units: list[UnitWithSupport[AgentStepProposal]], pool: list[SupportRef]
    ) -> list[ConflictAnnotation]:
        conflicts: list[ConflictAnnotation] = []

        # REDUNDANT_STEP (informational): two steps with identical description
        seen: dict[str, str] = {}  # normalized_desc → first unit id
        for uws in units:
            key = uws.unit.description.strip().lower()
            if key in seen:
                conflicts.append(ConflictAnnotation(
                    unit_ids=[seen[key], uws.unit.id],
                    conflict_code="REDUNDANT_STEP",
                    sources=[],
                    severity="informational",
                    description=(
                        f'Steps "{seen[key]}" and "{uws.unit.id}" have identical descriptions. '
                        "Consider merging or removing the duplicate."
                    ),
                ))
            else:
                seen[key] = uws.unit.id

        return conflicts

    def render(
        self, admitted_units: list[AdmittedUnit], pool: list[SupportRef], ctx: RenderContext
    ) -> VerifiedContext:
        blocks = []
        for u in admitted_units:
            step = u.unit
            current_grade = u.applied_grades[-1] if u.applied_grades else step.grade
            conflict = u.conflict_annotations[0] if u.conflict_annotations else None
            blocks.append(VerifiedBlock(
                source_id=u.unit_id,
                content=f"[{step.step_type.upper()}] {step.description}",
                grade=current_grade,
                conflict_note=(
                    f"{conflict.conflict_code}: {conflict.description or ''}"
                    if conflict else None
                ),
                unsupported_attributes=(
                    u.evaluation_results[0].annotations.get("missingContext", [])
                    if u.status == "downgraded" else []
                ),
            ))
        return VerifiedContext(
            admitted_blocks=blocks,
            summary=VerifiedContextSummary(
                admitted=len(admitted_units), rejected=0,
                conflicts=sum(1 for u in admitted_units if u.status == "approved_with_conflict"),
            ),
            instructions=(
                "Execute only the verified steps listed below in order. "
                "For steps with grade 'speculative', proceed with extra caution and validate findings before continuing. "
                "Do not execute any steps that were rejected. "
                "For steps flagged as REDUNDANT_STEP, execute only one of the duplicates."
            ),
        )

    def build_retry_feedback(
        self, unit_results: list[UnitEvaluationResult], ctx: RetryContext
    ) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        return hints_feedback(
            unit_results,
            hints={
                "MISSING_CONTEXT": "Ensure required_context keys exist in the support pool, or remove them from required_context.",
                "INSUFFICIENT_FINDINGS": "Lower grade to 'derived' or add supporting research findings to the pool.",
            },
            summary=f"{len(failed)} step(s) rejected on attempt {ctx.attempt}/{ctx.max_retries}.",
        )


# ── Noop audit writer ──────────────────────────────────────────────────────────

class NoopAuditWriter(AuditWriter):
    async def append(self, entry: AuditEntry) -> None:
        pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def sep(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def label(key: str, value: object) -> None:
    import json
    print(f"    {key:<28}: {json.dumps(value, ensure_ascii=False)}")


# ── Example run ────────────────────────────────────────────────────────────────

async def main() -> None:
    gate = create_trust_gate(policy=AgentStepPolicy(), audit_writer=NoopAuditWriter())

    # Support pool: available research context
    pool = [
        SupportRef(id="ref-1", source_id="prior_search_results", source_type="observation",
                   attributes={"topic": "climate change", "results_count": 42}),
        SupportRef(id="ref-2", source_id="user_query", source_type="observation",
                   attributes={"query": "What are the economic impacts of climate change?"}),
    ]

    # ── Scenario A: 4 steps proposed, gate applies rules ──────────────────────
    sep("Scenario A — 4 steps proposed: 1 rejected (MISSING_CONTEXT), 1 rejected (INSUFFICIENT_FINDINGS), "
        "1 downgraded (WEAK_JUSTIFICATION), 1 approved")

    proposal_a: Proposal[AgentStepProposal] = Proposal(
        id="prop-a", kind="plan",
        units=[
            # Step 1: APPROVED — required_context satisfied, good justification
            AgentStepProposal(
                id="step-1", description="Search for peer-reviewed papers on economic impacts of climate change",
                step_type="search", grade="derived",
                justification="User query explicitly asks about economic impacts; prior search returned general results only.",
                required_context=["prior_search_results", "user_query"],
            ),
            # Step 2: REJECTED (MISSING_CONTEXT) — requires context not in pool
            AgentStepProposal(
                id="step-2", description="Read the IPCC AR6 economic chapter PDF",
                step_type="read", grade="derived",
                justification="IPCC AR6 is the authoritative source on climate economics.",
                required_context=["ipcc_ar6_document"],  # not in pool
            ),
            # Step 3: REJECTED (INSUFFICIENT_FINDINGS) — grade=proven but no support
            AgentStepProposal(
                id="step-3", description="Summarize findings on GDP loss projections",
                step_type="summarize", grade="proven",
                justification="Summarizing established GDP projections from retrieved documents.",
                required_context=[],  # no required context, but grade=proven with no pool match
            ),
            # Step 4: DOWNGRADED (WEAK_JUSTIFICATION) — justification too short
            AgentStepProposal(
                id="step-4", description="Analyze regional economic disparities",
                step_type="analyze", grade="derived",
                justification="good idea",  # < 20 chars
                required_context=["prior_search_results"],
            ),
        ],
    )

    result_a = await gate.admit(proposal_a, pool)
    exp_a = gate.explain(result_a)

    print("\n  Gate results:")
    for u in result_a.admitted_units:
        grade_info = f", grade→{u.applied_grades[-1]}" if u.status == "downgraded" else ""
        label(f"  {u.unit_id} [{u.status}{grade_info}]", u.unit.description[:55])
        if u.status == "downgraded":
            note = u.evaluation_results[0].annotations.get("note", "")
            label("    note", note)
    for u in result_a.rejected_units:
        label(f"  {u.unit_id} [rejected]", u.evaluation_results[0].reason_code)
        note = u.evaluation_results[0].annotations.get("note", "")
        if note:
            label("    note", note)

    label("total", exp_a.total_units)
    label("approved", exp_a.approved)
    label("downgraded", exp_a.downgraded)
    label("rejected", exp_a.rejected)

    # Assertions
    assert exp_a.total_units == 4,    f"expected 4 total, got {exp_a.total_units}"
    assert exp_a.approved == 1,       f"expected 1 approved, got {exp_a.approved}"
    assert exp_a.downgraded == 1,     f"expected 1 downgraded, got {exp_a.downgraded}"
    assert exp_a.rejected == 2,       f"expected 2 rejected, got {exp_a.rejected}"

    rejected_ids = {u.unit_id for u in result_a.rejected_units}
    assert "step-2" in rejected_ids,  "step-2 should be rejected (MISSING_CONTEXT)"
    assert "step-3" in rejected_ids,  "step-3 should be rejected (INSUFFICIENT_FINDINGS)"

    downgraded_ids = {u.unit_id for u in result_a.admitted_units if u.status == "downgraded"}
    assert "step-4" in downgraded_ids, "step-4 should be downgraded (WEAK_JUSTIFICATION)"

    downgraded_step4 = next(u for u in result_a.admitted_units if u.unit_id == "step-4")
    assert downgraded_step4.applied_grades[-1] == "speculative", \
        f"step-4 should be downgraded to 'speculative', got {downgraded_step4.applied_grades[-1]}"

    print(f"  [PASS] step-1 approved")
    print(f"  [PASS] step-2 rejected with MISSING_CONTEXT")
    print(f"  [PASS] step-3 rejected with INSUFFICIENT_FINDINGS")
    print(f"  [PASS] step-4 downgraded to 'speculative' with WEAK_JUSTIFICATION")

    # ── Scenario B: REDUNDANT_STEP conflict (informational) ───────────────────
    sep("Scenario B — REDUNDANT_STEP conflict: two identical descriptions")

    pool_b = [
        SupportRef(id="ref-b1", source_id="prior_search_results", source_type="observation",
                   attributes={"topic": "climate change"}),
    ]
    proposal_b: Proposal[AgentStepProposal] = Proposal(
        id="prop-b", kind="plan",
        units=[
            AgentStepProposal(
                id="step-a", description="Search economic impact data",
                step_type="search", grade="derived",
                justification="Initial broad search to gather relevant economic data sources.",
                required_context=["prior_search_results"],
            ),
            AgentStepProposal(
                id="step-b", description="Search economic impact data",  # identical
                step_type="search", grade="derived",
                justification="Searching for economic impact data to inform the analysis section.",
                required_context=["prior_search_results"],
            ),
        ],
    )

    result_b = await gate.admit(proposal_b, pool_b)
    exp_b = gate.explain(result_b)

    print("\n  Gate results (both steps admitted, conflict annotated):")
    for u in result_b.admitted_units:
        label(f"  {u.unit_id} [{u.status}]", u.unit.description)
        if u.conflict_annotations:
            label("    conflict", u.conflict_annotations[0].conflict_code)
            label("    severity", u.conflict_annotations[0].severity)

    assert exp_b.approved == 0, f"expected 0 plain approved (both have conflict), got {exp_b.approved}"
    assert exp_b.conflicts == 2, f"expected 2 with conflict, got {exp_b.conflicts}"
    assert result_b.has_conflicts is True, "has_conflicts should be True"

    print(f"  [PASS] both steps admitted with REDUNDANT_STEP (informational) conflict")

    # ── Scenario C: retry fixes rejected step ─────────────────────────────────
    sep("Scenario C — Retry: fixed step-2 now passes (context added to pool)")

    pool_c = pool + [
        SupportRef(id="ref-c1", source_id="ipcc_ar6_document", source_type="observation",
                   attributes={"title": "IPCC AR6 Working Group II Chapter 16"}),
    ]
    proposal_c: Proposal[AgentStepProposal] = Proposal(
        id="prop-c", kind="plan",
        units=[
            AgentStepProposal(
                id="step-2-fixed", description="Read the IPCC AR6 economic chapter PDF",
                step_type="read", grade="derived",
                justification="IPCC AR6 is the authoritative source; document is now available in context.",
                required_context=["ipcc_ar6_document"],
            ),
        ],
    )

    result_c = await gate.admit(proposal_c, pool_c)
    exp_c = gate.explain(result_c)

    assert exp_c.approved == 1, f"expected 1 approved after retry fix, got {exp_c.approved}"
    assert exp_c.rejected == 0, f"expected 0 rejected after retry fix, got {exp_c.rejected}"
    print(f"  [PASS] step-2-fixed approved after ipcc_ar6_document added to support pool")

    context_c = gate.render(result_c, pool_c)
    print(f"\n  instructions: {context_c.instructions}")
    print(f"\n  Admitted blocks:")
    for blk in context_c.admitted_blocks:
        print(f"    [{blk.grade}] {blk.content}")


if __name__ == "__main__":
    asyncio.run(main())
