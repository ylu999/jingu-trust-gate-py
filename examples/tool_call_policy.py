"""
Tool call proposal policy for jingu-trust-gate.

Use case: an assistant with retrieval tools proposes tool calls.
jingu-trust-gate gates which calls are admitted before they are executed.

Gate rules:
  R1  justification is empty or too short (<20 chars)     → WEAK_JUSTIFICATION    → downgrade
  R2  intent not established (no user_query in pool)      → INTENT_NOT_ESTABLISHED → reject
  R3  same tool + same arguments already called            → REDUNDANT_CALL        → reject
  R4  expected_value is absent (None or empty string)     → MISSING_EXPECTED_VALUE → downgrade
  R5  everything else                                     → approve

Run:
  python examples/tool_call_policy.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from jingu_trust_gate import (
    AdmittedUnit,
    AuditEntry,
    AuditWriter,
    ConflictAnnotation,
    GatePolicy,
    Proposal,
    RenderContext,
    RetryContext,
    RetryError,
    RetryFeedback,
    SupportRef,
    StructureError,
    StructureValidationResult,
    UnitEvaluationResult,
    UnitWithSupport,
    VerifiedBlock,
    VerifiedContext,
    VerifiedContextSummary,
    create_trust_gate,
)


# ── Domain types ───────────────────────────────────────────────────────────────

@dataclass
class ToolCallProposal:
    id: str
    tool_name: str
    arguments: dict[str, Any]
    justification: str
    expected_value: Optional[str]    # what the agent expects to retrieve
    evidence_refs: list[str]         # source_ids of support refs that justify this call
    grade: str                       # "proven" | "derived" | "speculative"


# ── Policy ─────────────────────────────────────────────────────────────────────

class ToolCallPolicy(GatePolicy[ToolCallProposal]):

    def validate_structure(self, proposal: Proposal[ToolCallProposal]) -> StructureValidationResult:
        errors: list[StructureError] = []
        if not proposal.units:
            errors.append(StructureError(field="units", reason_code="EMPTY_PROPOSAL"))
            return StructureValidationResult(valid=False, errors=errors)
        for unit in proposal.units:
            if not unit.id.strip():
                errors.append(StructureError(field="id", reason_code="MISSING_UNIT_ID"))
            if not unit.tool_name.strip():
                errors.append(StructureError(field="tool_name", reason_code="MISSING_TOOL_NAME",
                                             message=f"unit {unit.id}"))
        return StructureValidationResult(valid=len(errors) == 0, errors=errors)

    def bind_support(self, unit: ToolCallProposal, pool: list[SupportRef]) -> UnitWithSupport[ToolCallProposal]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[ToolCallProposal], ctx: dict) -> UnitEvaluationResult:
        unit = uws.unit

        # R2: intent not established — check for user_query in the full pool via context
        # We detect this by checking if any support ref has source_type == "user_query"
        has_intent = any(s.source_type == "user_query" for s in uws.support_refs)
        if not has_intent:
            return UnitEvaluationResult(
                unit_id=unit.id, decision="reject", reason_code="INTENT_NOT_ESTABLISHED",
                annotations={
                    "note": "No user_query support ref bound to this tool call. "
                            "Tool calls must be justified by a stated user intent.",
                },
            )

        # R3: redundant call — tool_name + arguments key already called (tracked via support attrs)
        call_sig = f"{unit.tool_name}::{sorted(unit.arguments.items())}"
        for s in uws.support_refs:
            if s.attributes.get("call_signature") == call_sig:
                return UnitEvaluationResult(
                    unit_id=unit.id, decision="reject", reason_code="REDUNDANT_CALL",
                    annotations={
                        "note": f"Tool call {unit.tool_name}({unit.arguments}) was already executed.",
                        "priorCallRef": s.source_id,
                    },
                )

        # R1: weak justification
        if len(unit.justification.strip()) < 20:
            return UnitEvaluationResult(
                unit_id=unit.id, decision="downgrade", reason_code="WEAK_JUSTIFICATION",
                new_grade="speculative",
                annotations={
                    "note": f"Justification too vague ({len(unit.justification.strip())} chars). "
                            "Provide at least 20 characters.",
                },
            )

        # R4: missing expected value
        if not unit.expected_value or not unit.expected_value.strip():
            return UnitEvaluationResult(
                unit_id=unit.id, decision="downgrade", reason_code="MISSING_EXPECTED_VALUE",
                new_grade="derived",
                annotations={
                    "note": "No expected_value declared. Specify what you expect this tool call to return.",
                },
            )

        return UnitEvaluationResult(unit_id=unit.id, decision="approve", reason_code="OK")

    def detect_conflicts(
        self, units: list[UnitWithSupport[ToolCallProposal]], pool: list[SupportRef]
    ) -> list[ConflictAnnotation]:
        # No cross-unit conflict pattern for tool calls in this policy
        return []

    def render(
        self, admitted_units: list[AdmittedUnit], pool: list[SupportRef], ctx: RenderContext
    ) -> VerifiedContext:
        blocks = []
        for u in admitted_units:
            call = u.unit
            current_grade = u.applied_grades[-1] if u.applied_grades else call.grade
            blocks.append(VerifiedBlock(
                source_id=u.unit_id,
                content=f"{call.tool_name}({call.arguments})",
                grade=current_grade,
                unsupported_attributes=(
                    [u.evaluation_results[0].reason_code]
                    if u.status == "downgraded" else []
                ),
                conflict_note=None,
            ))
        return VerifiedContext(
            admitted_blocks=blocks,
            summary=VerifiedContextSummary(
                admitted=len(admitted_units), rejected=0,
                conflicts=sum(1 for u in admitted_units if u.status == "approved_with_conflict"),
            ),
            instructions=(
                "Execute only the admitted tool calls listed below. "
                "For calls with grade 'speculative' or 'derived', treat results as tentative and verify before using. "
                "Do not execute any tool call not listed here."
            ),
        )

    def build_retry_feedback(
        self, unit_results: list[UnitEvaluationResult], ctx: RetryContext
    ) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        hints = {
            "INTENT_NOT_ESTABLISHED": "Add a user_query support ref to evidence_refs to establish intent.",
            "REDUNDANT_CALL": "Remove this call; the same tool with the same arguments was already executed.",
        }
        return RetryFeedback(
            summary=f"{len(failed)} tool call(s) rejected on attempt {ctx.attempt}/{ctx.max_retries}.",
            errors=[
                RetryError(
                    unit_id=r.unit_id,
                    reason_code=r.reason_code,
                    details={"hint": hints.get(r.reason_code, "Review and resubmit.")},
                )
                for r in failed
            ],
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
    gate = create_trust_gate(policy=ToolCallPolicy(), audit_writer=NoopAuditWriter())

    # ── Scenario A: 3 tool calls, 1 rejected (INTENT_NOT_ESTABLISHED),
    #                              1 rejected (REDUNDANT_CALL), 1 approved ────
    sep("Scenario A — 3 tool calls: 1 rejected (INTENT_NOT_ESTABLISHED), "
        "1 rejected (REDUNDANT_CALL), 1 approved")

    call_sig_prior = "retrieval_search::[('query', 'climate GDP loss')]"

    pool_a = [
        # Establishes user intent for calls that reference it
        SupportRef(id="ref-a1", source_id="user_query_1", source_type="user_query",
                   attributes={"query": "What is the GDP loss from climate change by 2050?"}),
        # Prior call record for REDUNDANT_CALL detection
        SupportRef(id="ref-a2", source_id="prior_call_1", source_type="tool_call_result",
                   attributes={"call_signature": call_sig_prior,
                               "tool_name": "retrieval_search",
                               "result_summary": "Found 5 papers on GDP projections"}),
    ]

    proposal_a: Proposal[ToolCallProposal] = Proposal(
        id="prop-a", kind="plan",
        units=[
            # Call 1: APPROVED — intent established, good justification, expected_value set
            ToolCallProposal(
                id="call-1", tool_name="retrieval_search",
                arguments={"query": "economic cost of sea level rise 2050"},
                justification="User asked about GDP loss by 2050; searching for sea level rise costs fills a specific gap.",
                expected_value="Papers quantifying economic cost of sea level rise in dollar terms.",
                evidence_refs=["user_query_1"],
                grade="derived",
            ),
            # Call 2: REJECTED (INTENT_NOT_ESTABLISHED) — no user_query in evidence_refs
            ToolCallProposal(
                id="call-2", tool_name="retrieval_search",
                arguments={"query": "carbon tax policy effectiveness"},
                justification="This might provide useful background context for the analysis.",
                expected_value="Studies on carbon tax effectiveness.",
                evidence_refs=["prior_call_1"],  # prior_call_1 is tool_call_result, not user_query
                grade="speculative",
            ),
            # Call 3: REJECTED (REDUNDANT_CALL) — same tool + args as prior call
            ToolCallProposal(
                id="call-3", tool_name="retrieval_search",
                arguments={"query": "climate GDP loss"},
                justification="Searching for GDP loss data to answer the user query directly.",
                expected_value="GDP loss projections under different warming scenarios.",
                evidence_refs=["user_query_1", "prior_call_1"],
                grade="derived",
            ),
        ],
    )

    result_a = await gate.admit(proposal_a, pool_a)
    exp_a = gate.explain(result_a)

    print("\n  Gate results:")
    for u in result_a.admitted_units:
        grade_info = f", grade→{u.applied_grades[-1]}" if u.status == "downgraded" else ""
        label(f"  {u.unit_id} [{u.status}{grade_info}]", f"{u.unit.tool_name}({u.unit.arguments})")
    for u in result_a.rejected_units:
        label(f"  {u.unit_id} [rejected]", u.evaluation_results[0].reason_code)
        note = u.evaluation_results[0].annotations.get("note", "")
        if note:
            label("    note", note[:80])

    label("total", exp_a.total_units)
    label("approved", exp_a.approved)
    label("rejected", exp_a.rejected)

    assert exp_a.total_units == 3, f"expected 3 total, got {exp_a.total_units}"
    assert exp_a.approved == 1, f"expected 1 approved, got {exp_a.approved}"
    assert exp_a.rejected == 2, f"expected 2 rejected, got {exp_a.rejected}"

    rejected_ids = {u.unit_id for u in result_a.rejected_units}
    rejected_codes = {u.unit_id: u.evaluation_results[0].reason_code for u in result_a.rejected_units}

    assert "call-2" in rejected_ids, "call-2 should be rejected"
    assert rejected_codes["call-2"] == "INTENT_NOT_ESTABLISHED", \
        f"call-2 should be INTENT_NOT_ESTABLISHED, got {rejected_codes['call-2']}"
    assert "call-3" in rejected_ids, "call-3 should be rejected"
    assert rejected_codes["call-3"] == "REDUNDANT_CALL", \
        f"call-3 should be REDUNDANT_CALL, got {rejected_codes['call-3']}"

    print(f"  [PASS] call-1 approved")
    print(f"  [PASS] call-2 rejected with INTENT_NOT_ESTABLISHED")
    print(f"  [PASS] call-3 rejected with REDUNDANT_CALL")

    # ── Scenario B: downgrade cases ────────────────────────────────────────────
    sep("Scenario B — Downgrade cases: WEAK_JUSTIFICATION and MISSING_EXPECTED_VALUE")

    pool_b = [
        SupportRef(id="ref-b1", source_id="user_query_2", source_type="user_query",
                   attributes={"query": "List climate adaptation strategies."}),
    ]

    proposal_b: Proposal[ToolCallProposal] = Proposal(
        id="prop-b", kind="plan",
        units=[
            # Downgraded to speculative — justification too short
            ToolCallProposal(
                id="call-b1", tool_name="retrieval_search",
                arguments={"query": "adaptation strategies"},
                justification="relevant",  # < 20 chars
                expected_value="List of strategies.",
                evidence_refs=["user_query_2"],
                grade="derived",
            ),
            # Downgraded to derived — no expected_value
            ToolCallProposal(
                id="call-b2", tool_name="document_reader",
                arguments={"doc_id": "IPCC-SR15-Ch4"},
                justification="IPCC SR1.5 Chapter 4 covers adaptation strategies in detail.",
                expected_value=None,
                evidence_refs=["user_query_2"],
                grade="proven",
            ),
        ],
    )

    result_b = await gate.admit(proposal_b, pool_b)
    exp_b = gate.explain(result_b)

    print("\n  Gate results:")
    for u in result_b.admitted_units:
        label(f"  {u.unit_id} [{u.status}] grade={u.applied_grades[-1]}", u.unit.tool_name)
        label("    reason_code", u.evaluation_results[0].reason_code)

    assert exp_b.downgraded == 2, f"expected 2 downgraded, got {exp_b.downgraded}"
    assert exp_b.approved == 0, f"expected 0 plain approved, got {exp_b.approved}"

    b1 = next(u for u in result_b.admitted_units if u.unit_id == "call-b1")
    b2 = next(u for u in result_b.admitted_units if u.unit_id == "call-b2")
    assert b1.applied_grades[-1] == "speculative", f"call-b1 should be 'speculative', got {b1.applied_grades[-1]}"
    assert b2.applied_grades[-1] == "derived", f"call-b2 should be 'derived', got {b2.applied_grades[-1]}"

    print(f"  [PASS] call-b1 downgraded to 'speculative' (WEAK_JUSTIFICATION)")
    print(f"  [PASS] call-b2 downgraded to 'derived' (MISSING_EXPECTED_VALUE)")

    context_b = gate.render(result_b, pool_b)
    print(f"\n  Render output:")
    for blk in context_b.admitted_blocks:
        print(f"    [{blk.grade}] {blk.content}  flags={blk.unsupported_attributes}")
    print(f"\n  instructions: {context_b.instructions}")


if __name__ == "__main__":
    asyncio.run(main())
