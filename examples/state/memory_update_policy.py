"""
Personal memory write gate — state mutation policy for jingu-trust-gate.

Use case: a personal assistant proposes updates to a user's memory store
(preferences, facts the user has stated, contact info, recurring tasks).
Before any write reaches system state, the gate verifies that every proposed
fact was actually stated by the user — not inferred, hallucinated, or carried
over from a different user's session.

This is the "state" gating pattern: the gate controls what is allowed to be
written into persistent state, not just what is included in an LLM response.

Gate rules:
  R1  no "user_statement" evidence at all          → SOURCE_UNVERIFIED   → reject
  R2  value was inferred, not stated directly       → INFERRED_NOT_STATED → downgrade to "inferred"
  R3  evidence belongs to a different userId        → SCOPE_VIOLATION     → reject
  R4  everything else                               → approve

Key idea:
  source_type = "user_statement" represents something the user explicitly said.
  An LLM may propose writes that "seem reasonable" but were never actually stated.
  The gate blocks those writes at the boundary — they never reach the memory store.

Run:
  python examples/state/memory_update_policy.py
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
from jingu_trust_gate.helpers import (
    approve,
    reject,
    downgrade,
    first_failing,
    empty_proposal_errors,
    missing_id_errors,
    missing_text_field_errors,
    has_support_type,
)


# ── Domain types ───────────────────────────────────────────────────────────────

@dataclass
class MemoryWrite:
    id: str
    user_id: str           # which user's memory store this targets
    key: str               # memory key, e.g. "preferred_language", "dietary_restriction"
    value: str             # proposed value to write
    grade: str             # "stated" | "inferred" | "system"
    justification: str     # why the agent proposes this write
    evidence_refs: list[str]  # source_ids of user_statement refs supporting this write


# ── Policy ─────────────────────────────────────────────────────────────────────

class MemoryUpdatePolicy(GatePolicy[MemoryWrite]):

    def validate_structure(self, proposal: Proposal[MemoryWrite]) -> StructureValidationResult:
        errors: list[StructureError] = []
        errors.extend(empty_proposal_errors(proposal))
        if errors:
            return StructureValidationResult(valid=False, errors=errors)
        errors.extend(missing_id_errors(proposal.units))
        errors.extend(missing_text_field_errors(proposal.units, "key", reason_code="MISSING_KEY"))
        errors.extend(missing_text_field_errors(proposal.units, "user_id", reason_code="MISSING_USER_ID"))
        return StructureValidationResult(valid=len(errors) == 0, errors=errors)

    def bind_support(self, unit: MemoryWrite, pool: list[SupportRef]) -> UnitWithSupport[MemoryWrite]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[MemoryWrite], ctx: dict) -> UnitEvaluationResult:
        return first_failing([
            self._check_source(uws),
            self._check_scope(uws),
            self._check_inferred(uws),
        ]) or approve(uws.unit.id)

    def _check_source(self, uws: UnitWithSupport[MemoryWrite]) -> Optional[UnitEvaluationResult]:
        """R1: at least one piece of evidence must be a direct user_statement."""
        has_statement = any(
            s.attributes.get("type") == "user_statement"
            for s in uws.support_refs
        )
        if not has_statement:
            return reject(
                uws.unit.id, "SOURCE_UNVERIFIED",
                key=uws.unit.key,
                note=(
                    f'No user_statement evidence for "{uws.unit.key}={uws.unit.value}". '
                    "Memory writes require the user to have explicitly stated the value."
                ),
            )
        return None

    def _check_scope(self, uws: UnitWithSupport[MemoryWrite]) -> Optional[UnitEvaluationResult]:
        """R3: evidence must belong to the same user as the write target."""
        for s in uws.support_refs:
            evidence_user = s.attributes.get("user_id")
            if evidence_user is not None and evidence_user != uws.unit.user_id:
                return reject(
                    uws.unit.id, "SCOPE_VIOLATION",
                    target_user_id=uws.unit.user_id,
                    evidence_user_id=evidence_user,
                    note=(
                        f'Evidence user_id "{evidence_user}" does not match '
                        f'write target user_id "{uws.unit.user_id}"'
                    ),
                )
        return None

    def _check_inferred(self, uws: UnitWithSupport[MemoryWrite]) -> Optional[UnitEvaluationResult]:
        """R2: grade=stated but value not verbatim in any user statement → downgrade."""
        if uws.unit.grade == "stated":
            stated_refs = [
                s for s in uws.support_refs
                if s.attributes.get("type") == "user_statement"
            ]
            value_appears = any(
                uws.unit.value.lower() in s.attributes.get("content", "").lower()
                for s in stated_refs
            )
            if not value_appears:
                return downgrade(
                    uws.unit.id, "INFERRED_NOT_STATED", "inferred",
                    key=uws.unit.key,
                    proposed_value=uws.unit.value,
                    note=(
                        f'Value "{uws.unit.value}" does not appear verbatim in user statements. '
                        'Downgraded to grade="inferred" — memory store should mark provenance accordingly.'
                    ),
                )
        return None

    def detect_conflicts(
        self, units: list[UnitWithSupport[MemoryWrite]], pool: list[SupportRef]
    ) -> list[ConflictAnnotation]:
        return []

    def render(
        self, admitted_units: list[AdmittedUnit], pool: list[SupportRef], ctx: RenderContext
    ) -> VerifiedContext:
        blocks = []
        for u in admitted_units:
            write = u.unit
            current_grade = u.applied_grades[-1] if u.applied_grades else write.grade
            blocks.append(VerifiedBlock(
                source_id=u.unit_id,
                content=f'SET {write.user_id}::{write.key} = "{write.value}"',
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
                admitted=len(admitted_units), rejected=0, conflicts=0,
            ),
            instructions=(
                'Apply only the verified memory writes below. '
                'Writes with grade="inferred" should be stored with a provenance flag '
                'indicating the value was derived, not directly stated by the user. '
                'Never write a rejected entry — it was not verified as user-stated.'
            ),
        )

    def build_retry_feedback(
        self, unit_results: list[UnitEvaluationResult], ctx: RetryContext
    ) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        return RetryFeedback(
            summary=(
                f"{len(failed)} memory write(s) rejected on attempt "
                f"{ctx.attempt}/{ctx.max_retries}. "
                "Each write must be traceable to a direct user statement."
            ),
            errors=[
                {
                    "unitId": r.unit_id,
                    "reasonCode": r.reason_code,
                    "details": {
                        "hint": (
                            "Add a user_statement SupportRef containing the user's direct quote"
                            if r.reason_code == "SOURCE_UNVERIFIED"
                            else "Ensure evidence user_id matches the write target user_id"
                            if r.reason_code == "SCOPE_VIOLATION"
                            else "Review gate policy requirements"
                        ),
                    },
                }
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


def subsep(title: str) -> None:
    print(f"\n  ── {title}")


def label(key: str, value: object) -> None:
    import json
    print(f"    {key:<32}: {json.dumps(value, ensure_ascii=False)}")


# ── Example run ────────────────────────────────────────────────────────────────

async def main() -> None:
    gate = create_trust_gate(policy=MemoryUpdatePolicy(), audit_writer=NoopAuditWriter())
    user_id = "user-42"

    # ── Scenario A: User says "I'm vegetarian" and "I prefer dark mode" ────────
    #
    # Agent proposes 3 writes:
    #   - dietary_restriction=vegetarian  (user stated it directly → approve)
    #   - ui_theme=dark                   (user stated "dark mode" → approve)
    #   - notification_pref=email         (never stated → SOURCE_UNVERIFIED → reject)

    sep("Scenario A — User: 'I'm vegetarian and I prefer dark mode'")

    pool_a = [
        SupportRef(
            id="ref-stmt-1", source_id="stmt-001", source_type="observation",
            attributes={
                "user_id": user_id,
                "type": "user_statement",
                "content": "I'm vegetarian, so please keep that in mind for meal suggestions.",
                "session_id": "session-abc",
            },
        ),
        SupportRef(
            id="ref-stmt-2", source_id="stmt-002", source_type="observation",
            attributes={
                "user_id": user_id,
                "type": "user_statement",
                "content": "By the way, I always use dark mode on all my apps.",
                "session_id": "session-abc",
            },
        ),
    ]

    print("\n  User statements:")
    for ref in pool_a:
        label(f"  {ref.source_id}", ref.attributes["content"])

    proposal_a: Proposal[MemoryWrite] = Proposal(
        id="prop-mem-001", kind="response",
        units=[
            # write-1: user stated "I'm vegetarian" → APPROVE
            MemoryWrite(
                id="write-1", user_id=user_id, key="dietary_restriction",
                value="vegetarian", grade="stated",
                justification="User explicitly stated they are vegetarian in this session",
                evidence_refs=["stmt-001"],
            ),
            # write-2: user stated "dark mode" → APPROVE
            MemoryWrite(
                id="write-2", user_id=user_id, key="ui_theme",
                value="dark", grade="stated",
                justification="User explicitly stated they prefer dark mode on all apps",
                evidence_refs=["stmt-002"],
            ),
            # write-3: never stated → REJECT (SOURCE_UNVERIFIED)
            MemoryWrite(
                id="write-3", user_id=user_id, key="notification_pref",
                value="email", grade="stated",
                justification="User seems to prefer email-based communication based on context",
                evidence_refs=[],
            ),
        ],
    )

    result_a = await gate.admit(proposal_a, pool_a)
    exp_a = gate.explain(result_a)

    subsep("OUTPUT: gate results")
    print("\n  Admitted writes:")
    for u in result_a.admitted_units:
        label(f"  {u.unit_id} [{u.status}]", f"{u.unit.key} = '{u.unit.value}'")
    print("\n  Rejected writes:")
    for u in result_a.rejected_units:
        label(f"  {u.unit_id} [{u.evaluation_results[0].reason_code}]", f"{u.unit.key} = '{u.unit.value}'")
        note = u.evaluation_results[0].annotations.get("note", "")
        if note:
            label("    note", note[:80])

    label("approved", exp_a.approved)
    label("downgraded", exp_a.downgraded)
    label("rejected", exp_a.rejected)

    assert exp_a.approved == 2, f"expected 2 approved, got {exp_a.approved}"
    assert exp_a.rejected == 1, f"expected 1 rejected, got {exp_a.rejected}"

    w3 = next(u for u in result_a.rejected_units if u.unit_id == "write-3")
    assert w3.evaluation_results[0].reason_code == "SOURCE_UNVERIFIED"

    print("  [PASS] write-1 (dietary_restriction=vegetarian) approved")
    print("  [PASS] write-2 (ui_theme=dark) approved")
    print("  [PASS] write-3 (notification_pref=email) rejected — SOURCE_UNVERIFIED")

    # ── Scenario B: Agent infers a value not literally stated ───────────────────
    #
    # User says "I work best in the morning" — agent proposes
    # preferred_work_hours=06:00-10:00. Value never stated verbatim → DOWNGRADE.

    sep("Scenario B — Agent infers work hours from 'I work best in the morning'")

    pool_b = [
        SupportRef(
            id="ref-stmt-3", source_id="stmt-003", source_type="observation",
            attributes={
                "user_id": user_id,
                "type": "user_statement",
                "content": "I work best in the morning when I have a clear head.",
                "session_id": "session-abc",
            },
        ),
    ]

    proposal_b: Proposal[MemoryWrite] = Proposal(
        id="prop-mem-002", kind="response",
        units=[
            MemoryWrite(
                id="write-4", user_id=user_id, key="preferred_work_hours",
                value="06:00-10:00",   # inferred, never said verbatim
                grade="stated",
                justification="User said they work best in the morning; 06:00-10:00 is a reasonable window",
                evidence_refs=["stmt-003"],
            ),
        ],
    )

    result_b = await gate.admit(proposal_b, pool_b)
    w4 = result_b.admitted_units[0]

    label("write-4 status", w4.status)
    label("write-4 applied grade", w4.applied_grades[-1])
    label("write-4 reason_code", w4.evaluation_results[0].reason_code)

    assert w4.status == "downgraded"
    assert w4.applied_grades[-1] == "inferred"
    assert w4.evaluation_results[0].reason_code == "INFERRED_NOT_STATED"
    print("  [PASS] write-4 downgraded to grade=inferred (INFERRED_NOT_STATED)")

    # ── Scenario C: Cross-user scope violation ───────────────────────────────────
    #
    # Evidence belongs to user-99 but write targets user-42 → SCOPE_VIOLATION.

    sep("Scenario C — Cross-user scope violation")

    pool_c = [
        SupportRef(
            id="ref-stmt-4", source_id="stmt-004", source_type="observation",
            attributes={
                "user_id": "user-99",   # different user!
                "type": "user_statement",
                "content": "I'm vegan actually, not vegetarian.",
                "session_id": "session-xyz",
            },
        ),
    ]

    proposal_c: Proposal[MemoryWrite] = Proposal(
        id="prop-mem-003", kind="response",
        units=[
            MemoryWrite(
                id="write-5", user_id="user-42", key="dietary_restriction",
                value="vegan", grade="stated",
                justification="User stated they are vegan",
                evidence_refs=["stmt-004"],
            ),
        ],
    )

    result_c = await gate.admit(proposal_c, pool_c)
    w5 = result_c.rejected_units[0]

    label("write-5 reason_code", w5.evaluation_results[0].reason_code)
    note = w5.evaluation_results[0].annotations.get("note", "")
    if note:
        label("note", note)

    assert w5.evaluation_results[0].reason_code == "SCOPE_VIOLATION"
    print("  [PASS] write-5 rejected (SCOPE_VIOLATION — user-99 evidence, user-42 target)")

    print("\n  Done.\n")


if __name__ == "__main__":
    asyncio.run(main())
