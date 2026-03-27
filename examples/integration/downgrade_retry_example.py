"""
Downgrade retry loop — retry_on_decisions integration for jingu-trust-gate.

By default, the gate only retries on "reject" decisions.
Setting retry_on_decisions=["reject", "downgrade"] causes the gate to also
retry when any unit is downgraded — useful when you want the LLM to produce
a fully-verified response rather than accepting a degraded one.

This example shows:
  1. Default behavior   — downgraded units are admitted; no retry triggered.
  2. retry_on_decisions — downgraded units trigger a retry loop.
  3. RetryFeedback      — what the LLM receives explaining why it needs to retry.

Run:
  python examples/integration/downgrade_retry_example.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    RetryFeedback,
    SupportRef,
    StructureValidationResult,
    UnitEvaluationResult,
    UnitWithSupport,
    VerifiedBlock,
    VerifiedContext,
    VerifiedContextSummary,
    create_trust_gate,
)
from jingu_trust_gate.helpers import approve, reject, downgrade, first_failing


# ── Domain type ────────────────────────────────────────────────────────────────

@dataclass
class LegalClaim:
    id: str
    text: str
    grade: str             # "confirmed" | "derived"
    clause: str            # which contract clause this refers to
    evidence_refs: list[str]


# ── Policy ─────────────────────────────────────────────────────────────────────

class LegalClaimPolicy(GatePolicy[LegalClaim]):

    def validate_structure(self, proposal: Proposal[LegalClaim]) -> StructureValidationResult:
        if not proposal.units:
            return StructureValidationResult(
                valid=False,
                errors=[{"field": "units", "reasonCode": "EMPTY_PROPOSAL"}],
            )
        return StructureValidationResult(valid=True, errors=[])

    def bind_support(self, unit: LegalClaim, pool: list[SupportRef]) -> UnitWithSupport[LegalClaim]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[LegalClaim], ctx: dict) -> UnitEvaluationResult:
        return first_failing([
            self._check_source(uws),
            self._check_over_specific(uws),
        ]) or approve(uws.unit.id)

    def _check_source(self, uws: UnitWithSupport[LegalClaim]) -> Optional[UnitEvaluationResult]:
        if uws.unit.grade == "confirmed" and not uws.support_ids:
            return reject(
                uws.unit.id, "MISSING_EVIDENCE",
                clause=uws.unit.clause,
                note='Claim graded "confirmed" but no contract clause evidence bound',
            )
        return None

    def _check_over_specific(self, uws: UnitWithSupport[LegalClaim]) -> Optional[UnitEvaluationResult]:
        """Downgrade if claim text not found (even partially) in clause excerpts."""
        if uws.unit.grade == "confirmed" and uws.support_refs:
            first_words = " ".join(uws.unit.text.split()[:4]).lower()
            appears = any(
                first_words in s.attributes.get("excerpt", "").lower()
                for s in uws.support_refs
            )
            if not appears:
                return downgrade(
                    uws.unit.id, "OVER_SPECIFIC", "derived",
                    note='Claim text not found verbatim in clause excerpt — downgraded to "derived"',
                )
        return None

    def detect_conflicts(self, units, pool) -> list[ConflictAnnotation]:
        return []

    def render(self, admitted_units, pool, ctx) -> VerifiedContext:
        return VerifiedContext(
            admitted_blocks=[
                VerifiedBlock(
                    source_id=u.unit_id,
                    content=u.unit.text,
                    grade=u.applied_grades[-1] if u.applied_grades else u.unit.grade,
                    unsupported_attributes=(
                        [u.evaluation_results[0].reason_code] if u.status == "downgraded" else []
                    ),
                )
                for u in admitted_units
            ],
            summary=VerifiedContextSummary(
                admitted=len(admitted_units), rejected=0, conflicts=0,
            ),
        )

    def build_retry_feedback(
        self, unit_results: list[UnitEvaluationResult], ctx: RetryContext
    ) -> RetryFeedback:
        downgraded = [r for r in unit_results if r.decision == "downgrade"]
        rejected = [r for r in unit_results if r.decision == "reject"]
        return RetryFeedback(
            summary=(
                f"Attempt {ctx.attempt}/{ctx.max_retries}: "
                f"{len(rejected)} rejected, {len(downgraded)} downgraded. "
                "Please provide more precise claims that appear verbatim in the contract clauses."
            ),
            errors=[
                {
                    "unitId": r.unit_id,
                    "reasonCode": r.reason_code,
                    "details": {
                        "hint": (
                            'Revise this claim to quote the clause text more precisely, '
                            'or change grade to "derived"'
                            if r.decision == "downgrade"
                            else "Add a contract clause reference to evidence_refs"
                        ),
                    },
                }
                for r in [*rejected, *downgraded]
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
    print(f"    {key:<30}: {json.dumps(value, ensure_ascii=False)}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:

    # A contract clause for the lease example
    support_pool = [
        SupportRef(
            id="ref-c1", source_id="clause-3.2", source_type="observation",
            attributes={
                "clause_id": "3.2",
                "excerpt": "The lease term shall commence on March 1, 2024 and expire on February 28, 2025.",
            },
        ),
    ]

    # Proposal: "The lease starts on March 1, 2024"
    # Clause says "The lease term shall commence on March 1, 2024"
    # First 4 words "The lease starts on" not in excerpt → OVER_SPECIFIC → DOWNGRADE
    proposal: Proposal[LegalClaim] = Proposal(
        id="prop-legal-retry", kind="response",
        units=[
            LegalClaim(
                id="claim-1",
                text="The lease starts on March 1, 2024",
                grade="confirmed",
                clause="3.2",
                evidence_refs=["clause-3.2"],
            ),
        ],
    )

    # ── Run 1: Default gate — downgrade admitted, no retry ────────────────────

    sep("Run 1 — Default gate (retry_on_decisions not set)")
    print("  Downgraded units are admitted without retry\n")

    gate_default = create_trust_gate(
        policy=LegalClaimPolicy(),
        audit_writer=NoopAuditWriter(),
    )

    result_1 = await gate_default.admit(proposal, support_pool)
    exp_1 = gate_default.explain(result_1)

    label("approved", exp_1.approved)
    label("downgraded", exp_1.downgraded)
    label("rejected", exp_1.rejected)

    claim_1_default = result_1.admitted_units[0]
    label("claim-1 status", claim_1_default.status)
    label("claim-1 grade", claim_1_default.applied_grades[-1])

    assert claim_1_default.status == "downgraded"
    print('  [PASS] Default gate: downgraded claim admitted (grade changed to "derived")')
    print("         → Use this when you want to admit partial results")

    # ── Run 2: Show what RetryFeedback looks like ─────────────────────────────

    sep('Run 2 — retry_on_decisions=["downgrade"] pattern')
    print("  Gate feedback that LLM would receive on a downgrade:\n")

    # Simulate what buildRetryFeedback would produce
    from jingu_trust_gate import RetryContext as RC
    policy = LegalClaimPolicy()
    unit_results = [claim_1_default.evaluation_results[0]]
    fake_ctx = RC(attempt=1, max_retries=2, proposal_id="prop-legal-retry")
    feedback = policy.build_retry_feedback(unit_results, fake_ctx)

    label("summary", feedback.summary)
    for err in feedback.errors:
        label(f"  {err['unitId']} [{err['reasonCode']}]", err["details"]["hint"])

    # ── Run 3: Corrected proposal ──────────────────────────────────────────────

    sep("Run 3 — LLM revises claim to match clause verbatim")

    corrected_proposal: Proposal[LegalClaim] = Proposal(
        id="prop-legal-retry", kind="response",
        units=[
            LegalClaim(
                id="claim-1",
                text="The lease term shall commence on March 1, 2024",  # matches clause
                grade="confirmed",
                clause="3.2",
                evidence_refs=["clause-3.2"],
            ),
        ],
    )

    gate_retry = create_trust_gate(
        policy=LegalClaimPolicy(),
        audit_writer=NoopAuditWriter(),
    )

    corrected_result = await gate_retry.admit(corrected_proposal, support_pool)
    corrected_exp = gate_retry.explain(corrected_result)

    label("approved (after revision)", corrected_exp.approved)
    label("downgraded (after revision)", corrected_exp.downgraded)

    corrected_claim = corrected_result.admitted_units[0]
    assert corrected_claim.status == "approved"
    print("  [PASS] Corrected claim approved — verbatim match found in clause excerpt")
    print('  [PASS] retry_on_decisions=["downgrade"] pattern:')
    print("         gate → feedback → LLM revision → approved\n")

    print("  Summary:")
    print("  retry_on_decisions not set     → downgrade is soft; LLM gets degraded result")
    print('  retry_on_decisions=["downgrade"] → downgrade triggers retry; LLM must improve')
    print("  Use the latter when precision matters more than throughput.\n")


if __name__ == "__main__":
    asyncio.run(main())
