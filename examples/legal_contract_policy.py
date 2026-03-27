"""
Legal contract analysis — contract review assistant policy for jingu-trust-gate.

Use case: a lawyer or business user asks "Does this contract have a termination
clause?" or "What are the penalty terms?". The RAG pipeline retrieves relevant
contract clauses as evidence. The LLM proposes structured claims. jingu-trust-gate
admits only claims that match actual clause text.

Gate rules:
  R1  grade=proven + no bound evidence                          → MISSING_EVIDENCE     → reject
  R2  assertedTerm not present verbatim in evidence clause text → TERM_NOT_IN_EVIDENCE → reject
  R3  assertedFigure not found in evidence figures              → OVER_SPECIFIC_FIGURE → downgrade
  R4  assertedRight not in clause grants                        → SCOPE_EXCEEDED       → downgrade
  R5  everything else                                           → approve

Conflict patterns:
  CLAUSE_CONFLICT  blocking — irrevocable language vs termination rights

Run:
  python examples/legal_contract_policy.py
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
class AssertedFigure:
    type: str   # "percentage" | "days" | "amount"
    value: float


@dataclass
class ContractClaim:
    id: str
    claim: str
    grade: str
    evidence_refs: list[str]
    asserted_term: Optional[str] = None
    asserted_figure: Optional[AssertedFigure] = None
    asserted_right: Optional[str] = None


# ── Policy ─────────────────────────────────────────────────────────────────────

class LegalContractPolicy(GatePolicy[ContractClaim]):

    def validate_structure(self, proposal: Proposal[ContractClaim]) -> StructureValidationResult:
        errors: list[StructureError] = []
        if not proposal.units:
            errors.append(StructureError(field="units", reason_code="EMPTY_PROPOSAL"))
            return StructureValidationResult(valid=False, errors=errors)
        for unit in proposal.units:
            if not unit.id.strip():
                errors.append(StructureError(field="id", reason_code="MISSING_UNIT_ID"))
            if not unit.claim.strip():
                errors.append(StructureError(field="claim", reason_code="EMPTY_CLAIM", message=f"unit {unit.id}"))
            if not unit.grade:
                errors.append(StructureError(field="grade", reason_code="MISSING_GRADE", message=f"unit {unit.id}"))
        return StructureValidationResult(valid=len(errors) == 0, errors=errors)

    def bind_support(self, unit: ContractClaim, pool: list[SupportRef]) -> UnitWithSupport[ContractClaim]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[ContractClaim], ctx: dict) -> UnitEvaluationResult:
        unit = uws.unit

        # R1: proven with no evidence
        if unit.grade == "proven" and not uws.support_ids:
            return UnitEvaluationResult(unit_id=unit.id, decision="reject", reason_code="MISSING_EVIDENCE")

        # R2: asserted legal term must appear verbatim in clause text
        if unit.asserted_term:
            term = unit.asserted_term.lower()
            in_evidence = any(
                term in (s.attributes.get("clauseText") or "").lower() or
                any(term in t.lower() for t in (s.attributes.get("explicitTerms") or []))
                for s in uws.support_refs
            )
            if not in_evidence:
                return UnitEvaluationResult(
                    unit_id=unit.id, decision="reject", reason_code="TERM_NOT_IN_EVIDENCE",
                    annotations={
                        "assertedTerm": unit.asserted_term,
                        "note": f'Term "{unit.asserted_term}" does not appear in any bound clause',
                    },
                )

        # R3: specific figure must match evidence exactly
        if unit.asserted_figure:
            fig_type = unit.asserted_figure.type
            fig_val = unit.asserted_figure.value
            figure_in_evidence = any(
                any(f.get("type") == fig_type and f.get("value") == fig_val
                    for f in (s.attributes.get("figures") or []))
                for s in uws.support_refs
            )
            if not figure_in_evidence:
                return UnitEvaluationResult(
                    unit_id=unit.id, decision="downgrade", reason_code="OVER_SPECIFIC_FIGURE",
                    new_grade="derived",
                    annotations={
                        "unsupportedAttributes": [f"{fig_type}: {fig_val}"],
                        "note": f"Specific {fig_type} value {fig_val} not found in bound clauses",
                    },
                )

        # R4: asserted right must be explicitly granted by the clause
        if unit.asserted_right:
            right = unit.asserted_right.lower()
            right_granted = any(
                any(right in g.lower() for g in (s.attributes.get("grants") or []))
                for s in uws.support_refs
            )
            if not right_granted:
                return UnitEvaluationResult(
                    unit_id=unit.id, decision="downgrade", reason_code="SCOPE_EXCEEDED",
                    new_grade="derived",
                    annotations={
                        "unsupportedAttributes": [f"right: {unit.asserted_right}"],
                        "note": f'Claimed right "{unit.asserted_right}" is not explicitly granted in bound clauses',
                    },
                )

        return UnitEvaluationResult(unit_id=unit.id, decision="approve", reason_code="OK")

    def detect_conflicts(
        self, units: list[UnitWithSupport[ContractClaim]], pool: list[SupportRef]
    ) -> list[ConflictAnnotation]:
        irrevocable_refs = [
            s for s in pool
            if "irrevocable" in (s.attributes.get("clauseText") or "").lower() or
               any("irrevocable" in g.lower() for g in (s.attributes.get("grants") or []))
        ]
        revocable_refs = [
            s for s in pool
            if any(
                kw in g.lower()
                for g in (s.attributes.get("grants") or [])
                for kw in ("terminate", "cancel")
            )
        ]
        if not (irrevocable_refs and revocable_refs):
            return []

        all_ref_ids = [s.id for s in irrevocable_refs + revocable_refs]
        affected_ids = [
            uws.unit.id for uws in units
            if any(rid in uws.support_ids for rid in all_ref_ids)
        ]
        if not affected_ids:
            return []
        return [ConflictAnnotation(
            unit_ids=affected_ids,
            conflict_code="CLAUSE_CONFLICT",
            sources=all_ref_ids,
            severity="blocking",
            description="Contract contains both irrevocability language and termination rights — legal review required",
        )]

    def render(
        self,
        admitted_units: list[AdmittedUnit],
        pool: list[SupportRef],
        ctx: RenderContext,
    ) -> VerifiedContext:
        blocks = []
        for u in admitted_units:
            claim = u.unit
            current_grade = u.applied_grades[-1] if u.applied_grades else claim.grade
            conflict = u.conflict_annotations[0] if u.conflict_annotations else None
            blocks.append(VerifiedBlock(
                source_id=u.unit_id,
                content=claim.claim,
                grade=current_grade,
                unsupported_attributes=(
                    u.evaluation_results[0].annotations.get("unsupportedAttributes", [])
                    if u.status == "downgraded" else []
                ),
                conflict_note=(
                    f"{conflict.conflict_code}: {conflict.description or ''}"
                    if conflict else None
                ),
            ))
        return VerifiedContext(
            admitted_blocks=blocks,
            summary=VerifiedContextSummary(
                admitted=len(admitted_units), rejected=0,
                conflicts=sum(1 for u in admitted_units if u.status == "approved_with_conflict"),
            ),
            instructions=(
                "You are a contract analysis assistant. Use only the verified clause facts below. "
                "Do not invent legal terms, figures, or rights not present in the verified facts. "
                "For downgraded claims, use hedged language: 'the contract may include' rather than 'the contract includes'. "
                "If conflicting clauses are present, flag them explicitly and recommend legal review. "
                "This output is not legal advice."
            ),
        )

    def build_retry_feedback(
        self, unit_results: list[UnitEvaluationResult], ctx: RetryContext
    ) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        hints = {
            "TERM_NOT_IN_EVIDENCE": "The legal term you used does not appear in the cited clauses. Use the exact language from the clause text.",
            "MISSING_EVIDENCE": "Cite the clause ID in evidence_refs.",
        }
        return RetryFeedback(
            summary=f"{len(failed)} claim(s) rejected on attempt {ctx.attempt}/{ctx.max_retries}.",
            errors=[
                RetryError(
                    unit_id=r.unit_id,
                    reason_code=r.reason_code,
                    details={"hint": hints.get(r.reason_code, "Adjust the claim to match what the cited clause text explicitly states.")},
                )
                for r in failed
            ],
        )


# ── Noop audit writer ──────────────────────────────────────────────────────────

class NoopAuditWriter(AuditWriter):
    async def append(self, entry: AuditEntry) -> None:
        pass


# ── Example run ────────────────────────────────────────────────────────────────

def sep(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def label(key: str, value: object) -> None:
    import json
    print(f"    {key:<26}: {json.dumps(value, ensure_ascii=False)}")


async def main() -> None:
    gate = create_trust_gate(policy=LegalContractPolicy(), audit_writer=NoopAuditWriter())

    support_pool = [
        SupportRef(
            id="ref-001", source_id="clause-7b", source_type="observation",
            attributes={
                "clauseType": "cancellation",
                "clauseText": "Either party may cancel this agreement under the cancellation conditions set forth in Schedule A.",
                "explicitTerms": ["cancellation conditions", "schedule a"],
                "grants": ["either party may cancel"],
            },
        ),
        SupportRef(
            id="ref-002", source_id="clause-12a", source_type="observation",
            attributes={
                "clauseType": "penalty",
                "clauseText": "In the event of early cancellation, the cancelling party shall pay reasonable compensation to the other party.",
                "explicitTerms": ["early cancellation", "reasonable compensation"],
                "figures": [],
                "grants": [],
            },
        ),
    ]

    proposal: Proposal[ContractClaim] = Proposal(
        id="prop-legal-001",
        kind="response",
        units=[
            ContractClaim(id="u1", claim="The contract includes cancellation conditions",
                          grade="proven", evidence_refs=["clause-7b"], asserted_term="cancellation conditions"),
            ContractClaim(id="u2", claim="The contract includes a termination clause",
                          grade="proven", evidence_refs=["clause-7b"], asserted_term="termination clause"),
            ContractClaim(id="u3", claim="Early cancellation incurs a 20% penalty fee",
                          grade="proven", evidence_refs=["clause-12a"],
                          asserted_figure=AssertedFigure(type="percentage", value=20)),
            ContractClaim(id="u4", claim="Either party may terminate the contract with 30 days notice",
                          grade="proven", evidence_refs=["clause-7b"],
                          asserted_right="terminate with 30 days notice"),
            ContractClaim(id="u5", claim="Early cancellation requires payment of reasonable compensation",
                          grade="proven", evidence_refs=["clause-12a"]),
        ],
    )

    result = await gate.admit(proposal, support_pool)
    context = gate.render(result)
    explanation = gate.explain(result)

    sep("Legal Contract Policy — Admission Result")

    print("\n  Admitted units:")
    for u in result.admitted_units:
        label(f"  {u.unit_id} [{u.status}]", u.unit.claim)
        if u.status == "downgraded":
            label("    reasonCode", u.evaluation_results[0].reason_code)
            label("    unsupported", u.evaluation_results[0].annotations.get("unsupportedAttributes"))

    print("\n  Rejected units:")
    for u in result.rejected_units:
        label(f"  {u.unit_id} [rejected]", u.evaluation_results[0].reason_code)
        label("    claim", u.unit.claim)

    sep("Explanation")
    label("approved", explanation.approved)
    label("downgraded", explanation.downgraded)
    label("rejected", explanation.rejected)

    sep("Instructions injected into final LLM call")
    print(f"\n  {context.instructions}")


if __name__ == "__main__":
    asyncio.run(main())
