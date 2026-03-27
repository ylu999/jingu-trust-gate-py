"""
Medical symptom assessment — health assistant policy for jingu-trust-gate.

Port of examples/medical-symptom-policy.ts.

Gate rules:
  R1  grade=proven + no bound evidence          → MISSING_EVIDENCE       → reject
  R2  isDiagnosis=True, no confirmed lab result → DIAGNOSIS_UNCONFIRMED  → reject
  R3  assertedCondition, evidence only suggests → OVER_CERTAIN           → downgrade
  R4  isTreatment=True                          → TREATMENT_NOT_ADVISED  → reject
  R5  everything else                           → approve

Run:
  python examples/medical_symptom_policy.py
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
    RetryFeedback,
    RetryError,
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
class SymptomClaim:
    id: str
    claim: str
    grade: str  # "proven" | "derived" | "suspected"
    evidence_refs: list[str]
    asserted_condition: Optional[str] = None
    is_diagnosis: bool = False
    is_treatment: bool = False


# ── Policy ─────────────────────────────────────────────────────────────────────

class MedicalSymptomPolicy(GatePolicy[SymptomClaim]):

    def validate_structure(self, proposal: Proposal[SymptomClaim]) -> StructureValidationResult:
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

    def bind_support(self, unit: SymptomClaim, pool: list[SupportRef]) -> UnitWithSupport[SymptomClaim]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(
        self, uws: UnitWithSupport[SymptomClaim], ctx: dict
    ) -> UnitEvaluationResult:
        unit = uws.unit

        # R1: proven with no evidence
        if unit.grade == "proven" and not uws.support_ids:
            return UnitEvaluationResult(unit_id=unit.id, decision="reject", reason_code="MISSING_EVIDENCE")

        # R4: treatment claims always rejected
        if unit.is_treatment:
            return UnitEvaluationResult(
                unit_id=unit.id, decision="reject", reason_code="TREATMENT_NOT_ADVISED",
                annotations={"note": "Treatment recommendations require clinical evaluation"},
            )

        # R2: diagnosis requires confirmed lab result
        if unit.is_diagnosis:
            has_confirmed = any(s.attributes.get("confirmed") is True for s in uws.support_refs)
            if not has_confirmed:
                return UnitEvaluationResult(
                    unit_id=unit.id, decision="reject", reason_code="DIAGNOSIS_UNCONFIRMED",
                    annotations={"note": "A confirmed diagnosis requires lab results or clinical confirmation"},
                )

        # R3: over-certain condition assertion
        if unit.asserted_condition and not unit.is_diagnosis:
            cond = unit.asserted_condition.lower()
            directly = any(
                s.attributes.get("confirmed") is True and
                cond in [c.lower() for c in s.attributes.get("suggestsConditions", [])]
                for s in uws.support_refs
            )
            weakly = any(
                cond in [c.lower() for c in s.attributes.get("suggestsConditions", [])]
                for s in uws.support_refs
            )
            if not directly and weakly and unit.grade == "proven":
                return UnitEvaluationResult(
                    unit_id=unit.id, decision="downgrade", reason_code="OVER_CERTAIN",
                    new_grade="suspected",
                    annotations={"unsupportedAttributes": [f"confirmed condition: {unit.asserted_condition}"],
                                 "note": "Evidence suggests this condition but does not confirm it"},
                )

        # R5: approved
        return UnitEvaluationResult(unit_id=unit.id, decision="approve", reason_code="OK")

    def detect_conflicts(
        self, units: list[UnitWithSupport[SymptomClaim]], pool: list[SupportRef]
    ) -> list[ConflictAnnotation]:
        mutually_exclusive = [
            ("type 1 diabetes", "type 2 diabetes"),
            ("hypothyroidism", "hyperthyroidism"),
            ("viral infection", "bacterial infection"),
        ]
        condition_map: dict[str, str] = {}
        for uws in units:
            if uws.unit.asserted_condition:
                condition_map[uws.unit.id] = uws.unit.asserted_condition.lower()

        conflicts: list[ConflictAnnotation] = []
        for cond_a, cond_b in mutually_exclusive:
            ids_a = [uid for uid, c in condition_map.items() if cond_a in c]
            ids_b = [uid for uid, c in condition_map.items() if cond_b in c]
            if ids_a and ids_b:
                conflicts.append(ConflictAnnotation(
                    unit_ids=ids_a + ids_b,
                    conflict_code="CONDITION_CONFLICT",
                    sources=[],
                    severity="informational",
                    description=f'Mutually exclusive conditions both suggested: "{cond_a}" vs "{cond_b}"',
                ))
        return conflicts

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
            block = VerifiedBlock(
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
            )
            blocks.append(block)
        return VerifiedContext(
            admitted_blocks=blocks,
            summary=VerifiedContextSummary(
                admitted=len(admitted_units), rejected=0, conflicts=sum(1 for u in admitted_units if u.status == "approved_with_conflict")
            ),
            instructions=(
                "You are a health information assistant, not a doctor. "
                "Use only the verified facts below. "
                "For suspected conditions, use language like 'your symptoms may be consistent with' — never assert a diagnosis. "
                "Never recommend treatments or medications. "
                "Always end with: 'Please consult a qualified healthcare professional for a proper evaluation.'"
            ),
        )

    def build_retry_feedback(
        self, unit_results: list[UnitEvaluationResult], ctx: RetryContext
    ) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        hints = {
            "DIAGNOSIS_UNCONFIRMED": "Remove is_diagnosis=True or supply a lab-confirmed evidence ref.",
            "TREATMENT_NOT_ADVISED": "Remove treatment/medication recommendations entirely.",
        }
        return RetryFeedback(
            summary=f"{len(failed)} claim(s) rejected on attempt {ctx.attempt}/{ctx.max_retries}.",
            errors=[
                RetryError(
                    unit_id=r.unit_id,
                    reason_code=r.reason_code,
                    details={"hint": hints.get(r.reason_code, "Add evidence refs or lower the grade.")},
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
    gate = create_trust_gate(policy=MedicalSymptomPolicy(), audit_writer=NoopAuditWriter())

    support_pool = [
        SupportRef(id="ref-001", source_id="symptom-fatigue", source_type="observation",
                   attributes={"recordType": "symptom", "symptom": "fatigue", "severity": "moderate",
                               "suggestsConditions": ["diabetes", "hypothyroidism", "anemia"]}),
        SupportRef(id="ref-002", source_id="symptom-thirst", source_type="observation",
                   attributes={"recordType": "symptom", "symptom": "excessive thirst", "severity": "moderate",
                               "suggestsConditions": ["diabetes", "dehydration"]}),
        SupportRef(id="ref-003", source_id="kb-diabetes", source_type="observation",
                   attributes={"recordType": "knowledge_base", "suggestsConditions": ["diabetes"], "confirmed": False}),
    ]

    proposal: Proposal[SymptomClaim] = Proposal(
        id="prop-med-001",
        kind="response",
        units=[
            SymptomClaim(id="u1", claim="The patient reports fatigue and excessive thirst",
                         grade="proven", evidence_refs=["symptom-fatigue", "symptom-thirst"]),
            SymptomClaim(id="u2", claim="These symptoms may be consistent with diabetes",
                         grade="proven", evidence_refs=["symptom-fatigue", "symptom-thirst", "kb-diabetes"],
                         asserted_condition="diabetes"),
            SymptomClaim(id="u3", claim="The patient has diabetes",
                         grade="proven", evidence_refs=["symptom-fatigue", "symptom-thirst"],
                         asserted_condition="diabetes", is_diagnosis=True),
            SymptomClaim(id="u4", claim="The patient should start metformin",
                         grade="derived", evidence_refs=["symptom-fatigue"], is_treatment=True),
            SymptomClaim(id="u5", claim="The patient has had these symptoms for 3 months",
                         grade="proven", evidence_refs=[]),
        ],
    )

    result = await gate.admit(proposal, support_pool)
    context = gate.render(result)
    explanation = gate.explain(result)

    sep("Medical Symptom Policy — Admission Result")

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
