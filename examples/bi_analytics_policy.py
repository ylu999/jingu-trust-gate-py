"""
BI analytics dashboard — natural-language query policy for jingu-trust-gate.

Use case: a business user asks "What was last quarter's revenue by region?"
The BI agent queries the data warehouse, gets metric rows as SupportRefs,
then proposes structured AnalyticsClaims. jingu-trust-gate admits only
claims whose numbers and dimensions are grounded in the fetched data.

Gate rules:
  R1  grade=proven + no bound evidence                        → MISSING_EVIDENCE    → reject
  R2  assertedValue not in any bound metric row               → VALUE_MISMATCH      → reject
  R3  assertedDimension not present in any bound row labels   → DIMENSION_MISMATCH  → downgrade
  R4  time_period of claim does not match evidence period     → PERIOD_MISMATCH     → downgrade
  R5  everything else                                         → approve

Conflict patterns:
  METRIC_CONFLICT      blocking     — same metric, two rows with different aggregations in pool
  STALE_DATA_CONFLICT  informational — evidence row marked stale=true

Run:
  python examples/bi_analytics_policy.py
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
class AnalyticsClaim:
    id: str
    claim: str
    grade: str
    evidence_refs: list[str]
    asserted_metric: Optional[str] = None     # e.g. "revenue"
    asserted_value: Optional[float] = None    # e.g. 4_200_000.0
    asserted_dimension: Optional[str] = None  # e.g. "APAC"
    asserted_period: Optional[str] = None     # e.g. "Q4-2024"


# ── Policy ─────────────────────────────────────────────────────────────────────

class BiAnalyticsPolicy(GatePolicy[AnalyticsClaim]):

    def validate_structure(self, proposal: Proposal[AnalyticsClaim]) -> StructureValidationResult:
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

    def bind_support(self, unit: AnalyticsClaim, pool: list[SupportRef]) -> UnitWithSupport[AnalyticsClaim]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[AnalyticsClaim], ctx: dict) -> UnitEvaluationResult:
        unit = uws.unit

        # R1: proven with no evidence
        if unit.grade == "proven" and not uws.support_ids:
            return UnitEvaluationResult(unit_id=unit.id, decision="reject", reason_code="MISSING_EVIDENCE")

        # R2: assertedValue must match a value in evidence
        if unit.asserted_value is not None:
            matching = [
                s for s in uws.support_refs
                if s.attributes.get("metric") == unit.asserted_metric
                and s.attributes.get("value") == unit.asserted_value
            ]
            if not matching:
                evidence_vals = [
                    s.attributes.get("value")
                    for s in uws.support_refs
                    if s.attributes.get("metric") == unit.asserted_metric
                ]
                return UnitEvaluationResult(
                    unit_id=unit.id,
                    decision="reject",
                    reason_code="VALUE_MISMATCH",
                    annotations={
                        "assertedValue": unit.asserted_value,
                        "evidenceValues": evidence_vals,
                        "metric": unit.asserted_metric,
                        "note": f"Claimed value {unit.asserted_value} not found in evidence for metric '{unit.asserted_metric}'",
                    },
                )

        # R3: assertedDimension must appear in at least one evidence row's labels
        if unit.asserted_dimension:
            dim_lower = unit.asserted_dimension.lower()
            dim_found = any(
                dim_lower in (s.attributes.get("dimension") or "").lower() or
                dim_lower in [d.lower() for d in (s.attributes.get("labels") or [])]
                for s in uws.support_refs
            )
            if not dim_found:
                available_dims = list({
                    s.attributes.get("dimension")
                    for s in uws.support_refs
                    if s.attributes.get("dimension")
                })
                return UnitEvaluationResult(
                    unit_id=unit.id,
                    decision="downgrade",
                    reason_code="DIMENSION_MISMATCH",
                    new_grade="derived",
                    annotations={
                        "unsupportedAttributes": [f"dimension: {unit.asserted_dimension}"],
                        "availableDimensions": available_dims,
                        "note": f"Dimension '{unit.asserted_dimension}' not found in bound evidence rows",
                    },
                )

        # R4: assertedPeriod must match evidence period
        if unit.asserted_period:
            period_found = any(
                unit.asserted_period.lower() == (s.attributes.get("period") or "").lower()
                for s in uws.support_refs
            )
            if not period_found:
                evidence_periods = list({
                    s.attributes.get("period")
                    for s in uws.support_refs
                    if s.attributes.get("period")
                })
                return UnitEvaluationResult(
                    unit_id=unit.id,
                    decision="downgrade",
                    reason_code="PERIOD_MISMATCH",
                    new_grade="derived",
                    annotations={
                        "unsupportedAttributes": [f"period: {unit.asserted_period}"],
                        "evidencePeriods": evidence_periods,
                        "note": f"Claimed period '{unit.asserted_period}' does not match evidence",
                    },
                )

        return UnitEvaluationResult(unit_id=unit.id, decision="approve", reason_code="OK")

    def detect_conflicts(
        self, units: list[UnitWithSupport[AnalyticsClaim]], pool: list[SupportRef]
    ) -> list[ConflictAnnotation]:
        conflicts: list[ConflictAnnotation] = []

        # METRIC_CONFLICT (blocking) — same metric+period+dimension, two different values
        metric_map: dict[str, dict] = {}
        for ref in pool:
            metric = ref.attributes.get("metric")
            period = ref.attributes.get("period", "")
            dimension = ref.attributes.get("dimension", "")
            value = ref.attributes.get("value")
            if metric is None or value is None:
                continue
            key = f"{metric}::{period}::{dimension}"
            if key not in metric_map:
                metric_map[key] = {"values": set(), "ref_ids": []}
            metric_map[key]["values"].add(value)
            metric_map[key]["ref_ids"].append(ref.id)

        for key, data in metric_map.items():
            if len(data["values"]) <= 1:
                continue
            ref_ids = data["ref_ids"]
            affected = [uws.unit.id for uws in units if any(r in uws.support_ids for r in ref_ids)]
            if not affected:
                continue
            metric, period, dimension = key.split("::")
            conflicts.append(ConflictAnnotation(
                unit_ids=affected,
                conflict_code="METRIC_CONFLICT",
                sources=ref_ids,
                severity="blocking",
                description=(
                    f"Conflicting values for {metric}"
                    + (f" in {period}" if period else "")
                    + (f" / {dimension}" if dimension else "")
                    + f": [{', '.join(str(v) for v in data['values'])}]"
                ),
            ))

        # STALE_DATA_CONFLICT (informational) — evidence row marked stale=true
        stale_refs = [s for s in pool if s.attributes.get("stale") is True]
        if stale_refs:
            stale_ids = [s.id for s in stale_refs]
            affected = [uws.unit.id for uws in units if any(r in uws.support_ids for r in stale_ids)]
            if affected:
                conflicts.append(ConflictAnnotation(
                    unit_ids=affected,
                    conflict_code="STALE_DATA_CONFLICT",
                    sources=stale_ids,
                    severity="informational",
                    description="One or more evidence rows are marked stale — data may not reflect current state",
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
                "You are a BI assistant answering business questions from verified analytics data. "
                "Use only the verified metric values below — do not invent numbers. "
                "For downgraded claims (grade=derived), use hedged language: 'data suggests' not 'the figure is'. "
                "If a metric conflict is present, surface it explicitly and do not pick one value arbitrarily. "
                "If data is marked stale, note that figures may be outdated."
            ),
        )

    def build_retry_feedback(
        self, unit_results: list[UnitEvaluationResult], ctx: RetryContext
    ) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        hints = {
            "MISSING_EVIDENCE": "Cite the metric row source_id in evidence_refs.",
            "VALUE_MISMATCH": "The numeric value you stated does not match the evidence. Use the exact figure from the cited metric row.",
        }
        return RetryFeedback(
            summary=f"{len(failed)} claim(s) rejected on attempt {ctx.attempt}/{ctx.max_retries}.",
            errors=[
                RetryError(
                    unit_id=r.unit_id,
                    reason_code=r.reason_code,
                    details={"hint": hints.get(r.reason_code, "Adjust the claim to match the bound evidence exactly.")},
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
    gate = create_trust_gate(policy=BiAnalyticsPolicy(), audit_writer=NoopAuditWriter())

    support_pool = [
        SupportRef(
            id="ref-001", source_id="dw-revenue-q4",
            source_type="observation",
            attributes={
                "metric": "revenue", "value": 12_400_000.0,
                "period": "Q4-2024", "dimension": "Global", "currency": "USD",
            },
        ),
        SupportRef(
            id="ref-002", source_id="dw-revenue-apac-q4",
            source_type="observation",
            attributes={
                "metric": "revenue", "value": 4_200_000.0,
                "period": "Q4-2024", "dimension": "APAC", "currency": "USD",
            },
        ),
        SupportRef(
            id="ref-003", source_id="dw-revenue-emea-q4",
            source_type="observation",
            attributes={
                "metric": "revenue", "value": 5_100_000.0,
                "period": "Q4-2024", "dimension": "EMEA", "currency": "USD",
            },
        ),
        SupportRef(
            id="ref-004", source_id="dw-revenue-amer-q4",
            source_type="observation",
            attributes={
                "metric": "revenue", "value": 3_100_000.0,
                "period": "Q4-2024", "dimension": "AMER", "currency": "USD",
            },
        ),
        SupportRef(
            id="ref-005", source_id="dw-revenue-apac-q4-old",
            source_type="observation",
            attributes={
                "metric": "revenue", "value": 3_900_000.0,
                "period": "Q4-2024", "dimension": "APAC", "stale": True,
            },
        ),
    ]

    proposal: Proposal[AnalyticsClaim] = Proposal(
        id="prop-bi-001",
        kind="response",
        units=[
            AnalyticsClaim(
                id="u1",
                claim="Q4-2024 global revenue was $12.4M",
                grade="proven",
                evidence_refs=["dw-revenue-q4"],
                asserted_metric="revenue",
                asserted_value=12_400_000.0,
                asserted_period="Q4-2024",
                asserted_dimension="Global",
            ),
            AnalyticsClaim(
                id="u2",
                claim="APAC contributed $4.2M in Q4-2024 revenue",
                grade="proven",
                evidence_refs=["dw-revenue-apac-q4"],
                asserted_metric="revenue",
                asserted_value=4_200_000.0,
                asserted_period="Q4-2024",
                asserted_dimension="APAC",
            ),
            AnalyticsClaim(
                id="u3",
                claim="EMEA revenue in Q4-2024 was $5.5M",
                grade="proven",
                evidence_refs=["dw-revenue-emea-q4"],
                asserted_metric="revenue",
                asserted_value=5_500_000.0,  # wrong — evidence says 5.1M
                asserted_period="Q4-2024",
                asserted_dimension="EMEA",
            ),
            AnalyticsClaim(
                id="u4",
                claim="AMER revenue grew compared to Q3-2024",
                grade="derived",
                evidence_refs=["dw-revenue-amer-q4"],
                asserted_period="Q3-2024",  # mismatch — evidence only has Q4
            ),
            AnalyticsClaim(
                id="u5",
                claim="LATAM revenue data is not available for Q4-2024",
                grade="suspected",
                evidence_refs=[],
            ),
        ],
    )

    result = await gate.admit(proposal, support_pool)
    context = gate.render(result)
    explanation = gate.explain(result)

    sep("BI Analytics Policy — Admission Result")

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
        if u.evaluation_results[0].annotations:
            ann = u.evaluation_results[0].annotations
            if "assertedValue" in ann:
                label("    assertedValue", ann["assertedValue"])
                label("    evidenceValues", ann.get("evidenceValues"))

    sep("Explanation")
    label("approved", explanation.approved)
    label("downgraded", explanation.downgraded)
    label("rejected", explanation.rejected)

    sep("Conflicts")
    all_conflicts = [
        ann
        for u in result.admitted_units
        for ann in u.conflict_annotations
    ]
    if all_conflicts:
        for c in all_conflicts:
            label(f"  {c.conflict_code} [{c.severity}]", c.description)
    else:
        print("    (none)")

    sep("VerifiedContext (input to BI report LLM)")
    for block in context.admitted_blocks:
        label(f"{block.source_id} [{block.grade}]", block.content)
        if block.unsupported_attributes:
            label("  unsupported", block.unsupported_attributes)
        if block.conflict_note:
            label("  conflict", block.conflict_note)
    print(f"\n  instructions: {context.instructions}")


if __name__ == "__main__":
    asyncio.run(main())
