"""
E-commerce catalog chatbot policy for jingu-trust-gate.

Use case: customer asks about product features, stock, brand.
The RAG pipeline retrieves catalog and inventory records.
The LLM proposes claims. jingu-trust-gate gates what can be asserted.

Gate rules:
  R1  grade=proven + no evidence                        → MISSING_EVIDENCE    → reject
  R2  assertedFeature not in catalog features list      → UNSUPPORTED_FEATURE → downgrade
  R3  assertedBrand not in evidence                     → OVER_SPECIFIC_BRAND → downgrade
  R4  assertedStockCount not matching evidence          → OVER_SPECIFIC_STOCK → downgrade
  R5  everything else                                   → approve

Conflict patterns:
  STOCK_CONFLICT    blocking     — same SKU: one record in-stock, one out-of-stock
  FEATURE_CONFLICT  informational — same SKU: mutually exclusive features

Run:
  python examples/ecommerce_catalog_policy.py
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
class ProductClaim:
    id: str
    claim: str
    grade: str
    evidence_refs: list[str]
    asserted_feature: Optional[str] = None
    asserted_brand: Optional[str] = None
    asserted_stock_count: Optional[int] = None
    asserted_in_stock: Optional[bool] = None


# ── Policy ─────────────────────────────────────────────────────────────────────

class EcommerceCatalogPolicy(GatePolicy[ProductClaim]):

    def validate_structure(self, proposal: Proposal[ProductClaim]) -> StructureValidationResult:
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

    def bind_support(self, unit: ProductClaim, pool: list[SupportRef]) -> UnitWithSupport[ProductClaim]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[ProductClaim], ctx: dict) -> UnitEvaluationResult:
        unit = uws.unit

        # R1: proven with no evidence
        if unit.grade == "proven" and not uws.support_ids:
            return UnitEvaluationResult(unit_id=unit.id, decision="reject", reason_code="MISSING_EVIDENCE")

        # R2: asserted feature must be in catalog features
        if unit.asserted_feature:
            feature = unit.asserted_feature.lower()
            in_evidence = any(
                feature in [f.lower() for f in (s.attributes.get("features") or [])]
                for s in uws.support_refs
            )
            if not in_evidence:
                return UnitEvaluationResult(
                    unit_id=unit.id, decision="downgrade", reason_code="UNSUPPORTED_FEATURE",
                    new_grade="derived",
                    annotations={
                        "unsupportedAttributes": [unit.asserted_feature],
                        "note": f'Feature "{unit.asserted_feature}" not found in any bound catalog record',
                    },
                )

        # R3: asserted brand must match evidence
        if unit.asserted_brand:
            brand = unit.asserted_brand.lower()
            has_brand = any(
                (s.attributes.get("brand") or "").lower() == brand
                for s in uws.support_refs
            )
            if not has_brand:
                return UnitEvaluationResult(
                    unit_id=unit.id, decision="downgrade", reason_code="OVER_SPECIFIC_BRAND",
                    new_grade="derived",
                    annotations={
                        "unsupportedAttributes": [f"brand: {unit.asserted_brand}"],
                        "note": f'Brand "{unit.asserted_brand}" not found in bound catalog records',
                    },
                )

        # R4: exact stock count must match evidence
        if unit.asserted_stock_count is not None:
            claimed = unit.asserted_stock_count
            for ref in uws.support_refs:
                stock_range = ref.attributes.get("stockRange")
                stock_count = ref.attributes.get("stockCount")
                if stock_range:
                    if claimed < stock_range["min"] or claimed > stock_range["max"]:
                        return UnitEvaluationResult(
                            unit_id=unit.id, decision="downgrade", reason_code="OVER_SPECIFIC_STOCK",
                            new_grade="derived",
                            annotations={
                                "unsupportedAttributes": [f"exact stock count: {claimed}"],
                                "evidenceRange": stock_range,
                                "note": "Inventory record exposes a range, not an exact count",
                            },
                        )
                    else:
                        # within range but still over-specific
                        return UnitEvaluationResult(
                            unit_id=unit.id, decision="downgrade", reason_code="OVER_SPECIFIC_STOCK",
                            new_grade="derived",
                            annotations={
                                "unsupportedAttributes": [f"exact stock count: {claimed}"],
                                "evidenceRange": stock_range,
                                "note": "Inventory record exposes a range, not an exact count",
                            },
                        )
                if stock_count is not None and stock_count != claimed:
                    return UnitEvaluationResult(
                        unit_id=unit.id, decision="downgrade", reason_code="OVER_SPECIFIC_STOCK",
                        new_grade="derived",
                        annotations={
                            "unsupportedAttributes": [f"exact stock count: {claimed}"],
                            "evidenceCount": stock_count,
                        },
                    )

        return UnitEvaluationResult(unit_id=unit.id, decision="approve", reason_code="OK")

    def detect_conflicts(
        self, units: list[UnitWithSupport[ProductClaim]], pool: list[SupportRef]
    ) -> list[ConflictAnnotation]:
        conflicts: list[ConflictAnnotation] = []

        # STOCK_CONFLICT (blocking): same SKU, conflicting inStock values in pool
        sku_in: dict[str, list[str]] = {}
        sku_out: dict[str, list[str]] = {}
        for ref in pool:
            in_stock = ref.attributes.get("inStock")
            if in_stock is None:
                continue
            sku = ref.attributes.get("sku") or ref.source_id
            if in_stock:
                sku_in.setdefault(sku, []).append(ref.id)
            else:
                sku_out.setdefault(sku, []).append(ref.id)

        for sku, in_ids in sku_in.items():
            out_ids = sku_out.get(sku)
            if not out_ids:
                continue
            all_ref_ids = in_ids + out_ids
            affected = [uws.unit.id for uws in units if any(r in uws.support_ids for r in all_ref_ids)]
            if not affected:
                continue
            conflicts.append(ConflictAnnotation(
                unit_ids=affected,
                conflict_code="STOCK_CONFLICT",
                sources=all_ref_ids,
                severity="blocking",
                description=f"Conflicting stock status for SKU {sku}: pool contains both in-stock and out-of-stock records",
            ))

        # FEATURE_CONFLICT (informational): mutually exclusive features for same SKU
        mutually_exclusive = [
            ("active_noise_cancellation", "passive_noise_isolation"),
            ("wired", "wireless"),
        ]
        feature_map: dict[str, dict] = {}
        for ref in pool:
            for feat in (ref.attributes.get("features") or []):
                key = f"{ref.source_id}::{feat.split('_')[0]}"
                if key not in feature_map:
                    feature_map[key] = {"values": set(), "ref_ids": []}
                feature_map[key]["values"].add(feat)
                feature_map[key]["ref_ids"].append(ref.id)

        for key_a, key_b in mutually_exclusive:
            for key, data in feature_map.items():
                if key_a in data["values"] and key_b in data["values"]:
                    ref_ids = data["ref_ids"]
                    affected = [uws.unit.id for uws in units if any(r in uws.support_ids for r in ref_ids)]
                    if not affected:
                        continue
                    sku = key.split("::")[0]
                    conflicts.append(ConflictAnnotation(
                        unit_ids=affected,
                        conflict_code="FEATURE_CONFLICT",
                        sources=ref_ids,
                        severity="informational",
                        description=f'Conflicting feature data for SKU {sku}: "{key_a}" vs "{key_b}" both present',
                    ))

        return conflicts

    def render(
        self, admitted_units: list[AdmittedUnit], pool: list[SupportRef], ctx: RenderContext
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
                "Answer the customer's product question using only the verified facts below. "
                "For downgraded claims, hedge your language: say 'may support' or 'approximately' rather than stating facts with certainty. "
                "For conflicting claims, tell the customer the information is inconsistent and suggest they check the product page. "
                "Never invent feature names, stock numbers, or brand names not present in verified facts."
            ),
        )

    def build_retry_feedback(
        self, unit_results: list[UnitEvaluationResult], ctx: RetryContext
    ) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        return RetryFeedback(
            summary=f"{len(failed)} claim(s) rejected on attempt {ctx.attempt}/{ctx.max_retries}.",
            errors=[
                RetryError(
                    unit_id=r.unit_id, reason_code=r.reason_code,
                    details={"hint": "Add the SKU or record ID to evidence_refs, or lower grade to 'derived'"},
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
    gate = create_trust_gate(policy=EcommerceCatalogPolicy(), audit_writer=NoopAuditWriter())

    # ── Scenario A: feature query ──────────────────────────────────────────────
    sep("Scenario A — Feature query: noise cancellation")

    pool_a = [
        SupportRef(id="ref-a1", source_id="SKU-BH-4892", source_type="observation",
                   attributes={"sku": "SKU-BH-4892", "brand": "Anker",
                               "features": ["passive_noise_isolation", "bluetooth_5", "foldable"]}),
    ]
    proposal_a: Proposal[ProductClaim] = Proposal(
        id="prop-a", kind="response",
        units=[
            ProductClaim(id="u1", claim="This headphone is made by Anker",
                         grade="proven", evidence_refs=["SKU-BH-4892"], asserted_brand="Anker"),
            ProductClaim(id="u2", claim="This headphone supports active noise cancellation",
                         grade="proven", evidence_refs=["SKU-BH-4892"],
                         asserted_feature="active_noise_cancellation"),
            ProductClaim(id="u3", claim="This headphone supports passive noise isolation",
                         grade="proven", evidence_refs=["SKU-BH-4892"],
                         asserted_feature="passive_noise_isolation"),
            ProductClaim(id="u4", claim="This headphone retails for $79.99",
                         grade="proven", evidence_refs=[]),
        ],
    )
    result_a = await gate.admit(proposal_a, pool_a)
    exp_a = gate.explain(result_a)
    print("\n  Gate results:")
    for u in result_a.admitted_units:
        label(f"  {u.unit_id} [{u.status}]", u.unit.claim)
        if u.status == "downgraded":
            label("    unsupported", u.evaluation_results[0].annotations.get("unsupportedAttributes"))
    for u in result_a.rejected_units:
        label(f"  {u.unit_id} [rejected]", u.evaluation_results[0].reason_code)
    label("approved", exp_a.approved)
    label("downgraded", exp_a.downgraded)
    label("rejected", exp_a.rejected)

    # ── Scenario B: stock count query ──────────────────────────────────────────
    sep("Scenario B — Inventory query: exact stock count")

    pool_b = [
        SupportRef(id="ref-b1", source_id="SKU-BH-4892", source_type="observation",
                   attributes={"sku": "SKU-BH-4892", "inStock": True,
                               "stockRange": {"min": 10, "max": 50}}),
    ]
    proposal_b: Proposal[ProductClaim] = Proposal(
        id="prop-b", kind="response",
        units=[
            ProductClaim(id="u1", claim="This item is currently in stock",
                         grade="proven", evidence_refs=["SKU-BH-4892"], asserted_in_stock=True),
            ProductClaim(id="u2", claim="There are 99 units available",
                         grade="proven", evidence_refs=["SKU-BH-4892"], asserted_stock_count=99),
            ProductClaim(id="u3", claim="There are 30 units available",
                         grade="proven", evidence_refs=["SKU-BH-4892"], asserted_stock_count=30),
        ],
    )
    result_b = await gate.admit(proposal_b, pool_b)
    exp_b = gate.explain(result_b)
    print("\n  Gate results:")
    for u in result_b.admitted_units:
        label(f"  {u.unit_id} [{u.status}]", u.unit.claim)
        if u.status == "downgraded":
            ann = u.evaluation_results[0].annotations
            label("    unsupported", ann.get("unsupportedAttributes"))
            if "evidenceRange" in ann:
                label("    evidenceRange", ann["evidenceRange"])
    label("approved", exp_b.approved)
    label("downgraded", exp_b.downgraded)

    # ── Scenario C: blocking conflict ──────────────────────────────────────────
    sep("Scenario C — Blocking conflict: contradictory stock records")

    pool_c = [
        SupportRef(id="ref-c1", source_id="SKU-BH-4892", source_type="observation",
                   attributes={"sku": "SKU-BH-4892", "inStock": True}),
        SupportRef(id="ref-c2", source_id="SKU-BH-4892", source_type="observation",
                   attributes={"sku": "SKU-BH-4892", "inStock": False}),
    ]
    proposal_c: Proposal[ProductClaim] = Proposal(
        id="prop-c", kind="response",
        units=[
            ProductClaim(id="u1", claim="This item is available for purchase",
                         grade="proven", evidence_refs=["SKU-BH-4892"], asserted_in_stock=True),
            ProductClaim(id="u2", claim="This item is currently out of stock",
                         grade="proven", evidence_refs=["SKU-BH-4892"], asserted_in_stock=False),
        ],
    )
    result_c = await gate.admit(proposal_c, pool_c)
    context_c = gate.render(result_c)
    exp_c = gate.explain(result_c)

    print("\n  Gate results (blocking conflict — both units force-rejected):")
    for u in result_c.rejected_units:
        label(f"  {u.unit_id} [rejected]", u.unit.claim)
        label("    reasonCode", u.evaluation_results[0].reason_code)
    label("approved", exp_c.approved)
    label("rejected", exp_c.rejected)
    label("admittedBlocks", len(context_c.admitted_blocks))
    print(f"\n  instructions: {context_c.instructions}")
    print("\n  → LLM will tell the customer: stock status is inconsistent, check the product page.")


if __name__ == "__main__":
    asyncio.run(main())
