"""
HPC GPU cluster — SRE incident investigation policy for jingu-trust-gate.

Use case: an agent collects kernel logs, DCGM metrics, k8s events, and
PyTorch logs from a failed training job, packages them as a SupportRef pool,
then asks an LLM to propose structured DiagnosticClaims.

Gate rules:
  R1/R2  grade=proven|derived + no bound evidence       → MISSING_EVIDENCE       → reject
  R3     permanence/replacement claims without confirmed-loss signal
                                                        → UNSUPPORTED_SEVERITY   → downgrade
  R4     cluster-wide scope claims with < 2 nodes in pool
                                                        → UNSUPPORTED_SCOPE      → downgrade
  R5     specific numeric value not matching evidence   → OVER_SPECIFIC_METRIC   → downgrade
  R6     everything else                                → approve

Conflict patterns:
  NODE_HEALTH_CONFLICT      blocking     — same node healthy and failed
  TEMPORAL_METRIC_CONFLICT  informational — same metric, two values in pool

Run:
  python examples/hpc_diagnostic_policy.py
"""

from __future__ import annotations

import asyncio
import re
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
class DiagnosticClaim:
    id: str
    claim: str
    grade: str
    evidence_refs: list[str]


# ── Policy ─────────────────────────────────────────────────────────────────────

class HpcDiagnosticPolicy(GatePolicy[DiagnosticClaim]):

    def validate_structure(self, proposal: Proposal[DiagnosticClaim]) -> StructureValidationResult:
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

    def bind_support(self, unit: DiagnosticClaim, pool: list[SupportRef]) -> UnitWithSupport[DiagnosticClaim]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[DiagnosticClaim], ctx: dict) -> UnitEvaluationResult:
        unit = uws.unit
        claim_lower = unit.claim.lower()

        # R1/R2: proven or derived with no evidence
        if not uws.support_ids and unit.grade != "suspected":
            return UnitEvaluationResult(unit_id=unit.id, decision="reject", reason_code="MISSING_EVIDENCE")

        if unit.grade in ("proven", "derived"):
            # R3: permanence/replacement needs confirmed-loss signal
            permanence_phrases = [
                "permanently damaged", "must be replaced", "needs replacement",
                "needs rma", "rma required", "hardware failure confirmed", "replace the gpu",
            ]
            if any(p in claim_lower for p in permanence_phrases):
                has_confirmed_loss = any(
                    (s.attributes.get("sourceType") == "nvml" and "lost" in (s.attributes.get("message") or "").lower()) or
                    (s.attributes.get("sourceType") == "dmesg" and "gpu lost" in (s.attributes.get("message") or "").lower())
                    for s in uws.support_refs
                )
                if not has_confirmed_loss:
                    return UnitEvaluationResult(
                        unit_id=unit.id, decision="downgrade", reason_code="UNSUPPORTED_SEVERITY",
                        new_grade="derived",
                        annotations={
                            "unsupportedAttributes": ["permanently damaged / must be replaced"],
                            "note": "ECC threshold breach or Xid requires hardware diagnostics to confirm permanent damage",
                        },
                    )

            # R4: cluster-wide scope needs ≥ 2 distinct nodes in pool
            scope_phrases = ["all nodes", "all other nodes", "entire cluster", "every node", "cluster-wide"]
            if any(p in claim_lower for p in scope_phrases):
                covered_nodes = {s.source_id for s in uws.support_refs}
                if len(covered_nodes) < 2:
                    return UnitEvaluationResult(
                        unit_id=unit.id, decision="downgrade", reason_code="UNSUPPORTED_SCOPE",
                        new_grade="derived",
                        annotations={
                            "unsupportedAttributes": ["all nodes / all other nodes / entire cluster"],
                            "coveredNodes": list(covered_nodes),
                            "note": "Scope claim exceeds evidence coverage in support pool",
                        },
                    )

            # R5: specific numeric value must match evidence
            m = re.search(r'\b(\d+)\s*(ecc errors?|errors?|gpus?|nodes?|ranks?)', claim_lower)
            if m:
                claimed_val = int(m.group(1))
                evidence_vals = [
                    s.attributes["value"] for s in uws.support_refs
                    if "value" in s.attributes
                ]
                if evidence_vals and claimed_val not in evidence_vals:
                    return UnitEvaluationResult(
                        unit_id=unit.id, decision="downgrade", reason_code="OVER_SPECIFIC_METRIC",
                        new_grade="derived",
                        annotations={"claimedValue": claimed_val, "evidenceValues": evidence_vals},
                    )

        return UnitEvaluationResult(unit_id=unit.id, decision="approve", reason_code="OK")

    def detect_conflicts(
        self, units: list[UnitWithSupport[DiagnosticClaim]], pool: list[SupportRef]
    ) -> list[ConflictAnnotation]:
        conflicts: list[ConflictAnnotation] = []

        # NODE_HEALTH_CONFLICT (blocking)
        healthy_by_node: dict[str, list[str]] = {}
        failed_by_node: dict[str, list[str]] = {}
        for uws in units:
            c = uws.unit.claim.lower()
            is_healthy = any(kw in c for kw in ("healthy", "no errors", "no issues"))
            is_failed = any(kw in c for kw in ("hardware failure", "fallen off the bus", "xid", "ecc error"))
            for ref in uws.support_refs:
                node = ref.source_id
                if is_healthy:
                    healthy_by_node.setdefault(node, []).append(uws.unit.id)
                if is_failed:
                    failed_by_node.setdefault(node, []).append(uws.unit.id)

        for node, healthy_ids in healthy_by_node.items():
            failed_ids = failed_by_node.get(node)
            if failed_ids:
                conflicts.append(ConflictAnnotation(
                    unit_ids=healthy_ids + failed_ids,
                    conflict_code="NODE_HEALTH_CONFLICT",
                    sources=[s.id for s in pool if s.source_id == node],
                    severity="blocking",
                    description=f"Conflicting health assertions for {node}: both healthy and failed claimed",
                ))

        # TEMPORAL_METRIC_CONFLICT (informational)
        metric_map: dict[str, dict] = {}
        for ref in pool:
            metric = ref.attributes.get("metric")
            value = ref.attributes.get("value")
            if metric is None or value is None:
                continue
            key = f"{ref.source_id}::{metric}"
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
            node_id, metric = key.split("::")
            conflicts.append(ConflictAnnotation(
                unit_ids=affected,
                conflict_code="TEMPORAL_METRIC_CONFLICT",
                sources=ref_ids,
                severity="informational",
                description=f"{metric} on {node_id} has conflicting values in pool: [{', '.join(str(v) for v in data['values'])}]",
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
                "Generate an incident report. "
                "For downgraded claims (grade=derived), hedge your language: say 'evidence suggests' not 'confirmed'. "
                "For conflicting claims, surface the conflict explicitly — do not silently pick one side. "
                "Do not assert permanent hardware damage unless grade=proven."
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
                    unit_id=r.unit_id,
                    reason_code=r.reason_code,
                    details={"hint": "Add at least one SupportRef source_id to evidence_refs, or lower grade to 'suspected'"},
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
    gate = create_trust_gate(policy=HpcDiagnosticPolicy(), audit_writer=NoopAuditWriter())

    support_pool = [
        SupportRef(id="ref-001", source_id="node-gpu-07", source_type="observation",
                   attributes={"sourceType": "kernel_log", "message": "Xid 79: GPU has fallen off the bus"}),
        SupportRef(id="ref-002", source_id="node-gpu-07", source_type="observation",
                   attributes={"sourceType": "dcgm_metric", "metric": "GPU_MEMORY_ECC_ERRORS", "value": 847, "threshold": 100}),
        SupportRef(id="ref-003", source_id="job-447", source_type="observation",
                   attributes={"sourceType": "pytorch_log", "message": "NCCL error: unhandled system error", "rank": 3}),
        SupportRef(id="ref-004", source_id="job-447", source_type="observation",
                   attributes={"sourceType": "pytorch_log", "message": "Watchdog caught collective operation timeout"}),
        SupportRef(id="ref-005", source_id="node-gpu-05", source_type="observation",
                   attributes={"sourceType": "dcgm_metric", "metric": "GPU_MEMORY_ECC_ERRORS", "value": 2, "threshold": 100}),
    ]

    proposal: Proposal[DiagnosticClaim] = Proposal(
        id="prop-hpc-001",
        kind="response",
        units=[
            DiagnosticClaim(id="u1", claim="node-gpu-07 experienced a hardware GPU failure (Xid 79)",
                            grade="proven", evidence_refs=["node-gpu-07"]),
            DiagnosticClaim(id="u2", claim="NCCL rank 3 dropped due to GPU failure on node-gpu-07, triggering collective timeout",
                            grade="derived", evidence_refs=["node-gpu-07", "job-447"]),
            DiagnosticClaim(id="u3", claim="node-gpu-07 GPU is permanently damaged and must be replaced",
                            grade="proven", evidence_refs=["node-gpu-07"]),
            DiagnosticClaim(id="u4", claim="All other nodes in the job are healthy",
                            grade="proven", evidence_refs=["node-gpu-05"]),
            DiagnosticClaim(id="u5", claim="The training job had been running stably for 2 hours before failure",
                            grade="proven", evidence_refs=[]),
        ],
    )

    result = await gate.admit(proposal, support_pool)
    context = gate.render(result)
    explanation = gate.explain(result)

    sep("HPC Diagnostic Policy — Admission Result")

    print("\n  Admitted units:")
    for u in result.admitted_units:
        label(f"  {u.unit_id} [{u.status}]", u.unit.claim[:60] + "...")
        if u.status == "downgraded":
            label("    unsupportedAttributes",
                  u.evaluation_results[0].annotations.get("unsupportedAttributes"))

    print("\n  Rejected units:")
    for u in result.rejected_units:
        label(f"  {u.unit_id} [rejected]", u.evaluation_results[0].reason_code)

    sep("Explanation")
    label("approved", explanation.approved)
    label("downgraded", explanation.downgraded)
    label("rejected", explanation.rejected)

    sep("VerifiedContext (input to incident report LLM)")
    for block in context.admitted_blocks:
        label(f"{block.source_id} [{block.grade}]", block.content[:55] + "...")
        if block.unsupported_attributes:
            label("  unsupported", block.unsupported_attributes)
    print(f"\n  instructions: {context.instructions}")


if __name__ == "__main__":
    asyncio.run(main())
