"""
Tests for GateRunner and TrustGate — pytest test suite for GateRunner and TrustGate.
Run: pytest tests/
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import pytest

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
    StructureValidationResult,
    StructureError,
    UnitEvaluationResult,
    UnitWithSupport,
    VerifiedBlock,
    VerifiedContext,
    VerifiedContextSummary,
    create_trust_gate,
)


# ── Minimal domain type ───────────────────────────────────────────────────────

@dataclass
class SimpleClaim:
    id: str
    text: str
    grade: str
    evidence_refs: list[str] = field(default_factory=list)


# ── Minimal policy ────────────────────────────────────────────────────────────

class SimplePolicy(GatePolicy[SimpleClaim]):
    """Approves all units with bound evidence; rejects proven units with no evidence."""

    def __init__(self, conflicts: Optional[list[ConflictAnnotation]] = None) -> None:
        self._conflicts = conflicts or []

    def validate_structure(self, proposal: Proposal[SimpleClaim]) -> StructureValidationResult:
        if not proposal.units:
            return StructureValidationResult(valid=False, errors=[
                StructureError(field="units", reason_code="EMPTY_PROPOSAL")
            ])
        return StructureValidationResult(valid=True, errors=[])

    def bind_support(self, unit: SimpleClaim, pool: list[SupportRef]) -> UnitWithSupport[SimpleClaim]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[SimpleClaim], ctx: dict) -> UnitEvaluationResult:
        unit = uws.unit
        if unit.grade == "proven" and not uws.support_ids:
            return UnitEvaluationResult(unit_id=unit.id, decision="reject", reason_code="MISSING_EVIDENCE")
        return UnitEvaluationResult(unit_id=unit.id, decision="approve", reason_code="OK")

    def detect_conflicts(self, units: list[UnitWithSupport[SimpleClaim]], pool: list[SupportRef]) -> list[ConflictAnnotation]:
        return self._conflicts

    def render(self, admitted_units: list[AdmittedUnit], pool: list[SupportRef], ctx: RenderContext) -> VerifiedContext:
        return VerifiedContext(
            admitted_blocks=[VerifiedBlock(source_id=u.unit_id, content=u.unit.text) for u in admitted_units],
            summary=VerifiedContextSummary(admitted=len(admitted_units), rejected=0, conflicts=0),
        )

    def build_retry_feedback(self, unit_results: list[UnitEvaluationResult], ctx: RetryContext) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        return RetryFeedback(
            summary=f"{len(failed)} rejected",
            errors=[RetryError(unit_id=r.unit_id, reason_code=r.reason_code) for r in failed],
        )


class NoopAuditWriter(AuditWriter):
    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    async def append(self, entry: AuditEntry) -> None:
        self.entries.append(entry)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pool() -> list[SupportRef]:
    return [
        SupportRef(id="s1", source_id="src-a", source_type="doc"),
        SupportRef(id="s2", source_id="src-b", source_type="doc"),
    ]


def make_proposal(*units: SimpleClaim) -> Proposal[SimpleClaim]:
    return Proposal(id="prop-1", kind="response", units=list(units))


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_approve_unit_with_evidence():
    gate = create_trust_gate(policy=SimplePolicy(), audit_writer=NoopAuditWriter())
    proposal = make_proposal(SimpleClaim(id="u1", text="hello", grade="proven", evidence_refs=["src-a"]))
    result = await gate.admit(proposal, make_pool())
    assert len(result.admitted_units) == 1
    assert len(result.rejected_units) == 0
    assert result.admitted_units[0].status == "approved"


@pytest.mark.asyncio
async def test_reject_proven_unit_with_no_evidence():
    gate = create_trust_gate(policy=SimplePolicy(), audit_writer=NoopAuditWriter())
    proposal = make_proposal(SimpleClaim(id="u1", text="no refs", grade="proven", evidence_refs=[]))
    result = await gate.admit(proposal, make_pool())
    assert len(result.rejected_units) == 1
    assert result.rejected_units[0].evaluation_results[0].reason_code == "MISSING_EVIDENCE"


@pytest.mark.asyncio
async def test_structure_invalid_empty_proposal():
    gate = create_trust_gate(policy=SimplePolicy(), audit_writer=NoopAuditWriter())
    proposal = make_proposal()  # no units
    result = await gate.admit(proposal, [])
    assert len(result.rejected_units) == 0
    assert len(result.admitted_units) == 0
    assert not result.has_conflicts


@pytest.mark.asyncio
async def test_blocking_conflict_force_rejects_both_units():
    blocking = [ConflictAnnotation(
        unit_ids=["u1", "u2"],
        conflict_code="DATA_CONFLICT",
        sources=[],
        severity="blocking",
        description="mutually exclusive",
    )]
    gate = create_trust_gate(policy=SimplePolicy(conflicts=blocking), audit_writer=NoopAuditWriter())
    pool = make_pool()
    proposal = make_proposal(
        SimpleClaim(id="u1", text="claim A", grade="proven", evidence_refs=["src-a"]),
        SimpleClaim(id="u2", text="claim B", grade="proven", evidence_refs=["src-b"]),
    )
    result = await gate.admit(proposal, pool)
    assert len(result.admitted_units) == 0
    assert len(result.rejected_units) == 2
    reason_codes = {u.evaluation_results[0].reason_code for u in result.rejected_units}
    assert reason_codes == {"BLOCKING_CONFLICT"}


@pytest.mark.asyncio
async def test_informational_conflict_admits_with_conflict_status():
    info = [ConflictAnnotation(
        unit_ids=["u1", "u2"],
        conflict_code="SOFT_CONFLICT",
        sources=[],
        severity="informational",
    )]
    gate = create_trust_gate(policy=SimplePolicy(conflicts=info), audit_writer=NoopAuditWriter())
    pool = make_pool()
    proposal = make_proposal(
        SimpleClaim(id="u1", text="claim A", grade="proven", evidence_refs=["src-a"]),
        SimpleClaim(id="u2", text="claim B", grade="proven", evidence_refs=["src-b"]),
    )
    result = await gate.admit(proposal, pool)
    assert len(result.rejected_units) == 0
    assert all(u.status == "approved_with_conflict" for u in result.admitted_units)
    assert result.has_conflicts


@pytest.mark.asyncio
async def test_mixed_approve_reject():
    gate = create_trust_gate(policy=SimplePolicy(), audit_writer=NoopAuditWriter())
    pool = make_pool()
    proposal = make_proposal(
        SimpleClaim(id="u1", text="ok", grade="proven", evidence_refs=["src-a"]),
        SimpleClaim(id="u2", text="no evidence", grade="proven", evidence_refs=[]),
        SimpleClaim(id="u3", text="derived ok", grade="derived", evidence_refs=[]),
    )
    result = await gate.admit(proposal, pool)
    assert len(result.admitted_units) == 2
    assert len(result.rejected_units) == 1
    assert result.rejected_units[0].unit_id == "u2"


@pytest.mark.asyncio
async def test_audit_writer_called():
    writer = NoopAuditWriter()
    gate = create_trust_gate(policy=SimplePolicy(), audit_writer=writer)
    proposal = make_proposal(SimpleClaim(id="u1", text="x", grade="proven", evidence_refs=["src-a"]))
    await gate.admit(proposal, make_pool())
    assert len(writer.entries) == 1
    assert writer.entries[0].proposal_id == "prop-1"


@pytest.mark.asyncio
async def test_render_returns_verified_context():
    gate = create_trust_gate(policy=SimplePolicy(), audit_writer=NoopAuditWriter())
    pool = make_pool()
    proposal = make_proposal(SimpleClaim(id="u1", text="hello", grade="proven", evidence_refs=["src-a"]))
    result = await gate.admit(proposal, pool)
    ctx = gate.render(result, pool)
    assert len(ctx.admitted_blocks) == 1
    assert ctx.admitted_blocks[0].content == "hello"
    assert ctx.summary.rejected == 0


@pytest.mark.asyncio
async def test_explain_counts():
    gate = create_trust_gate(policy=SimplePolicy(), audit_writer=NoopAuditWriter())
    pool = make_pool()
    proposal = make_proposal(
        SimpleClaim(id="u1", text="ok", grade="proven", evidence_refs=["src-a"]),
        SimpleClaim(id="u2", text="no ev", grade="proven", evidence_refs=[]),
    )
    result = await gate.admit(proposal, pool)
    exp = gate.explain(result)
    assert exp.total_units == 2
    assert exp.approved == 1
    assert exp.rejected == 1
    assert "MISSING_EVIDENCE" in exp.gate_reason_codes
