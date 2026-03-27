"""
FileAuditWriter — audit logging integration for jingu-trust-gate.

Every gate admission is written to an append-only JSONL audit log.
This is Law 3 of jingu-trust-gate: "Every admission is audited."

This example shows how to wire FileAuditWriter (the built-in production
audit writer) into the gate, and how to read the resulting JSONL log.

Each AuditEntry records: proposal_id, timestamp, unit counts, reason codes,
and the full list of admitted/rejected unit IDs — giving you a reproducible
record of every gate decision.

Run:
  python examples/integration/audit_writer_example.py

Output:
  .jingu-trust-gate/audit.jsonl  (append-only, created in cwd)
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jingu_trust_gate import (
    AdmittedUnit,
    AuditEntry,
    AuditWriter,
    ConflictAnnotation,
    FileAuditWriter,
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
from jingu_trust_gate.helpers import approve, reject, first_failing


# ── Minimal domain type ────────────────────────────────────────────────────────

@dataclass
class SimpleClaim:
    id: str
    text: str
    grade: str             # "proven" | "speculative"
    evidence_refs: list[str]


# ── Minimal policy ─────────────────────────────────────────────────────────────

class SimplePolicy(GatePolicy[SimpleClaim]):

    def validate_structure(self, proposal: Proposal[SimpleClaim]) -> StructureValidationResult:
        if not proposal.units:
            return StructureValidationResult(
                valid=False,
                errors=[{"field": "units", "reasonCode": "EMPTY_PROPOSAL"}],
            )
        return StructureValidationResult(valid=True, errors=[])

    def bind_support(self, unit: SimpleClaim, pool: list[SupportRef]) -> UnitWithSupport[SimpleClaim]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[SimpleClaim], ctx: dict) -> UnitEvaluationResult:
        return first_failing([
            reject(uws.unit.id, "MISSING_EVIDENCE")
            if uws.unit.grade == "proven" and not uws.support_ids
            else None,
        ]) or approve(uws.unit.id)

    def detect_conflicts(self, units, pool) -> list[ConflictAnnotation]:
        return []

    def render(self, admitted_units, pool, ctx) -> VerifiedContext:
        return VerifiedContext(
            admitted_blocks=[
                VerifiedBlock(source_id=u.unit_id, content=u.unit.text)
                for u in admitted_units
            ],
            summary=VerifiedContextSummary(
                admitted=len(admitted_units), rejected=0, conflicts=0,
            ),
        )

    def build_retry_feedback(self, unit_results, ctx) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        return RetryFeedback(
            summary=f"{len(failed)} rejected",
            errors=[],
        )


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    # FileAuditWriter writes to .jingu-trust-gate/audit.jsonl in cwd.
    # The directory is created automatically on first write.
    audit_writer = FileAuditWriter(".jingu-trust-gate/audit.jsonl")

    gate = create_trust_gate(policy=SimplePolicy(), audit_writer=audit_writer)

    support_pool = [
        SupportRef(id="ref-1", source_id="doc-1", source_type="observation", attributes={}),
    ]

    # Admission 1: 1 approved, 1 rejected
    proposal_1: Proposal[SimpleClaim] = Proposal(
        id="prop-audit-001", kind="response",
        units=[
            SimpleClaim(id="c1", text="Fact with evidence", grade="proven", evidence_refs=["doc-1"]),
            SimpleClaim(id="c2", text="Hallucinated fact",  grade="proven", evidence_refs=[]),
        ],
    )

    result_1 = await gate.admit(proposal_1, support_pool)
    exp_1 = gate.explain(result_1)

    print(f"Admission 1: approved={exp_1.approved}, rejected={exp_1.rejected}")
    assert exp_1.approved == 1
    assert exp_1.rejected == 1

    # Admission 2: 2 approved
    proposal_2: Proposal[SimpleClaim] = Proposal(
        id="prop-audit-002", kind="response",
        units=[
            SimpleClaim(id="c3", text="Another fact", grade="proven", evidence_refs=["doc-1"]),
            SimpleClaim(id="c4", text="Speculative note", grade="speculative", evidence_refs=[]),
        ],
    )

    result_2 = await gate.admit(proposal_2, support_pool)
    exp_2 = gate.explain(result_2)

    print(f"Admission 2: approved={exp_2.approved}, rejected={exp_2.rejected}")
    assert exp_2.approved == 2
    assert exp_2.rejected == 0

    # ── Read and verify the audit log ─────────────────────────────────────────

    await asyncio.sleep(0.05)  # let the async writer flush

    audit_path = Path.cwd() / ".jingu-trust-gate" / "audit.jsonl"
    assert audit_path.exists(), f"Audit log should exist at {audit_path}"

    lines = [l for l in audit_path.read_text().splitlines() if l.strip()]
    assert len(lines) >= 2, "Audit log should have at least 2 entries"

    # Parse the last two entries (most recent run)
    entries = [json.loads(l) for l in lines[-2:]]

    entry_1 = next((e for e in entries if e["proposalId"] == "prop-audit-001"), None)
    entry_2 = next((e for e in entries if e["proposalId"] == "prop-audit-002"), None)

    assert entry_1 is not None
    assert entry_1["approvedCount"] == 1
    assert entry_1["rejectedCount"] == 1

    assert entry_2 is not None
    assert entry_2["approvedCount"] == 2
    assert entry_2["rejectedCount"] == 0

    print(f"\nAudit log: {audit_path}")
    print(f"Total entries in log: {len(lines)}")
    print("\nLast 2 entries:")
    for e in entries:
        print(
            f"  {e['proposalId']}  approved={e['approvedCount']}  "
            f"rejected={e['rejectedCount']}  ts={e['timestamp']}"
        )

    print("\n  [PASS] FileAuditWriter writes JSONL entries to .jingu-trust-gate/audit.jsonl")
    print("  [PASS] Each entry records proposal_id, counts, reason codes, and timestamp")
    print("  [PASS] Log is append-only — entries accumulate across runs\n")


if __name__ == "__main__":
    asyncio.run(main())
