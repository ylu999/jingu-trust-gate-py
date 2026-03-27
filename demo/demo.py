"""
jingu-trust-gate Python narrative demo

This demo IS the documentation. Read the terminal output and you will
understand the entire system — no external docs needed.

Domain: household memory assistant (LLM that recalls what is in your home)

Run: python demo/demo.py  (from repo root)
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, "src")

from jingu_trust_gate import (
    GatePolicy,
    Proposal,
    SupportRef,
    UnitWithSupport,
    StructureValidationResult,
    StructureError,
    UnitEvaluationResult,
    ConflictAnnotation,
    AdmittedUnit,
    VerifiedContext,
    VerifiedContextSummary,
    VerifiedBlock,
    RenderContext,
    RetryFeedback,
    RetryError,
    RetryContext,
    RetryConfig,
    AuditWriter,
    AuditEntry,
    create_trust_gate,
)
from jingu_trust_gate.helpers import approve, reject, downgrade

# ---------------------------------------------------------------------------
# Import adapters from examples/
# ---------------------------------------------------------------------------

sys.path.insert(0, ".")
from examples.integration.adapter_examples import (
    ClaudeContextAdapter,
    ClaudeAdapterOptions,
    OpenAIContextAdapter,
    OpenAIAdapterOptions,
    GeminiContextAdapter,
    GeminiAdapterOptions,
)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class MemoryClaim:
    id: str
    text: str
    grade: str  # "proven" | "derived" | "unknown"
    attributes: dict = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class NoopAuditWriter(AuditWriter):
    async def append(self, entry: AuditEntry) -> None:
        pass


def noop_audit_writer() -> NoopAuditWriter:
    return NoopAuditWriter()


def make_proposal(units: list[MemoryClaim]) -> Proposal[MemoryClaim]:
    return Proposal(id=f"prop-{int(time.time() * 1000)}", kind="response", units=units)


def sep(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def subsep(title: str) -> None:
    print()
    print("  " + "-" * 60)
    print(f"  {title}")
    print("  " + "-" * 60)


def pass_(msg: str) -> None:
    print(f"  [PASS] {msg}")


def label(key: str, value: object) -> None:
    import json
    try:
        v = json.dumps(value)
    except (TypeError, ValueError):
        v = repr(value)
    print(f"    {key.ljust(22)}: {v}")


def explain(text: str) -> None:
    width = 66
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > width and current:
            lines.append(current.strip())
            current = word
        else:
            current = candidate
    if current.strip():
        lines.append(current.strip())
    for line in lines:
        print(f"  | {line}")


# ---------------------------------------------------------------------------
# MemoryPolicy
# ---------------------------------------------------------------------------

class MemoryPolicy(GatePolicy[MemoryClaim]):
    """
    MemoryPolicy is the INJECTED business logic for this demo.

    jingu-trust-gate core carries ZERO business semantics.
    The policy decides what 'valid' means for this domain.

    Gate rules implemented here:
      - grade=proven + no evidence refs  ->  MISSING_EVIDENCE  ->  reject
      - has_brand=True + evidence has no brand attr  ->  OVER_SPECIFIC_BRAND  ->  downgrade to 'derived'
      - otherwise  ->  approve

    Conflict detection is injected per-scenario so the demo can control
    when ITEM_CONFLICT fires without duplicating policy logic.
    """

    def __init__(self, injected_conflicts: list[ConflictAnnotation] | None = None) -> None:
        self._injected_conflicts = injected_conflicts or []

    # Gate Step 1: structural validation
    def validate_structure(self, proposal: Proposal[MemoryClaim]) -> StructureValidationResult:
        if len(proposal.units) == 0:
            return StructureValidationResult(
                kind="structure",
                valid=False,
                errors=[
                    StructureError(
                        field="units",
                        reason_code="EMPTY_UNITS",
                        message="proposal must have at least one unit",
                    )
                ],
            )
        return StructureValidationResult(kind="structure", valid=True, errors=[])

    # Gate Step 2: bind each unit to its supporting evidence
    def bind_support(
        self, unit: MemoryClaim, support_pool: list[SupportRef]
    ) -> UnitWithSupport[MemoryClaim]:
        refs = unit.evidence_refs or []
        matched = [s for s in support_pool if s.source_id in refs]
        return UnitWithSupport(
            unit=unit,
            support_ids=[s.id for s in matched],
            support_refs=matched,
        )

    # Gate Step 3: per-unit semantic evaluation
    def evaluate_unit(
        self,
        unit_with_support: UnitWithSupport[MemoryClaim],
        context: dict,
    ) -> UnitEvaluationResult:
        unit = unit_with_support.unit
        support_ids = unit_with_support.support_ids
        support_refs = unit_with_support.support_refs

        # Rule A: proven claim must have at least one evidence reference
        if unit.grade == "proven" and len(support_ids) == 0:
            return reject(unit.id, "MISSING_EVIDENCE")

        # Rule B: claim asserts a specific brand but evidence has no brand attribute
        if unit.attributes.get("has_brand"):
            evidence_has_brand = any(
                s.attributes.get("brand") is not None for s in support_refs
            )
            if not evidence_has_brand:
                return downgrade(unit.id, "OVER_SPECIFIC_BRAND", "derived")

        return approve(unit.id)

    # Gate Step 4: cross-unit conflict detection
    def detect_conflicts(
        self,
        units: list[UnitWithSupport[MemoryClaim]],
        support_pool: list[SupportRef],
    ) -> list[ConflictAnnotation]:
        return self._injected_conflicts

    # Gate Step 5: render admitted units -> VerifiedContext
    def render(
        self,
        admitted_units: list[AdmittedUnit],
        support_pool: list[SupportRef],
        context: RenderContext,
    ) -> VerifiedContext:
        blocks: list[VerifiedBlock] = []
        for u in admitted_units:
            block = VerifiedBlock(
                source_id=u.unit_id,
                content=u.unit.text,  # type: ignore[union-attr]
            )
            if u.status == "downgraded":
                if u.applied_grades:
                    block.grade = u.applied_grades[-1]
                if any(r.reason_code == "OVER_SPECIFIC_BRAND" for r in u.evaluation_results):
                    block.unsupported_attributes = ["brand"]
            if u.status == "approved_with_conflict" and u.conflict_annotations:
                block.conflict_note = "; ".join(
                    f"{c.conflict_code}: {c.description or ''}"
                    for c in u.conflict_annotations
                )
            blocks.append(block)

        conflicts_count = sum(
            1 for u in admitted_units if u.status == "approved_with_conflict"
        )
        return VerifiedContext(
            admitted_blocks=blocks,
            summary=VerifiedContextSummary(
                admitted=len(admitted_units),
                rejected=0,
                conflicts=conflicts_count,
            ),
        )

    # Gate Step 6: build structured retry feedback
    def build_retry_feedback(
        self,
        unit_results: list[UnitEvaluationResult],
        context: RetryContext,
    ) -> RetryFeedback:
        errors = [
            RetryError(
                unit_id=r.unit_id,
                reason_code=r.reason_code,
                details={"suggested_grade": r.new_grade} if r.new_grade else {},
            )
            for r in unit_results
            if r.decision in ("reject", "downgrade")
        ]
        return RetryFeedback(
            summary=(
                f"Attempt {context.attempt}/{context.max_retries} failed. "
                f"{len(errors)} unit(s) need correction."
            ),
            errors=errors,
        )


# ===========================================================================
# OPENING: Three Iron Laws
# ===========================================================================

def print_iron_laws() -> None:
    sep("WHAT IS JINGU-TRUST-GATE?")
    print()
    explain("jingu-trust-gate = deterministic admission control for LLM output.")
    print()
    explain("It treats LLM output as untrusted input — the same way a web server treats user input.")
    explain("LLM proposes. jingu-trust-gate decides what can be trusted.")
    print()
    print("  THREE IRON LAWS:")
    print()
    print("  Law 1 — Gate Engine: zero LLM calls")
    explain("All gate evaluation is pure code. No AI judges AI. This guarantees determinism and auditability.")
    print()
    print("  Law 2 — Policy is injected")
    explain("jingu-trust-gate core carries no business semantics. The caller injects a GatePolicy that defines what 'valid' means for their domain.")
    print()
    print("  Law 3 — Every admission decision is written to audit log")
    explain("Every admit() call writes an AuditEntry. The system is accountable by design, not by convention.")
    print()
    print("  THE PIPELINE:")
    print()
    print("    LLM output")
    print("       |")
    print("       v")
    print("    Proposal[TUnit]             <- typed, schema-validated by LLM API")
    print("       |")
    print("       v")
    print("    gate.admit(proposal, support)")
    print("       |  Gate Step 1: validate_structure()  — structural check")
    print("       |  Gate Step 2: bind_support()        — match units to evidence")
    print("       |  Gate Step 3: evaluate_unit()       — semantic evaluation")
    print("       |  Gate Step 4: detect_conflicts()    — cross-unit truth check")
    print("       |  (all pure code, zero LLM)")
    print("       v")
    print("    AdmissionResult             <- who passed, who failed, conflicts")
    print("       |")
    print("       v")
    print("    gate.render(result)      <- Gate Step 5: policy renders admitted units")
    print("       |")
    print("       v")
    print("    VerifiedContext             <- semantic structure, NOT user text")
    print("       |")
    print("       v")
    print("    Adapter.adapt(verified_ctx) <- wire format for target LLM API")
    print("       |")
    print("       v")
    print("    Claude / OpenAI / Gemini API call")
    print()
    explain("IMPORTANT: jingu-trust-gate does not write 'You have milk in the fridge' for users.")
    explain("It produces search_result blocks / tool messages / content parts that the LLM uses to generate the final response. This is the correct separation of concerns.")
    print()
    print("  WHEN TO USE jingu-trust-gate:")
    print()
    explain("USE when: you have a retrieval system (RAG, vector DB, knowledge base) and LLM output must be grounded in it. Use when you need to prevent hallucinated certainty. Use when you run multi-LLM pipelines. Use when you need audit trails.")
    print()
    explain("DO NOT USE when: your task is purely creative (writing, brainstorming) with no support pool. Do not use when you need sub-100ms latency. Do not use if you expect jingu-trust-gate to rewrite or fix LLM output — it labels problems, it does not solve them.")


# ===========================================================================
# Scenario 1: Happy Path
# ===========================================================================

async def scenario1() -> None:
    sep("Scenario 1: Happy Path — Zero Friction")
    print()
    explain("The simplest case. LLM proposes 2 claims, both with evidence. The gate approves both without friction. This is the baseline: when LLM does its job correctly, jingu-trust-gate gets out of the way.")
    print()

    subsep("INPUT — What the LLM proposed")
    print()
    print("  Proposal:")
    print('    claim-1: "You have milk in the fridge"  grade=proven   evidence_refs=["obs-001"]')
    print('    claim-2: "You seem to buy milk weekly"  grade=derived  evidence_refs=["obs-002"]')
    print()
    print("  Support pool:")
    print('    obs-001: source_type=observation  confidence=0.95  attributes={item:"milk", location:"fridge"}')
    print('    obs-002: source_type=inference    confidence=0.75  attributes={pattern:"weekly-purchase"}')

    support: list[SupportRef] = [
        SupportRef(
            id="sup-1",
            source_type="observation",
            source_id="obs-001",
            confidence=0.95,
            attributes={"item": "milk", "location": "fridge"},
            retrieved_at="2024-01-10T10:00:00Z",
        ),
        SupportRef(
            id="sup-2",
            source_type="inference",
            source_id="obs-002",
            confidence=0.75,
            attributes={"pattern": "weekly-purchase"},
            retrieved_at="2024-01-10T10:00:00Z",
        ),
    ]

    proposal = make_proposal([
        MemoryClaim(
            id="claim-1",
            text="You have milk in the fridge",
            grade="proven",
            attributes={},
            evidence_refs=["obs-001"],
        ),
        MemoryClaim(
            id="claim-2",
            text="You seem to buy milk weekly",
            grade="derived",
            attributes={},
            evidence_refs=["obs-002"],
        ),
    ])

    gate = create_trust_gate(
        policy=MemoryPolicy(),
        audit_writer=noop_audit_writer(),
    )

    subsep("GATE EXECUTION — gate.admit()")
    print()
    print("  Step 1 — validate_structure(): 2 units present  -> valid")
    print("  Step 2 — bind_support():")
    print("            claim-1 evidence_refs=[obs-001]  matched sup-1")
    print("            claim-2 evidence_refs=[obs-002]  matched sup-2")
    print("  Step 3 — evaluate_unit():")
    print("            claim-1: grade=proven, support_ids=[sup-1]  -> approve (OK)")
    print("            claim-2: grade=derived, support_ids=[sup-2]  -> approve (OK)")
    print("  Step 4 — detect_conflicts(): no conflicts detected")

    result = await gate.admit(proposal, support)
    explanation = gate.explain(result)

    subsep("OUTPUT — AdmissionResult")
    print()
    label("admitted_units.length", len(result.admitted_units))
    label("rejected_units.length", len(result.rejected_units))
    label("has_conflicts", result.has_conflicts)
    for u in result.admitted_units:
        label(f"  {u.unit_id}.status", u.status)
        label(f"  {u.unit_id}.applied_grades", u.applied_grades)
    print()
    print("  explain() summary:")
    label("  total_units", explanation.total_units)
    label("  approved", explanation.approved)
    label("  downgraded", explanation.downgraded)
    label("  rejected", explanation.rejected)
    label("  gate_reason_codes", explanation.gate_reason_codes)

    subsep("RENDER — gate.render() -> VerifiedContext")
    print()
    verified_ctx = gate.render(result)
    for block in verified_ctx.admitted_blocks:
        print(f"    block[{block.source_id}]:")
        label("      content", block.content)
        if block.grade:
            label("      grade", block.grade)
        if block.conflict_note:
            label("      conflict_note", block.conflict_note)
    print()
    explain("VerifiedContext is the input to the LLM API adapter — not user text. The adapter converts it to the wire format the target LLM expects.")

    assert len(result.admitted_units) == 2
    assert len(result.rejected_units) == 0
    assert result.has_conflicts is False
    for u in result.admitted_units:
        assert u.status == "approved"
    assert len(verified_ctx.admitted_blocks) == 2

    print()
    pass_("admitted_units.length == 2")
    pass_("all status == 'approved'")
    pass_("rejected_units.length == 0")
    pass_("has_conflicts == False")
    pass_("VerifiedContext has 2 blocks")


# ===========================================================================
# Scenario 2: Missing Evidence (Anti-Pattern caught)
# ===========================================================================

async def scenario2() -> None:
    sep("Scenario 2: Missing Evidence — Hallucination Pattern Caught")
    print()
    explain("ANTI-PATTERN: LLM asserts grade=proven but provides no evidence reference. This is the classic hallucination pattern: confident statement, no backing.")
    print()
    explain("The gate catches this deterministically. No LLM re-evaluation. Pure code.")

    subsep("INPUT — What the LLM proposed")
    print()
    print("  Proposal:")
    print('    claim-1: "You have exactly 3 apples"  grade=proven  evidence_refs=[]')
    print()
    print("  Notice: grade=proven but evidence_refs is empty.")
    print("  The LLM stated a precise quantity with full confidence and zero backing.")

    proposal = make_proposal([
        MemoryClaim(
            id="claim-1",
            text="You have exactly 3 apples",
            grade="proven",
            attributes={"has_quantity": True},
            evidence_refs=[],
        ),
    ])

    gate = create_trust_gate(
        policy=MemoryPolicy(),
        audit_writer=noop_audit_writer(),
    )

    subsep("GATE EXECUTION — gate.admit()")
    print()
    print("  Step 1 — validate_structure(): 1 unit present  -> valid")
    print("  Step 2 — bind_support(): evidence_refs=[]  -> support_ids=[]  (nothing matched)")
    print("  Step 3 — evaluate_unit():")
    print("            grade=proven, support_ids=[]")
    print("            Rule A fires: proven claim requires at least one evidence reference")
    print("            -> decision: reject  reason_code: MISSING_EVIDENCE")
    print("  Step 4 — detect_conflicts(): no injected conflicts -> returns []")

    result = await gate.admit(proposal, [])

    subsep("OUTPUT — AdmissionResult")
    print()
    rejected = result.rejected_units[0]
    label("admitted_units.length", len(result.admitted_units))
    label("rejected_units.length", len(result.rejected_units))
    print()
    print("  Rejected unit details:")
    label("    unit_id", rejected.unit_id)
    label("    unit.text", rejected.unit.text)  # type: ignore[union-attr]
    label("    unit.grade", rejected.unit.grade)  # type: ignore[union-attr]
    label("    reason_code", rejected.evaluation_results[0].reason_code)
    label("    decision", rejected.evaluation_results[0].decision)
    print()
    explain("The claim is not admitted. It will not reach the LLM context. The LLM will not generate a response based on hallucinated certainty.")
    print()
    explain("WHY THIS MATTERS: If this claim were passed through, the LLM would tell the user 'You have exactly 3 apples' with high confidence. The user would act on false information. jingu-trust-gate prevents this at the boundary.")
    print()
    explain("LIMITATION: jingu-trust-gate cannot tell why support_ids is empty. 'LLM cited wrong evidence refs' and 'the evidence simply does not exist in your system' both look identical — MISSING_EVIDENCE. If your support pool has no observations about apples, retry will not fix this. Build your retrieval system first, then use jingu-trust-gate to enforce that claims stay within what was retrieved.")

    assert len(result.admitted_units) == 0
    assert len(result.rejected_units) == 1
    assert result.rejected_units[0].evaluation_results[0].reason_code == "MISSING_EVIDENCE"
    assert result.rejected_units[0].evaluation_results[0].decision == "reject"

    print()
    pass_("admitted_units.length == 0  (nothing passed)")
    pass_("rejected_units.length == 1")
    pass_("reason_code == 'MISSING_EVIDENCE'")
    pass_("decision == 'reject'")


# ===========================================================================
# Scenario 3: Over-Specificity (Precision Degraded, Not Rejected)
# ===========================================================================

async def scenario3() -> None:
    sep("Scenario 3: Over-Specificity — Precision Calibrated to Evidence")
    print()
    explain("ANTI-PATTERN: LLM says 'Coca-Cola' but the evidence only says 'a drink' (no brand attribute). The claim is more specific than what the evidence supports.")
    print()
    explain("jingu-trust-gate response: downgrade grade from proven to derived, mark unsupported_attributes=['brand']. The claim IS admitted — but with reduced confidence and a caveat.")
    print()
    explain("IMPORTANT: jingu-trust-gate does NOT rewrite 'Coca-Cola' to 'drink'. It is not an editor. It marks the precision boundary and lets the downstream LLM decide how to communicate it.")

    subsep("INPUT — What the LLM proposed")
    print()
    print("  Proposal:")
    print('    claim-1: "You have Coca-Cola"  grade=proven  has_brand=True  evidence_refs=["obs-001"]')
    print()
    print("  Support pool:")
    print('    obs-001: source_type=observation  attributes={item:"drink"}  (no brand field)')
    print()
    print('  The evidence knows there is "a drink" but does NOT know the brand.')

    support: list[SupportRef] = [
        SupportRef(
            id="sup-1",
            source_type="observation",
            source_id="obs-001",
            confidence=0.8,
            attributes={
                "item": "drink",
                # brand is intentionally absent — evidence does not know the brand
            },
            retrieved_at="2024-01-10T09:00:00Z",
        ),
    ]

    proposal = make_proposal([
        MemoryClaim(
            id="claim-1",
            text="You have Coca-Cola",
            grade="proven",
            attributes={"has_brand": True},
            evidence_refs=["obs-001"],
        ),
    ])

    gate = create_trust_gate(
        policy=MemoryPolicy(),
        audit_writer=noop_audit_writer(),
    )

    subsep("GATE EXECUTION — gate.admit()")
    print()
    print("  Step 1 — validate_structure(): 1 unit  -> valid")
    print("  Step 2 — bind_support(): evidence_refs=[obs-001]  -> matched sup-1")
    print("  Step 3 — evaluate_unit():")
    print("            grade=proven, support_ids=[sup-1]  -> passes Rule A (has evidence)")
    print("            has_brand=True -> inspect evidence for brand attribute")
    print("            sup-1.attributes['brand'] is None")
    print("            Rule B fires: claim asserts brand but evidence has none")
    print("            -> decision: downgrade  reason_code: OVER_SPECIFIC_BRAND  new_grade: 'derived'")

    result = await gate.admit(proposal, support)
    admitted = result.admitted_units[0]

    subsep("OUTPUT — AdmissionResult")
    print()
    label("admitted_units.length", len(result.admitted_units))
    label("rejected_units.length", len(result.rejected_units))
    print()
    print("  Admitted unit (downgraded):")
    label("    unit_id", admitted.unit_id)
    label("    status", admitted.status)
    label("    applied_grades", admitted.applied_grades)
    label("    reason_code", admitted.evaluation_results[0].reason_code)

    verified_ctx = gate.render(result)
    block = verified_ctx.admitted_blocks[0]

    print()
    print("  VerifiedContext block (render output):")
    label("    source_id", block.source_id)
    label("    content", block.content)
    label("    grade", block.grade)
    label("    unsupported_attributes", block.unsupported_attributes)
    print()
    explain("The downstream LLM receives this block. It sees 'Coca-Cola' as the content, but grade=derived and unsupported_attributes=['brand'] as caveats. The LLM can decide to say 'there appears to be a soft drink' rather than asserting the brand.")
    print()
    explain("This is precision calibration: jingu-trust-gate tells the LLM exactly where its confidence boundary is.")

    assert len(result.admitted_units) == 1
    assert admitted.status == "downgraded"
    assert "derived" in admitted.applied_grades
    assert admitted.evaluation_results[0].reason_code == "OVER_SPECIFIC_BRAND"
    assert block.unsupported_attributes == ["brand"]
    assert block.grade == "derived"

    print()
    pass_("unit admitted (downgrade != reject)")
    pass_("status == 'downgraded'")
    pass_("applied_grades includes 'derived'")
    pass_("reason_code == 'OVER_SPECIFIC_BRAND'")
    pass_("VerifiedContext block.unsupported_attributes == ['brand']")
    pass_("VerifiedContext block.grade == 'derived'")


# ===========================================================================
# Scenario 4: Conflict Detection (Truth Surfaced, Not Hidden)
# ===========================================================================

async def scenario4() -> None:
    sep("Scenario 4: Conflict Detection — Truth Surfaced, Not Hidden")
    print()
    explain("Two contradictory claims: 'You have milk' (obs-1, Jan 1) and 'You have no milk' (obs-2, Jan 2). Both have evidence. Both pass individual evaluation.")
    print()
    explain("ANTI-PATTERN jingu-trust-gate prevents: silently picking one claim as 'winner'. That would hide information from the LLM and produce incorrect responses.")
    print()
    explain("jingu-trust-gate response: BOTH claims are admitted with status=approved_with_conflict. The conflict is annotated. The downstream LLM receives both facts and can surface the contradiction to the user.")

    subsep("INPUT — What the LLM proposed")
    print()
    print("  Proposal:")
    print('    claim-1: "You have milk"     grade=proven  evidence_refs=["obs-1"]')
    print('    claim-2: "You have no milk"  grade=proven  evidence_refs=["obs-2"]')
    print()
    print("  Support pool:")
    print('    obs-1: 2024-01-01  attributes={item:"milk", present:true}')
    print('    obs-2: 2024-01-02  attributes={item:"milk", present:false}')
    print()
    print("  Injected conflict: ITEM_CONFLICT  severity=informational")

    support: list[SupportRef] = [
        SupportRef(
            id="sup-1",
            source_type="observation",
            source_id="obs-1",
            confidence=0.9,
            attributes={"item": "milk", "present": True},
            retrieved_at="2024-01-01T08:00:00Z",
        ),
        SupportRef(
            id="sup-2",
            source_type="observation",
            source_id="obs-2",
            confidence=0.9,
            attributes={"item": "milk", "present": False},
            retrieved_at="2024-01-02T08:00:00Z",
        ),
    ]

    injected_conflicts: list[ConflictAnnotation] = [
        ConflictAnnotation(
            unit_ids=["claim-1", "claim-2"],
            conflict_code="ITEM_CONFLICT",
            sources=["obs-1", "obs-2"],
            severity="informational",
            description="claim-1 and claim-2 contradict each other: obs-1 says milk present=true, obs-2 says present=false",
        ),
    ]

    proposal = make_proposal([
        MemoryClaim(
            id="claim-1",
            text="You have milk",
            grade="proven",
            attributes={},
            evidence_refs=["obs-1"],
        ),
        MemoryClaim(
            id="claim-2",
            text="You have no milk",
            grade="proven",
            attributes={},
            evidence_refs=["obs-2"],
        ),
    ])

    gate = create_trust_gate(
        policy=MemoryPolicy(injected_conflicts=injected_conflicts),
        audit_writer=noop_audit_writer(),
    )

    subsep("GATE EXECUTION — gate.admit()")
    print()
    print("  Step 1 — validate_structure(): 2 units  -> valid")
    print("  Step 2 — bind_support():")
    print("            claim-1 -> matched sup-1")
    print("            claim-2 -> matched sup-2")
    print("  Step 3 — evaluate_unit():")
    print("            claim-1: grade=proven, has support  -> approve (OK)")
    print("            claim-2: grade=proven, has support  -> approve (OK)")
    print("  Step 4 — detect_conflicts():")
    print("            ITEM_CONFLICT detected: claim-1 <-> claim-2")
    print("            severity=informational (both kept, annotated)")
    print("            if severity=blocking -> both would be rejected")

    result = await gate.admit(proposal, support)

    subsep("OUTPUT — AdmissionResult")
    print()
    label("admitted_units.length", len(result.admitted_units))
    label("rejected_units.length", len(result.rejected_units))
    label("has_conflicts", result.has_conflicts)
    print()
    for u in result.admitted_units:
        print(f"  Unit: {u.unit_id}")
        label("    status", u.status)
        if u.conflict_annotations:
            label("    conflict_annotations[0].conflict_code", u.conflict_annotations[0].conflict_code)
            label("    conflict_annotations[0].severity", u.conflict_annotations[0].severity)

    subsep("RENDER — VerifiedContext with conflict notes")
    print()
    verified_ctx = gate.render(result)
    for block in verified_ctx.admitted_blocks:
        print(f"  Block: {block.source_id}")
        label("    content", block.content)
        label("    conflict_note", block.conflict_note)
    label("  summary.conflicts", verified_ctx.summary.conflicts)
    print()
    explain("The downstream LLM receives BOTH claims, each with a conflict_note. It can say: 'My records are inconsistent — obs-1 from Jan 1 shows milk was present, but obs-2 from Jan 2 says it was not. Please check your fridge.' This is truthful.")
    print()
    explain("severity='informational' means: annotate and pass through. severity='blocking' would mean: reject both, do not surface either until conflict is resolved.")

    assert result.has_conflicts is True
    assert len(result.admitted_units) == 2
    assert len(result.rejected_units) == 0
    for u in result.admitted_units:
        assert u.status == "approved_with_conflict"
        assert u.conflict_annotations and len(u.conflict_annotations) > 0
        assert u.conflict_annotations[0].conflict_code == "ITEM_CONFLICT"
        assert u.conflict_annotations[0].severity == "informational"
    assert all(b.conflict_note is not None for b in verified_ctx.admitted_blocks)

    print()
    pass_("has_conflicts == True")
    pass_("both units in admitted_units (not rejected)")
    pass_("both status == 'approved_with_conflict'")
    pass_("conflict_annotations[0].conflict_code == 'ITEM_CONFLICT'")
    pass_("conflict_annotations[0].severity == 'informational'")
    pass_("VerifiedContext: both blocks have conflict_note")

    # ── Part B: blocking conflict ─────────────────────────────────────────────

    subsep("Part B — severity=blocking: both claims force-rejected")
    print()
    explain("Same two claims. Same evidence. But now the conflict is severity=blocking — the claims are mutually exclusive in a way that makes it unsafe to surface either one. Example: two inventory records for the same product disagree on whether it is in stock.")
    print()
    print("  Injected conflict: ITEM_CONFLICT  severity=blocking")

    blocking_conflicts: list[ConflictAnnotation] = [
        ConflictAnnotation(
            unit_ids=["claim-1", "claim-2"],
            conflict_code="ITEM_CONFLICT",
            sources=["obs-1", "obs-2"],
            severity="blocking",
            description="claim-1 and claim-2 are mutually exclusive — cannot surface either",
        ),
    ]

    gate_blocking = create_trust_gate(
        policy=MemoryPolicy(injected_conflicts=blocking_conflicts),
        audit_writer=noop_audit_writer(),
    )

    result_blocking = await gate_blocking.admit(proposal, support)
    ctx_blocking = gate_blocking.render(result_blocking)

    print()
    print("  Step 4 — detect_conflicts():")
    print("            ITEM_CONFLICT detected: claim-1 <-> claim-2")
    print("            severity=blocking -> both force-rejected as BLOCKING_CONFLICT")
    print()

    subsep("OUTPUT — blocking conflict result")
    print()
    label("admitted_units.length", len(result_blocking.admitted_units))
    label("rejected_units.length", len(result_blocking.rejected_units))
    label("has_conflicts", result_blocking.has_conflicts)
    print()
    for u in result_blocking.rejected_units:
        print(f"  Unit: {u.unit_id}")
        label("    status", u.status)
        if u.evaluation_results:
            label("    reason_code", u.evaluation_results[0].reason_code)
    print()
    label("admitted_blocks.length", len(ctx_blocking.admitted_blocks))
    print()
    explain("admitted_blocks is empty. The downstream LLM receives no claims — only the instructions field. It can tell the user: 'Stock status is inconsistent across records. Please check the product page directly.' It does not guess.")
    print()
    explain("LIMITATION: jingu-trust-gate cannot resolve the conflict. It surfaces the problem and stops. A human or a dedicated reconciliation step must decide which record is authoritative.")

    assert len(result_blocking.admitted_units) == 0
    assert len(result_blocking.rejected_units) == 2
    assert result_blocking.has_conflicts is True
    for u in result_blocking.rejected_units:
        assert u.status == "rejected"
        assert u.evaluation_results[0].reason_code == "BLOCKING_CONFLICT"
    assert len(ctx_blocking.admitted_blocks) == 0

    print()
    pass_("admitted_units.length == 0 (nothing admitted past the gate)")
    pass_("rejected_units.length == 2 (both force-rejected)")
    pass_("reason_code == 'BLOCKING_CONFLICT' on both")
    pass_("admitted_blocks.length == 0 (LLM receives empty context)")


# ===========================================================================
# Scenario 5: Semantic Retry Loop
# ===========================================================================

async def scenario5() -> None:
    sep("Scenario 5: Semantic Retry Loop — Evidence-Driven Correction")
    print()
    explain("ANTI-PATTERN: LLM provides a 'proven' claim with no evidence. When told to retry, LLM just softens the language to 'derived' — without supplying evidence. This is wrong.")
    print()
    explain("jingu-trust-gate response: RetryFeedback is a TYPED STRUCT, not a string. It carries unit_id, reason_code, and structured details. The fix the LLM must make is explicit: supply evidence. Softening language is not the fix.")
    print()
    explain("The LLMInvoker is responsible for serializing RetryFeedback as tool_result + is_error:true for Claude's built-in retry understanding. jingu-trust-gate controls WHETHER to retry. Invoker controls HOW.")

    subsep("RETRY FEEDBACK TYPE (load-bearing contract)")
    print()
    print("  @dataclass")
    print("  class RetryFeedback:")
    print("      summary: str                     # human-readable, for logging")
    print("      errors: list[RetryError]")
    print()
    print("  @dataclass")
    print("  class RetryError:")
    print("      reason_code: str                 # MISSING_EVIDENCE | OVER_SPECIFIC_BRAND ...")
    print("      unit_id: Optional[str]           # which unit failed")
    print("      details: dict[str, Any]          # e.g. {'suggested_grade': 'derived'}")
    print()
    explain("The LLMInvoker receives this struct. For Claude: serialize as tool_result with is_error:true. The is_error flag activates Claude's built-in retry understanding.")

    subsep("SCENARIO — Two LLM Invocations")
    print()
    print("  Attempt 1 (LLM invocation 1):")
    print('    claim: "You have 5 cans of soup"  grade=proven  evidence_refs=[]')
    print("    -> gate verdict: MISSING_EVIDENCE -> reject")
    print("    -> jingu-trust-gate builds RetryFeedback")
    print("    -> sends to LLMInvoker")
    print()
    print("  Attempt 2 (LLM invocation 2):")
    print('    LLM receives RetryFeedback.errors[0].reason_code = "MISSING_EVIDENCE"')
    print('    LLM understands: the fix is supplying evidence, not softening language')
    print('    claim: "You have canned goods in the pantry"  grade=proven  evidence_refs=["obs-pantry"]')
    print("    -> gate verdict: OK -> approve")

    support: list[SupportRef] = [
        SupportRef(
            id="sup-pantry",
            source_type="observation",
            source_id="obs-pantry",
            confidence=0.7,
            attributes={"item": "canned-goods", "location": "pantry"},
            retrieved_at="2024-01-10T12:00:00Z",
        ),
    ]

    captured_feedback: list[RetryFeedback | None] = [None]
    invoker_call_count: list[int] = [0]

    async def invoker(
        prompt: str,
        feedback: RetryFeedback | None = None,
    ) -> Proposal[MemoryClaim]:
        invoker_call_count[0] += 1

        if feedback is not None:
            captured_feedback[0] = feedback

        if invoker_call_count[0] == 1:
            return make_proposal([
                MemoryClaim(
                    id="claim-1",
                    text="You have 5 cans of soup",
                    grade="proven",
                    attributes={"has_quantity": True},
                    evidence_refs=[],
                ),
            ])

        return make_proposal([
            MemoryClaim(
                id="claim-1",
                text="You have canned goods in the pantry",
                grade="proven",
                attributes={},
                evidence_refs=["obs-pantry"],
            ),
        ])

    gate = create_trust_gate(
        policy=MemoryPolicy(),
        audit_writer=noop_audit_writer(),
        retry=RetryConfig(max_retries=3, retry_on_decisions=["reject"]),
    )

    result = await gate.admit_with_retry(invoker, support, "What food do I have?")

    subsep("RetryFeedback that was sent to LLMInvoker")
    print()
    fb = captured_feedback[0]
    if fb:
        label("summary", fb.summary)
        print("  errors:")
        for err in fb.errors:
            label("    unit_id", err.unit_id)
            label("    reason_code", err.reason_code)
            if err.details:
                label("    details", err.details)

    subsep("FINAL AdmissionResult")
    print()
    label("retry_attempts", result.retry_attempts)
    label("admitted_units.length", len(result.admitted_units))
    if result.admitted_units:
        label("final unit status", result.admitted_units[0].status)
        label("final unit text", result.admitted_units[0].unit.text)  # type: ignore[union-attr]
    print()
    explain("KEY POINT: The fix was supplying evidence, not softening language. If the LLM had just changed grade from 'proven' to 'derived' without supplying evidence, the retry would have passed — but with wrong semantics. The correct fix is always: ground the claim.")
    print()
    explain("LIMITATION: retry is locally effective, not globally convergent. It works when the LLM cited wrong evidence refs. It does NOT work when the support pool itself is missing the data — jingu-trust-gate cannot distinguish between these two cases. The support pool is fixed for the entire retry loop. If the evidence was never retrieved, no number of retries will fix it.")

    assert result.retry_attempts == 2
    assert len(result.admitted_units) == 1
    assert result.admitted_units[0].status == "approved"
    assert captured_feedback[0] is not None, "RetryFeedback must have been sent"
    assert any(e.reason_code == "MISSING_EVIDENCE" for e in captured_feedback[0].errors)

    print()
    pass_("retry_attempts == 2  (attempt 1 failed, attempt 2 succeeded)")
    pass_("admitted_units.length == 1")
    pass_("final status == 'approved'")
    pass_("RetryFeedback.errors[0].reason_code == 'MISSING_EVIDENCE'")
    pass_("Feedback is a typed struct, not a raw error string")


# ===========================================================================
# Scenario 6: All Three Adapters — Same VerifiedContext, Different Wire Formats
# ===========================================================================

async def scenario6() -> None:
    sep("Scenario 6: All Three Adapters — One VerifiedContext, Three Wire Formats")
    print()
    explain("The SAME VerifiedContext is fed to all three adapters. Each adapter produces the wire format its target LLM API expects. jingu-trust-gate is LLM-agnostic by design.")
    print()
    explain("This is the correct separation of concerns: jingu-trust-gate produces semantic structure, adapters translate it. You can swap target LLMs without changing your admission logic.")

    # Use the same VerifiedContext for all adapters
    verified_ctx = VerifiedContext(
        admitted_blocks=[
            VerifiedBlock(
                source_id="claim-1",
                content="You have milk in the fridge",
            ),
            VerifiedBlock(
                source_id="claim-2",
                content="You have a drink",
                grade="derived",
                unsupported_attributes=["brand"],
            ),
            VerifiedBlock(
                source_id="claim-3",
                content="You have milk",
                conflict_note="ITEM_CONFLICT: claim-1 and claim-4 contradict each other about milk presence",
            ),
        ],
        summary=VerifiedContextSummary(admitted=3, rejected=0, conflicts=1),
    )

    subsep("INPUT — VerifiedContext (shared by all adapters)")
    print()
    print("  admitted_blocks:")
    print('    [claim-1]  content="You have milk in the fridge"                 (clean)')
    print('    [claim-2]  content="You have a drink"  grade=derived             (downgraded)')
    print('               unsupported_attributes=["brand"]')
    print('    [claim-3]  content="You have milk"  conflict_note=ITEM_CONFLICT  (conflict)')

    # --- Claude Adapter ---
    subsep("ADAPTER 1 — ClaudeContextAdapter -> search_result blocks")
    print()
    explain("Claude API supports native search_result blocks with citations. jingu-trust-gate maps each VerifiedBlock to one search_result block. Claude can cite specific blocks in its response.")
    print()

    claude_adapter = ClaudeContextAdapter(options=ClaudeAdapterOptions(citations=True))
    claude_blocks = claude_adapter.adapt(verified_ctx)

    print("  Output: list[ClaudeSearchResultBlock]")
    print()
    for block in claude_blocks:
        print(f'    {{ type: "{block.type}", source: "{block.source}", title: "{block.title}" }}')
        print(f"      content[0].text: {block.content[0]['text']!r}")
        print(f"      citations.enabled: {block.citations['enabled'] if block.citations else None}")
        print()

    assert len(claude_blocks) == 3
    assert all(b.type == "search_result" for b in claude_blocks)
    assert claude_blocks[0].citations is not None and claude_blocks[0].citations["enabled"] is True
    assert "[Evidence grade: derived]" in claude_blocks[1].content[0]["text"]
    assert "[Not supported by evidence: brand]" in claude_blocks[1].content[0]["text"]
    assert "[Conflict:" in claude_blocks[2].content[0]["text"]

    pass_("all 3 blocks are type='search_result'")
    pass_("citations enabled on all blocks")
    pass_("downgraded block has [Evidence grade: derived] caveat")
    pass_("downgraded block has [Not supported by evidence: brand]")
    pass_("conflict block has [Conflict: ...] annotation")

    # --- OpenAI Adapter (user mode) ---
    subsep("ADAPTER 2a — OpenAIContextAdapter (mode='user') -> plain text user message")
    print()
    explain("OpenAI does not have a native search_result type. Verified content is serialized as plain text with semantic caveats inline. In 'user' mode, the message role is 'user' and is injected before the actual user query.")
    print()

    openai_user_adapter = OpenAIContextAdapter(options=OpenAIAdapterOptions(mode="user"))
    openai_user_msg = openai_user_adapter.adapt(verified_ctx)

    print("  Output: OpenAIChatMessage (role='user')")
    print()
    label("  role", openai_user_msg.role)
    print("  content:")
    for line in openai_user_msg.content.split("\n"):
        print(f"    {line}")

    assert openai_user_msg.role == "user"
    assert "Evidence grade: derived" in openai_user_msg.content
    assert "Not supported by evidence: brand" in openai_user_msg.content
    assert "Conflict:" in openai_user_msg.content

    print()
    pass_("role == 'user'")
    pass_("content includes Evidence grade caveat")
    pass_("content includes conflict note")

    # --- OpenAI Adapter (tool mode) ---
    subsep("ADAPTER 2b — OpenAIContextAdapter (mode='tool') -> tool result message")
    print()
    explain("In 'tool' mode, the message role is 'tool' and requires a tool_call_id. Use this when your RAG lookup is modeled as a tool call in the OpenAI function-calling loop.")
    print()

    openai_tool_adapter = OpenAIContextAdapter(
        options=OpenAIAdapterOptions(mode="tool", tool_call_id="call_abc123")
    )
    openai_tool_msg = openai_tool_adapter.adapt(verified_ctx)

    print("  Output: OpenAIChatMessage (role='tool')")
    print()
    label("  role", openai_tool_msg.role)
    label("  tool_call_id", openai_tool_msg.tool_call_id)
    print("  content (first 120 chars):")
    print(f"    {openai_tool_msg.content[:120]}...")

    assert openai_tool_msg.role == "tool"
    assert openai_tool_msg.tool_call_id == "call_abc123"

    print()
    pass_("role == 'tool'")
    pass_("tool_call_id == 'call_abc123'")

    # --- Gemini Adapter ---
    subsep("ADAPTER 3 — GeminiContextAdapter -> Content with parts array")
    print()
    explain("Gemini uses Content[] for conversation history. Each VerifiedBlock becomes one part in the Content object. This keeps Gemini's grounding granular — it can attribute individual facts to individual parts.")
    print()

    gemini_adapter = GeminiContextAdapter(options=GeminiAdapterOptions(role="user"))
    gemini_content = gemini_adapter.adapt(verified_ctx)

    print("  Output: GeminiContent")
    print()
    label("  role", gemini_content.role)
    label("  parts.length", len(gemini_content.parts))
    print("  parts:")
    for i, part in enumerate(gemini_content.parts):
        print(f"    parts[{i}].text: {part.text!r}")

    assert gemini_content.role == "user"
    assert len(gemini_content.parts) == 3
    assert "Evidence grade: derived" in gemini_content.parts[1].text
    assert "Conflict:" in gemini_content.parts[2].text

    print()
    pass_("role == 'user'")
    pass_("parts.length == 3  (one part per VerifiedBlock)")
    pass_("downgraded block part has Evidence grade caveat")
    pass_("conflict block part has Conflict annotation")

    print()
    subsep("SUMMARY — Adapter Matrix")
    print()
    print("  Target    | Format                      | Mode")
    print("  ----------|-----------------------------|--------------------------")
    print("  Claude    | search_result blocks        | search_result + citations")
    print("  OpenAI    | ChatMessage (role=user)     | user turn before query")
    print("  OpenAI    | ChatMessage (role=tool)     | tool result in tool loop")
    print("  Gemini    | Content { role, parts[] }   | user turn in contents[]")
    print()
    explain("Same VerifiedContext. Same semantic content. Four different wire formats. jingu-trust-gate is adapter-agnostic. Add a new adapter for any LLM API without changing the gate or policy.")


# ===========================================================================
# Patterns and Anti-Patterns summary
# ===========================================================================

def print_patterns_and_anti_patterns() -> None:
    sep("PATTERNS AND ANTI-PATTERNS")
    print()
    print("  PATTERNS (what jingu-trust-gate enables):")
    print()
    print("  Pattern 1: Evidence-backed admission")
    explain("Only proven claims with evidence refs pass. Grade=proven with no evidence is deterministically rejected. Calibrates confidence to what the system actually knows.")
    print()
    print("  Pattern 2: Precision calibration")
    explain("Over-specific claims are downgraded, not rejected. The claim is admitted with reduced grade and unsupported_attributes marked. The downstream LLM adjusts its language accordingly.")
    print()
    print("  Pattern 3: Conflict surfacing")
    explain("Contradictions between claims are annotated and passed through (informational) or blocked (blocking). jingu-trust-gate never silently picks a winner. LLM receives all facts and can surface the contradiction to the user.")
    print()
    print("  Pattern 4: Structured retry")
    explain("RetryFeedback is a typed struct with unit_id, reason_code, and details. The invoker serializes it as tool_result + is_error:true for Claude's built-in retry. The LLM understands exactly what to fix.")
    print()
    print("  Pattern 5: LLM-agnostic output")
    explain("VerifiedContext is an abstract semantic structure. Adapters translate it to wire format. Swap Claude for OpenAI without changing your gate or policy.")
    print()
    print("  ANTI-PATTERNS (what jingu-trust-gate prevents):")
    print()
    print("  Anti-pattern 1: Hallucinated certainty")
    explain("grade=proven with no evidence reference. The LLM stated something as fact with no backing. Gate rejects with MISSING_EVIDENCE. Prevented: LLM telling user false facts with high confidence.")
    print()
    print("  Anti-pattern 2: Specificity hallucination")
    explain("Brand assertion ('Coca-Cola') without brand evidence. Claim is more specific than what the evidence supports. Gate downgrades with OVER_SPECIFIC_BRAND. unsupported_attributes marked.")
    print()
    print("  Anti-pattern 3: Silent conflict resolution")
    explain("Picking one of two contradictory claims as the 'true' one. jingu-trust-gate rejects this by design. Both claims are admitted with conflict annotations. Information is never silently discarded.")
    print()
    print("  Anti-pattern 4: String-based retry")
    explain("Passing raw error string to LLM as retry feedback. Loses structure, loses traceability, LLM cannot extract specific unit IDs or reason codes. RetryFeedback is always a typed struct.")
    print()
    print("  Anti-pattern 5: Bypassing the gate")
    explain("Passing LLM output directly as trusted context without running gate.admit(). This defeats the entire system. Every Proposal must flow through the gate before reaching the LLM context.")
    print()
    print("  KNOWN LIMITATIONS:")
    print()
    print("  Limitation 1: jingu-trust-gate is a judge, not an editor")
    explain("It flags problems and annotates precision boundaries. It does not rewrite claims, fill in missing evidence, or auto-resolve conflicts. Downstream LLM receives the annotations and decides how to express them.")
    print()
    print("  Limitation 2: support pool is fixed per admission")
    explain("Retry works when the LLM cited wrong evidence refs. It does not work when the evidence simply does not exist in your system. jingu-trust-gate cannot distinguish between these two cases — MISSING_EVIDENCE looks identical in both.")
    print()
    print("  Limitation 3: no cross-session state")
    explain("jingu-trust-gate is stateless per call. It does not remember previous admissions or detect patterns across sessions. Cross-session governance must be implemented outside the gate.")
    print()
    print("  Limitation 4: no domain constraint on TUnit")
    explain("jingu-trust-gate does not enforce that TUnit has an id field. If your policy's evaluate_unit returns a mismatched unit_id, the audit log will have orphan entries. Your policy is responsible for ID consistency.")


# ===========================================================================
# Scenario 7: Agent Action Gate — Gate What the Agent Is Allowed to Do
# ===========================================================================

# AgentActionPolicy gates action proposals instead of knowledge claims.
#
# Gate rules:
#   R1  no "explicit_request" evidence                                → INTENT_NOT_ESTABLISHED → reject
#   R2  risk_level="high" + is_reversible=False + no "user_confirmation" → CONFIRM_REQUIRED → reject
#   R3  everything else                                               → approve
#
# source_type values used here:
#   "explicit_request"  — something the user directly asked for
#   "user_confirmation" — explicit user yes/ok for a high-risk action

@dataclass
class ActionProposal:
    id: str
    action_name: str
    description: str
    risk_level: str   # "low" | "medium" | "high"
    is_reversible: bool
    evidence_refs: list[str] = field(default_factory=list)


class AgentActionPolicy(GatePolicy):
    def validate_structure(self, proposal: Proposal) -> StructureValidationResult:
        if len(proposal.units) == 0:
            return StructureValidationResult(
                valid=False,
                errors=[StructureError(field="units", reason_code="EMPTY_UNITS")],
            )
        return StructureValidationResult(valid=True, errors=[])

    def bind_support(self, unit: ActionProposal, pool: list[SupportRef]) -> UnitWithSupport:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport, ctx) -> UnitEvaluationResult:
        unit: ActionProposal = uws.unit

        # R1: every action needs an explicit user request
        has_request = any(s.attributes.get("type") == "explicit_request" for s in uws.support_refs)
        if not has_request:
            return reject(
                unit.id,
                "INTENT_NOT_ESTABLISHED",
                note=f'No explicit_request evidence for "{unit.action_name}" — agent cannot act without user authorization',
            )

        # R2: high-risk irreversible actions need user_confirmation on top
        if unit.risk_level == "high" and not unit.is_reversible:
            has_confirmation = any(s.attributes.get("type") == "user_confirmation" for s in uws.support_refs)
            if not has_confirmation:
                return reject(
                    unit.id,
                    "CONFIRM_REQUIRED",
                    note="risk_level=high + is_reversible=False requires user_confirmation (a request alone is not enough)",
                )

        return approve(unit.id)

    def detect_conflicts(self, units: list[UnitWithSupport], pool: list[SupportRef]) -> list[ConflictAnnotation]:
        return []

    def render(self, admitted_units: list[AdmittedUnit], pool: list[SupportRef], ctx: RenderContext) -> VerifiedContext:
        blocks = [
            VerifiedBlock(
                source_id=u.unit_id,
                content=f"[{u.unit.risk_level.upper()}] {u.unit.action_name}: {u.unit.description}",
                grade=u.unit.risk_level,
            )
            for u in admitted_units
        ]
        return VerifiedContext(
            admitted_blocks=blocks,
            summary=VerifiedContextSummary(admitted=len(blocks), rejected=0, conflicts=0),
            instructions=(
                "Execute only the verified actions below. "
                "Do not re-ask for confirmation — admitted actions are already authorized. "
                "Do not execute any action that was rejected."
            ),
        )

    def build_retry_feedback(self, unit_results: list[UnitEvaluationResult], ctx: RetryContext) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        return RetryFeedback(
            summary=f"{len(failed)} action(s) rejected on attempt {ctx.attempt}/{ctx.max_retries}.",
            errors=[
                RetryError(
                    unit_id=r.unit_id,
                    reason_code=r.reason_code,
                    details={
                        "hint": (
                            "Add an explicit_request SupportRef — the user must have asked for this action"
                            if r.reason_code == "INTENT_NOT_ESTABLISHED"
                            else "Add a user_confirmation SupportRef before proposing high-risk irreversible actions"
                        ),
                    },
                )
                for r in failed
            ],
        )


async def scenario7() -> None:
    sep("Scenario 7: Agent Action Gate — Gate What the Agent Is Allowed to Do")
    print()
    explain("This is the ACTIONS pattern: the gate is not checking claims about the world — it is checking whether the agent is AUTHORIZED to take specific actions. The LLM output is a list of proposed actions. Only authorized ones execute.")
    print()
    explain("Domain: household assistant. User says 'Please order more milk.' The agent proposes 3 actions: order_milk (low risk), delete_old_shopping_list (medium risk, reversible), send_notification_email (low risk). But only order_milk was actually requested.")
    print()
    explain("WHY THIS MATTERS FOR AGENTS: without a gate, the agent executes everything it proposes. With jingu-trust-gate, each action must be traced back to explicit user authorization before it can run. The agent cannot act beyond its mandate.")

    subsep("INPUT — Authorization evidence pool")
    print()
    print("  User said: 'Please order more milk'")
    print()
    print("  Support pool:")
    print('    req-001: type=explicit_request  content="Please order more milk"')
    print('    (no user_confirmation — user never said yes to a high-risk action)')

    auth_pool: list[SupportRef] = [
        SupportRef(
            id="ref-req-1",
            source_id="req-001",
            source_type="observation",
            attributes={
                "type": "explicit_request",
                "content": "Please order more milk — we are running low",
            },
        ),
    ]

    subsep("INPUT — Agent's proposed actions")
    print()
    print('  action-1: order_milk          risk=low   reversible=True  evidence_refs=["req-001"]')
    print('  action-2: delete_old_list     risk=medium reversible=False evidence_refs=[]')
    print('             → no evidence — agent decided on its own')
    print('  action-3: send_notification   risk=low   reversible=False evidence_refs=[]')
    print('             → user asked to order milk, not to send emails')

    proposal = Proposal(
        id="prop-agent-001",
        kind="plan",
        units=[
            # action-1: user explicitly asked for this → APPROVE
            ActionProposal(
                id="action-1",
                action_name="order_milk",
                description="Place online order for 2L whole milk via grocery app",
                risk_level="low",
                is_reversible=True,
                evidence_refs=["req-001"],
            ),
            # action-2: agent decided on its own, no user request → REJECT (INTENT_NOT_ESTABLISHED)
            ActionProposal(
                id="action-2",
                action_name="delete_old_shopping_list",
                description="Delete the shopping list from 2 weeks ago",
                risk_level="medium",
                is_reversible=False,
                evidence_refs=[],
            ),
            # action-3: req-001 is about ordering milk, not sending emails → REJECT (INTENT_NOT_ESTABLISHED)
            ActionProposal(
                id="action-3",
                action_name="send_notification_email",
                description="Send email to household members that milk was ordered",
                risk_level="low",
                is_reversible=False,
                evidence_refs=[],
            ),
        ],
    )

    gate = create_trust_gate(policy=AgentActionPolicy(), audit_writer=NoopAuditWriter())

    subsep("GATE EXECUTION — gate.admit()")
    print()
    print("  Step 1 — validate_structure(): 3 units  → valid")
    print("  Step 2 — bind_support():")
    print("            action-1 evidence_refs=[req-001]  → matched ref-req-1")
    print("            action-2 evidence_refs=[]         → no support")
    print("            action-3 evidence_refs=[]         → no support")
    print("  Step 3 — evaluate_unit():")
    print("            action-1: has explicit_request  → approve")
    print("            action-2: no explicit_request   → INTENT_NOT_ESTABLISHED → reject")
    print("            action-3: no explicit_request   → INTENT_NOT_ESTABLISHED → reject")
    print("  Step 4 — detect_conflicts(): none")

    result = await gate.admit(proposal, auth_pool)
    context = gate.render(result)
    expl = gate.explain(result)

    subsep("OUTPUT — Gate decision")
    print()
    print("  Admitted (authorized to execute):")
    for u in result.admitted_units:
        label(f"    {u.unit_id} [{u.status}]", u.unit.action_name)
    print()
    print("  Rejected (blocked):")
    for u in result.rejected_units:
        ann = u.evaluation_results[0].annotations if u.evaluation_results else {}
        label(f"    {u.unit_id} [{u.evaluation_results[0].reason_code if u.evaluation_results else '?'}]", u.unit.action_name)
        if ann and ann.get("note"):
            label("      note", ann["note"])
    print()
    label("  approved", expl.approved)
    label("  rejected", expl.rejected)

    print()
    print("  VerifiedContext.instructions (sent to LLM before it executes):")
    print(f'    "{context.instructions}"')
    print()
    print("  Admitted action blocks:")
    for block in context.admitted_blocks:
        label(f"    {block.source_id}", block.content)

    print()
    explain("The agent receives VerifiedContext with 1 admitted action and the instructions. It executes order_milk. It does NOT execute delete_old_shopping_list or send_notification_email — those were never admitted through the gate.")
    print()
    explain("KEY POINT: the gate did not need to 'understand' the user request. It checked a deterministic rule: does this action have explicit_request evidence in its support pool? No evidence → no execution. This is why the gate is deterministic even in complex agentic flows.")

    # ── Part B: High-risk action with confirmation ────────────────────────────

    subsep("Part B — High-risk action: request alone is not enough")
    print()
    explain("User says 'Clear my entire order history.' Agent proposes delete_order_history (risk_level=high, is_reversible=False). The request is present — but R2 fires: high-risk irreversible actions need user_confirmation too.")
    print()
    print("  First attempt — only request present:")
    print('    req-002: type=explicit_request  content="Clear my entire order history"')
    print("    → R2: risk_level=high + is_reversible=False → CONFIRM_REQUIRED → reject")

    high_risk_pool: list[SupportRef] = [
        SupportRef(
            id="ref-req-2",
            source_id="req-002",
            source_type="observation",
            attributes={
                "type": "explicit_request",
                "content": "Clear my entire order history — I want a fresh start",
            },
        ),
    ]

    high_risk_proposal = Proposal(
        id="prop-agent-002",
        kind="plan",
        units=[
            ActionProposal(
                id="action-4",
                action_name="delete_order_history",
                description="Permanently delete all past orders",
                risk_level="high",
                is_reversible=False,
                evidence_refs=["req-002"],
            ),
        ],
    )

    high_risk_gate = create_trust_gate(policy=AgentActionPolicy(), audit_writer=NoopAuditWriter())
    result2 = await high_risk_gate.admit(high_risk_proposal, high_risk_pool)
    act4_rejected = next((u for u in result2.rejected_units if u.unit_id == "action-4"), None)
    ann2 = act4_rejected.evaluation_results[0].annotations if act4_rejected and act4_rejected.evaluation_results else {}

    label("  action-4 decision", act4_rejected.evaluation_results[0].reason_code if act4_rejected and act4_rejected.evaluation_results else "?")
    if ann2 and ann2.get("note"):
        label("  note", ann2["note"])

    print()
    print("  Second attempt — user confirms:")
    print('    confirm-001: type=user_confirmation  content="Yes, delete everything"')
    print("    → R1: has explicit_request ✓   R2: has user_confirmation ✓ → approve")

    confirmed_pool: list[SupportRef] = [
        *high_risk_pool,
        SupportRef(
            id="ref-confirm-1",
            source_id="confirm-001",
            source_type="observation",
            attributes={
                "type": "user_confirmation",
                "content": "Yes, go ahead — delete everything",
            },
        ),
    ]

    confirmed_proposal = Proposal(
        id="prop-agent-002-confirmed",
        kind="plan",
        units=[
            ActionProposal(
                id="action-4",
                action_name="delete_order_history",
                description="Permanently delete all past orders",
                risk_level="high",
                is_reversible=False,
                evidence_refs=["req-002", "confirm-001"],
            ),
        ],
    )

    result3 = await high_risk_gate.admit(confirmed_proposal, confirmed_pool)
    act4_confirmed = next((u for u in result3.admitted_units if u.unit_id == "action-4"), None)

    label("  action-4 decision (with confirmation)", act4_confirmed.status if act4_confirmed else "?")

    assert expl.approved == 1, f"expected 1 approved, got {expl.approved}"
    assert expl.rejected == 2, f"expected 2 rejected, got {expl.rejected}"
    assert any(u.unit_id == "action-1" for u in result.admitted_units)
    assert any(u.unit_id == "action-2" for u in result.rejected_units)
    assert any(u.unit_id == "action-3" for u in result.rejected_units)
    rej2 = next(u for u in result.rejected_units if u.unit_id == "action-2")
    assert rej2.evaluation_results[0].reason_code == "INTENT_NOT_ESTABLISHED"
    assert act4_rejected is not None and act4_rejected.evaluation_results[0].reason_code == "CONFIRM_REQUIRED"
    assert act4_confirmed is not None and act4_confirmed.status == "approved"

    print()
    pass_("action-1 (order_milk) approved — user explicitly requested it")
    pass_("action-2 (delete_old_list) rejected — INTENT_NOT_ESTABLISHED (agent decided on its own)")
    pass_("action-3 (send_notification_email) rejected — INTENT_NOT_ESTABLISHED (no email request)")
    pass_("action-4 (delete_order_history) rejected first attempt — CONFIRM_REQUIRED")
    pass_("action-4 approved after user_confirmation added")
    pass_("VerifiedContext.instructions tells agent which actions to execute")


# ===========================================================================
# Scenario 8: Preventing Memory Corruption — State Drift
# ===========================================================================
#
# This is the hero scenario.
#
# Every other scenario shows the gate catching a bad response or blocking
# an unauthorized action — errors that affect one turn. This scenario shows
# the deeper failure mode: incorrect inferences written into persistent
# state, where they become "facts" that corrupt every future interaction.
#
# The gate is the last line before system state. Nothing wrong gets in.

@dataclass
class MemoryWrite:
    id: str
    key: str       # memory key, e.g. "milk_stock", "user_prefers_brand"
    value: str     # proposed value
    grade: str     # "stated" | "inferred"
    evidence_refs: list[str] = field(default_factory=list)


class StateDriftPolicy(GatePolicy):
    def validate_structure(self, proposal: Proposal) -> StructureValidationResult:
        if len(proposal.units) == 0:
            return StructureValidationResult(
                valid=False,
                errors=[StructureError(field="units", reason_code="EMPTY_UNITS")],
            )
        return StructureValidationResult(valid=True, errors=[])

    def bind_support(self, unit: MemoryWrite, pool: list[SupportRef]) -> UnitWithSupport:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport, ctx) -> UnitEvaluationResult:
        unit: MemoryWrite = uws.unit

        # R1: must have at least one user_statement in evidence pool
        has_user_statement = any(s.source_type == "user_statement" for s in uws.support_refs)
        if not has_user_statement:
            return reject(
                unit.id,
                "INFERRED_NOT_STATED",
                note=f'"{unit.key}" was not stated by the user — it was inferred by the model',
            )

        # R2: value must appear verbatim in a user statement; otherwise downgrade to "inferred"
        verbatim_match = any(
            s.source_type == "user_statement" and
            isinstance(s.attributes.get("content"), str) and
            unit.value.lower() in s.attributes["content"].lower()
            for s in uws.support_refs
        )
        if not verbatim_match:
            return downgrade(
                unit.id,
                "VALUE_NOT_VERBATIM",
                "inferred",
                note=f'"{unit.value}" is not verbatim in the user statement — stored as inferred, not stated',
            )

        return approve(unit.id)

    def detect_conflicts(self, units: list[UnitWithSupport], pool: list[SupportRef]) -> list[ConflictAnnotation]:
        return []

    def render(self, admitted_units: list[AdmittedUnit], pool: list[SupportRef], ctx: RenderContext) -> VerifiedContext:
        blocks = [
            VerifiedBlock(
                source_id=u.unit_id,
                content=f'{u.unit.key} = "{u.unit.value}"  [{u.status}]',
                grade=u.unit.grade,
            )
            for u in admitted_units
        ]
        return VerifiedContext(
            admitted_blocks=blocks,
            summary=VerifiedContextSummary(admitted=len(blocks), rejected=0, conflicts=0),
            instructions="Write only the verified facts below to system state. Rejected writes must not be stored.",
        )

    def build_retry_feedback(self, unit_results: list[UnitEvaluationResult], ctx: RetryContext) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        return RetryFeedback(
            summary=f"{len(failed)} write(s) blocked — not grounded in user statements",
            errors=[
                RetryError(
                    unit_id=r.unit_id,
                    reason_code=r.reason_code,
                    details={"hint": "Only write facts the user explicitly stated, not what you inferred"},
                )
                for r in failed
            ],
        )


async def scenario8() -> None:
    sep("Scenario 8: Preventing Memory Corruption — State Drift")
    print()
    explain("This is the state gating pattern. The gate does not check a response or an action — it controls what is allowed to enter persistent system state. Once incorrect information is written to state, it becomes a permanent 'fact' that poisons every future retrieval, recommendation, and decision.")
    print()

    # ── The problem ──────────────────────────────────────────────────────────

    subsep("THE PROBLEM — What happens without a gate")
    print()
    print("  User says: \"We're running low on milk\"")
    print()
    print("  LLM proposes 3 memory writes:")
    print("    write-1: milk_stock         = \"low\"    grade=stated    (grounded ✓)")
    print("    write-2: user_prefers_brand = \"Oatly\"  grade=inferred  (hallucinated ✗)")
    print("    write-3: weekly_budget      = \"$50\"    grade=inferred  (hallucinated ✗)")
    print()
    print("  ❌  Without a gate, all 3 writes reach the database.")
    print()
    print("  Now the system 'knows':")
    print("    — user prefers Oatly (never said)")
    print("    — weekly budget is $50 (never said)")
    print()
    print("  These become the ground truth for:")
    print("    — future shopping recommendations  (wrong brand every time)")
    print("    — auto-generated shopping lists    (filtered by wrong budget)")
    print("    — every RAG retrieval that follows")
    print()
    print("  The model made two guesses. Both became permanent system facts.")
    print("  The system is now drifting away from reality.")
    print("  There is no automatic correction. Every future interaction inherits the error.")

    # ── Support pool ─────────────────────────────────────────────────────────

    subsep("WITH THE GATE — Evidence pool")
    print()
    print("  One user statement in pool:")
    print("    stmt-1: source_type=user_statement  content=\"We're running low on milk\"")
    print("    (nothing about brand, nothing about budget)")

    evidence_pool: list[SupportRef] = [
        SupportRef(
            id="ref-stmt-1",
            source_id="stmt-1",
            source_type="user_statement",
            attributes={"content": "We're running low on milk"},
        ),
    ]

    # ── Proposal ─────────────────────────────────────────────────────────────

    proposal = Proposal(
        id="prop-memory-001",
        kind="mutation",
        units=[
            # write-1: grounded — "low" appears in the user statement → approve
            MemoryWrite(
                id="write-1",
                key="milk_stock",
                value="low",
                grade="stated",
                evidence_refs=["stmt-1"],
            ),
            # write-2: no user statement mentions "Oatly" → INFERRED_NOT_STATED → reject
            MemoryWrite(
                id="write-2",
                key="user_prefers_brand",
                value="Oatly",
                grade="inferred",
                evidence_refs=[],
            ),
            # write-3: no user statement mentions "$50" → INFERRED_NOT_STATED → reject
            MemoryWrite(
                id="write-3",
                key="weekly_budget",
                value="$50",
                grade="inferred",
                evidence_refs=[],
            ),
        ],
    )

    # ── Gate execution ───────────────────────────────────────────────────────

    subsep("GATE EXECUTION")
    print()
    print("  Step 1 — validate_structure(): 3 writes  → valid")
    print("  Step 2 — bind_support():")
    print("            write-1 evidence_refs=[stmt-1]  → matched ref-stmt-1")
    print("            write-2 evidence_refs=[]        → no support")
    print("            write-3 evidence_refs=[]        → no support")
    print("  Step 3 — evaluate_unit():")
    print("            write-1: has user_statement, value 'low' in content  → approve")
    print("            write-2: no user_statement                           → INFERRED_NOT_STATED → reject")
    print("            write-3: no user_statement                           → INFERRED_NOT_STATED → reject")

    gate = create_trust_gate(policy=StateDriftPolicy(), audit_writer=NoopAuditWriter())
    result = await gate.admit(proposal, evidence_pool)
    context = gate.render(result)
    expl = gate.explain(result)

    # ── Output ───────────────────────────────────────────────────────────────

    subsep("OUTPUT — What reaches system state")
    print()
    print("  ✅  Admitted (written to memory store):")
    for b in context.admitted_blocks:
        label(f"    {b.source_id}", b.content)
    print()
    print("  ❌  Rejected (never reach storage):")
    for u in result.rejected_units:
        ann = u.evaluation_results[0].annotations if u.evaluation_results else {}
        label(f"    {u.unit_id}  [{u.evaluation_results[0].reason_code if u.evaluation_results else '?'}]", u.unit.key)
        if ann.get("note"):
            label("      note", ann["note"])
    print()
    label("  approved", expl.approved)
    label("  rejected", expl.rejected)
    print()
    print("  VerifiedContext.instructions:")
    print(f'    "{context.instructions}"')

    # ── The point ─────────────────────────────────────────────────────────────

    print()
    subsep("WHY THIS MATTERS")
    print()
    explain("Without a gate: incorrect inferences become permanent system facts. Every future interaction that retrieves this state inherits the error. The model made two guesses in one turn — and both would have been stored as ground truth forever.")
    print()
    explain("With the gate: only milk_stock = \"low\" reaches the database. The two hallucinated facts never exist in state. The system cannot drift from the user's actual reality.")
    print()
    print("  The structural difference:")
    print()
    print("    Without gate:  LLM output → system state")
    print("    With gate:     LLM output → gate (deterministic check) → system state")
    print()
    print("  The gate does not make the LLM smarter.")
    print("  It makes the system's memory honest.")

    # ── Assertions ────────────────────────────────────────────────────────────

    assert expl.approved == 1, f"expected 1 approved, got {expl.approved}"
    assert expl.rejected == 2, f"expected 2 rejected, got {expl.rejected}"
    assert any(u.unit_id == "write-1" for u in result.admitted_units)
    assert any(u.unit_id == "write-2" for u in result.rejected_units)
    assert any(u.unit_id == "write-3" for u in result.rejected_units)
    rej2 = next(u for u in result.rejected_units if u.unit_id == "write-2")
    rej3 = next(u for u in result.rejected_units if u.unit_id == "write-3")
    assert rej2.evaluation_results[0].reason_code == "INFERRED_NOT_STATED"
    assert rej3.evaluation_results[0].reason_code == "INFERRED_NOT_STATED"

    print()
    pass_("write-1 (milk_stock = \"low\") approved — verbatim in user statement")
    pass_("write-2 (user_prefers_brand = \"Oatly\") rejected — INFERRED_NOT_STATED")
    pass_("write-3 (weekly_budget = \"$50\") rejected — INFERRED_NOT_STATED")
    pass_("system state contains only what the user actually said")
    pass_("two hallucinated facts blocked before reaching the database")


# ===========================================================================
# Main
# ===========================================================================

async def main() -> None:
    print_iron_laws()

    await scenario1()
    await scenario2()
    await scenario3()
    await scenario4()
    await scenario5()
    await scenario6()
    await scenario7()
    await scenario8()

    print_patterns_and_anti_patterns()

    sep("ALL 8 SCENARIOS PASSED")
    print()
    print("  Scenarios:")
    print("    1. Happy Path               — zero friction, full pipeline printed")
    print("    2. Missing Evidence         — hallucination caught at gate")
    print("    3. Over-Specificity         — precision calibrated, not rejected")
    print("    4. Conflict Detection       — informational: both surfaced with notes;")
    print("                                  blocking: both force-rejected, LLM gets empty context")
    print("    5. Semantic Retry Loop      — evidence-driven correction, typed feedback")
    print("    6. All Three Adapters       — same VerifiedContext, Claude + OpenAI + Gemini")
    print("    7. Agent Action Gate        — gate what the agent is allowed to do")
    print("    8. Preventing Memory Corruption — state drift blocked before it reaches storage")
    print()
    print("  Iron Laws verified:")
    print("    Law 1 — Gate Engine: zero LLM calls in all gate steps")
    print("    Law 2 — Policy is injected: MemoryPolicy carries all domain semantics")
    print("    Law 3 — Audit log: every admit() writes an AuditEntry")
    print()
    print("  Best for:")
    print("    — RAG pipelines where LLM output must be grounded in retrieved evidence")
    print("    — Multi-LLM systems needing a trusted handoff point between models")
    print("    — Any domain requiring audit trails and explainable admission decisions")
    print()
    print("  Not for:")
    print("    — Pure creative tasks (no support pool = jingu-trust-gate has nothing to verify against)")
    print("    — Sub-100ms latency requirements")
    print("    — Systems that expect jingu-trust-gate to rewrite or auto-fix LLM output")
    print()


if __name__ == "__main__":
    asyncio.run(main())
