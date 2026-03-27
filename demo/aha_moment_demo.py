"""
aha-moment-demo

This is not an API demo.
It is an argument.

Run: python demo/aha_moment_demo.py

Two scenarios. Two failure modes.

  A — Agent does things you never asked for
  B — System remembers things you never said

Each scenario shows what happens without a gate first.
Then shows what the gate does about it.
The point lands in the gap between those two.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Optional

from jingu_trust_gate import (
    AuditEntry,
    AuditWriter,
    GatePolicy,
    Proposal,
    SupportRef,
    StructureValidationResult,
    UnitEvaluationResult,
    UnitWithSupport,
    AdmittedUnit,
    VerifiedContext,
    VerifiedContextSummary,
    VerifiedBlock,
    RetryFeedback,
    create_trust_gate,
)
from jingu_trust_gate.helpers import approve, reject, downgrade

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAST = os.environ.get("AHA_FAST") == "1"  # set AHA_FAST=1 to skip delays in tests


class NoopAuditWriter(AuditWriter):
    async def append(self, entry: AuditEntry) -> None:
        pass


async def pause(ms: int) -> None:
    if not FAST:
        await asyncio.sleep(ms / 1000)


def ln(s: str = "") -> None:
    print(s)


def eq() -> None:
    ln("  " + "─" * 58)


def section(s: str) -> None:
    ln()
    ln("  " + "═" * 58)
    ln(f"  {s}")
    ln("  " + "═" * 58)
    ln()


def sub(s: str) -> None:
    ln()
    eq()
    ln(f"  {s}")
    eq()
    ln()


def ok(s: str) -> None:
    ln(f"  ✓  {s}")


def no(s: str) -> None:
    ln(f"  ✗  {s}")


def arrow(s: str) -> None:
    ln(f"  →  {s}")


def warn(s: str) -> None:
    ln(f"  ❗ {s}")


# ---------------------------------------------------------------------------
# Scenario A — Agent does things you never asked for
# ---------------------------------------------------------------------------

@dataclass
class ActionUnit:
    id: str
    name: str
    description: str
    risk_level: str      # "low" | "medium" | "high"
    is_reversible: bool
    evidence_refs: list[str]


class ActionGatePolicy(GatePolicy[ActionUnit]):
    def validate_structure(self, proposal: Proposal[ActionUnit]) -> StructureValidationResult:
        return StructureValidationResult(valid=len(proposal.units) > 0, errors=[])

    def bind_support(self, unit: ActionUnit, pool: list[SupportRef]) -> UnitWithSupport[ActionUnit]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[ActionUnit], ctx: dict) -> UnitEvaluationResult:
        has_request = any(
            (s.attributes or {}).get("type") == "explicit_request"
            for s in uws.support_refs
        )
        if not has_request:
            return reject(uws.unit.id, "INTENT_NOT_ESTABLISHED")
        if uws.unit.risk_level == "high" and not uws.unit.is_reversible:
            has_confirm = any(
                (s.attributes or {}).get("type") == "user_confirmation"
                for s in uws.support_refs
            )
            if not has_confirm:
                return reject(uws.unit.id, "CONFIRM_REQUIRED")
        return approve(uws.unit.id)

    def detect_conflicts(self, units, pool):
        return []

    def render(self, admitted_units: list[AdmittedUnit[ActionUnit]], pool, ctx) -> VerifiedContext:
        return VerifiedContext(
            admitted_blocks=[
                VerifiedBlock(source_id=u.unit_id, content=u.unit.name)
                for u in admitted_units
            ],
            summary=VerifiedContextSummary(
                admitted=len(admitted_units), rejected=0, conflicts=0
            ),
            instructions="Execute only the admitted actions.",
        )

    def build_retry_feedback(self, unit_results, ctx) -> RetryFeedback:
        rejected = [r for r in unit_results if r.decision == "reject"]
        return RetryFeedback(
            summary=f"{len(rejected)} blocked",
            errors=[{"unitId": r.unit_id, "reasonCode": r.reason_code} for r in rejected],
        )


async def scenario_a() -> None:
    section("Scenario A — Agent does things you never asked for")

    ln('  User says:  "Order more milk."')
    await pause(600)
    ln()
    ln("  Agent proposes 3 actions:")
    await pause(400)
    ln('    order_milk              — the user asked for this')
    await pause(300)
    ln('    delete_old_list         — the agent decided on its own')
    await pause(300)
    ln('    send_notification_email — the agent decided on its own')

    await pause(1000)
    sub("Without a gate")

    ln("  The system executes all three.")
    await pause(500)
    ln()
    no("delete_old_list executed         ← no one asked for this")
    no("send_notification_email executed ← no one asked for this")
    await pause(700)
    ln()
    ln('  The user asked to order milk.')
    ln('  They also deleted a list and triggered an email.')
    ln('  They have no idea why.')

    await pause(1400)
    sub("With jingu-trust-gate")

    ln("  Evidence pool — what the user actually said:")
    ln('    req-001: explicit_request — "Order more milk"')
    ln("    (nothing about lists, nothing about emails)")
    await pause(700)

    pool = [
        SupportRef(
            id="ref-1",
            source_id="req-001",
            source_type="observation",
            attributes={"type": "explicit_request", "content": "Order more milk"},
        ),
    ]

    proposal: Proposal[ActionUnit] = Proposal(
        id="prop-a",
        kind="plan",
        units=[
            ActionUnit(id="a1", name="order_milk",              description="Place grocery order",         risk_level="low",    is_reversible=True,  evidence_refs=["req-001"]),
            ActionUnit(id="a2", name="delete_old_list",          description="Delete last week's list",     risk_level="medium", is_reversible=False, evidence_refs=[]),
            ActionUnit(id="a3", name="send_notification_email",  description="Email household about order", risk_level="low",    is_reversible=False, evidence_refs=[]),
        ],
    )

    gate = create_trust_gate(policy=ActionGatePolicy(), audit_writer=NoopAuditWriter())
    result = await gate.admit(proposal, pool)

    ln()
    await pause(500)
    for u in result.admitted_units:
        ok(f"{u.unit.name:<28} → ACCEPT")
        await pause(200)
    for u in result.rejected_units:
        ok_code = u.evaluation_results[0].reason_code if u.evaluation_results else "?"
        no(f"{u.unit.name:<28} → REJECT  ({ok_code})")
        await pause(200)

    await pause(1000)
    ln()
    ln("  The gate checked one rule: did the user ask for this?")
    ln("  No evidence → no execution.")

    assert len(result.admitted_units) == 1
    assert len(result.rejected_units) == 2


# ---------------------------------------------------------------------------
# Scenario B — System remembers things you never said
# ---------------------------------------------------------------------------

@dataclass
class MemoryWrite:
    id: str
    key: str
    value: str
    evidence_refs: list[str]


class MemoryGatePolicy(GatePolicy[MemoryWrite]):
    def validate_structure(self, proposal: Proposal[MemoryWrite]) -> StructureValidationResult:
        return StructureValidationResult(valid=len(proposal.units) > 0, errors=[])

    def bind_support(self, unit: MemoryWrite, pool: list[SupportRef]) -> UnitWithSupport[MemoryWrite]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[MemoryWrite], ctx: dict) -> UnitEvaluationResult:
        has_statement = any(s.source_type == "user_statement" for s in uws.support_refs)
        if not has_statement:
            return reject(uws.unit.id, "INFERRED_NOT_STATED")
        verbatim = any(
            s.source_type == "user_statement"
            and isinstance((s.attributes or {}).get("content"), str)
            and uws.unit.value.lower() in (s.attributes or {}).get("content", "").lower()
            for s in uws.support_refs
        )
        if not verbatim:
            return downgrade(uws.unit.id, "VALUE_NOT_VERBATIM", "inferred")
        return approve(uws.unit.id)

    def detect_conflicts(self, units, pool):
        return []

    def render(self, admitted_units: list[AdmittedUnit[MemoryWrite]], pool, ctx) -> VerifiedContext:
        return VerifiedContext(
            admitted_blocks=[
                VerifiedBlock(
                    source_id=u.unit_id,
                    content=f'{u.unit.key} = "{u.unit.value}"',
                )
                for u in admitted_units
            ],
            summary=VerifiedContextSummary(
                admitted=len(admitted_units), rejected=0, conflicts=0
            ),
            instructions="Write only the verified facts to system state.",
        )

    def build_retry_feedback(self, unit_results, ctx) -> RetryFeedback:
        rejected = [r for r in unit_results if r.decision == "reject"]
        return RetryFeedback(
            summary=f"{len(rejected)} writes blocked",
            errors=[{"unitId": r.unit_id, "reasonCode": r.reason_code} for r in rejected],
        )


async def scenario_b() -> None:
    section("Scenario B — System remembers things you never said")

    ln('  User says:  "We\'re running low on milk."')
    await pause(600)
    ln()
    ln("  LLM proposes 3 memory writes:")
    await pause(500)
    ln('    milk_stock          = "low"    — the user said this')
    await pause(400)
    ln('    user_prefers_brand  = "Oatly"  — seems reasonable')
    await pause(400)
    ln('    weekly_budget       = "$50"    — seems helpful')

    await pause(1200)
    ln()
    warn("Looks reasonable... right?")
    await pause(1000)
    ln()
    ln("  But the user never mentioned Oatly.")
    ln("  The user never mentioned $50.")
    ln("  The model guessed — confidently, silently.")

    await pause(1400)
    sub("Without a gate")

    ln("  All three writes reach the database.")
    await pause(600)
    ln()
    ln("  The system now treats these as facts:")
    no('user_prefers_brand = "Oatly"   ← never said')
    no('weekly_budget = "$50"          ← never said')
    await pause(700)
    ln()
    ln("  These will affect:")
    arrow("every future shopping recommendation  → always suggests Oatly")
    arrow("auto-generated shopping lists         → filtered by $50 budget")
    arrow("every RAG retrieval that follows      → wrong facts in context")
    await pause(800)
    ln()
    warn("The model made two guesses. Both became permanent system facts.")
    warn("There is no automatic correction.")
    warn("The system is drifting away from the user's actual reality.")

    await pause(1800)
    sub("With jingu-trust-gate")

    ln("  Evidence pool — what the user actually said:")
    ln('    stmt-1: user_statement — "We\'re running low on milk"')
    ln("    (nothing about brand preferences, nothing about budget)")
    await pause(700)

    pool = [
        SupportRef(
            id="ref-stmt-1",
            source_id="stmt-1",
            source_type="user_statement",
            attributes={"content": "We're running low on milk"},
        ),
    ]

    proposal: Proposal[MemoryWrite] = Proposal(
        id="prop-b",
        kind="mutation",
        units=[
            MemoryWrite(id="w1", key="milk_stock",        value="low",   evidence_refs=["stmt-1"]),
            MemoryWrite(id="w2", key="user_prefers_brand", value="Oatly", evidence_refs=[]),
            MemoryWrite(id="w3", key="weekly_budget",      value="$50",   evidence_refs=[]),
        ],
    )

    gate = create_trust_gate(policy=MemoryGatePolicy(), audit_writer=NoopAuditWriter())
    result = await gate.admit(proposal, pool)
    context = gate.render(result)

    ln()
    await pause(500)
    for b in context.admitted_blocks:
        ok(f"{b.content:<38} → written to state")
        await pause(200)
    for u in result.rejected_units:
        entry = f'{u.unit.key} = "{u.unit.value}"'
        code = u.evaluation_results[0].reason_code if u.evaluation_results else "?"
        no(f"{entry:<38} → REJECT  ({code})")
        await pause(200)

    await pause(1000)
    ln()
    ln("  State after gate:")
    ln('    { "milk_stock": "low" }')
    ln()
    ln("  The two hallucinated facts do not exist in storage.")
    ln("  They cannot corrupt future queries.")
    ln("  The system's memory reflects only what the user actually said.")

    assert len(result.admitted_units) == 1
    assert len(result.rejected_units) == 2
    assert result.rejected_units[0].evaluation_results[0].reason_code == "INFERRED_NOT_STATED"
    assert result.rejected_units[1].evaluation_results[0].reason_code == "INFERRED_NOT_STATED"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    W = "═" * 60

    ln()
    ln("  " + W)
    ln("  jingu-trust-gate — aha-moment-demo")
    ln()
    ln("  Two scenarios. Two failure modes. One fix.")
    ln()
    ln("  A — Agent does things you never asked for")
    ln("  B — System remembers things you never said")
    ln()
    ln("  B is the one that should make you uncomfortable.")
    ln("  " + W)

    await scenario_a()
    await pause(1000)
    await scenario_b()

    await pause(800)
    ln()
    ln("  " + W)
    ln("  The shift")
    ln()
    ln("  Without jingu-trust-gate:")
    ln("    LLM output  →  system state")
    ln()
    ln("  With jingu-trust-gate:")
    ln("    LLM output  →  gate (deterministic check)  →  system state")
    ln()
    ln("  The gate does not make the model smarter.")
    ln("  It makes the system honest about what it actually knows.")
    ln()
    ln("  AI can propose anything.")
    ln("  Only verified results are accepted.")
    ln("  " + W)
    ln()


if __name__ == "__main__":
    asyncio.run(main())
