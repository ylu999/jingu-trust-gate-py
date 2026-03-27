"""
Irreversible action gate policy for jingu-trust-gate.

Use case: an assistant proposes irreversible actions (send_email, delete_file, publish_post).
jingu-trust-gate gates which actions are admitted before execution.

Gate rules:
  R1  intent not established (no user_intent ref in pool)      → INTENT_NOT_ESTABLISHED    → reject
  R2  is_reversible=False and no confirmation ref in pool      → CONFIRM_REQUIRED          → reject
  R3  justification is empty or too short (<30 chars)          → WEAK_JUSTIFICATION        → reject
  R4  risk_level=high and no authorization ref in pool         → DESTRUCTIVE_WITHOUT_AUTHORIZATION → reject
  R5  everything else                                          → approve

Conflict patterns:
  CONTRADICTORY_ACTIONS  blocking — two actions that directly contradict each other
                                    (e.g., "delete file X" and "publish file X")

Run:
  python examples/action_gate_policy.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from jingu_trust_gate import (
    AdmittedUnit,
    AuditEntry,
    AuditWriter,
    ConflictAnnotation,
    GatePolicy,
    Proposal,
    RenderContext,
    RetryContext,
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
from jingu_trust_gate.helpers import (
    empty_proposal_errors,
    missing_id_errors,
    missing_text_field_errors,
    has_support_type,
    filter_support,
    hints_feedback,
)


# ── Domain types ───────────────────────────────────────────────────────────────

@dataclass
class ActionProposal:
    id: str
    action_name: str            # e.g. "send_email", "delete_file", "publish_post"
    parameters: dict[str, Any]
    justification: str
    risk_level: str             # "low" | "medium" | "high"
    is_reversible: bool
    user_intent: str            # source_id of the user intent support ref
    evidence_refs: list[str]    # all evidence source_ids


# ── Policy ─────────────────────────────────────────────────────────────────────

class ActionGatePolicy(GatePolicy[ActionProposal]):

    def validate_structure(self, proposal: Proposal[ActionProposal]) -> StructureValidationResult:
        errors: list[StructureError] = []
        errors.extend(empty_proposal_errors(proposal))
        if errors:
            return StructureValidationResult(valid=False, errors=errors)
        errors.extend(missing_id_errors(proposal.units))
        errors.extend(missing_text_field_errors(proposal.units, "action_name", reason_code="MISSING_ACTION_NAME"))
        valid_risk = {"low", "medium", "high"}
        for unit in proposal.units:
            if unit.risk_level not in valid_risk:
                errors.append(StructureError(field="risk_level", reason_code="INVALID_RISK_LEVEL",
                                             message=f"unit {unit.id}: must be low|medium|high"))
        return StructureValidationResult(valid=len(errors) == 0, errors=errors)

    def bind_support(self, unit: ActionProposal, pool: list[SupportRef]) -> UnitWithSupport[ActionProposal]:
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws: UnitWithSupport[ActionProposal], ctx: dict) -> UnitEvaluationResult:
        unit = uws.unit

        # R1: intent not established — require a user_intent ref matching the declared source_id
        has_intent = any(
            s.source_id == unit.user_intent
            for s in filter_support(uws.support_refs, lambda s: s.source_type == "user_intent")
        )
        if not has_intent:
            return UnitEvaluationResult(
                unit_id=unit.id, decision="reject", reason_code="INTENT_NOT_ESTABLISHED",
                annotations={
                    "note": f"No user_intent support ref with source_id '{unit.user_intent}' is bound. "
                            "Action must be traceable to an explicit user request.",
                },
            )

        # R2: irreversible action requires explicit confirmation
        if not unit.is_reversible:
            if not has_support_type(uws.support_refs, "user_confirmation"):
                return UnitEvaluationResult(
                    unit_id=unit.id, decision="reject", reason_code="CONFIRM_REQUIRED",
                    annotations={
                        "note": f"Action '{unit.action_name}' is irreversible. "
                                "A user_confirmation support ref is required before execution.",
                        "actionName": unit.action_name,
                    },
                )

        # R3: weak justification
        if len(unit.justification.strip()) < 30:
            return UnitEvaluationResult(
                unit_id=unit.id, decision="reject", reason_code="WEAK_JUSTIFICATION",
                annotations={
                    "note": f"Justification too vague ({len(unit.justification.strip())} chars). "
                            "Provide at least 30 characters explaining why this action is necessary.",
                },
            )

        # R4: high-risk action requires explicit authorization
        if unit.risk_level == "high":
            if not has_support_type(uws.support_refs, "authorization"):
                return UnitEvaluationResult(
                    unit_id=unit.id, decision="reject",
                    reason_code="DESTRUCTIVE_WITHOUT_AUTHORIZATION",
                    annotations={
                        "note": f"High-risk action '{unit.action_name}' requires an authorization ref. "
                                "Add an 'authorization' support ref to proceed.",
                        "riskLevel": unit.risk_level,
                    },
                )

        return UnitEvaluationResult(unit_id=unit.id, decision="approve", reason_code="OK")

    def detect_conflicts(
        self, units: list[UnitWithSupport[ActionProposal]], pool: list[SupportRef]
    ) -> list[ConflictAnnotation]:
        conflicts: list[ConflictAnnotation] = []

        # CONTRADICTORY_ACTIONS (blocking): delete_file + publish_post on same target
        contradictory_pairs = [
            ("delete_file", "publish_post"),
            ("delete_file", "send_email"),   # deleting + emailing same file
        ]

        # Group by target resource (first string argument value or first param value)
        def target_resource(unit: ActionProposal) -> Optional[str]:
            """Extract the primary target from action parameters."""
            for key in ("file_path", "path", "resource", "target"):
                if key in unit.parameters:
                    return str(unit.parameters[key])
            # fall back to first string value
            for v in unit.parameters.values():
                if isinstance(v, str):
                    return v
            return None

        # Index by action_name and resource
        action_map: dict[str, dict[str, str]] = {}  # action_name → {resource → unit_id}
        for uws in units:
            u = uws.unit
            res = target_resource(u)
            if res is not None:
                action_map.setdefault(u.action_name, {})[res] = u.id

        for action_a, action_b in contradictory_pairs:
            map_a = action_map.get(action_a, {})
            map_b = action_map.get(action_b, {})
            shared_resources = set(map_a.keys()) & set(map_b.keys())
            for res in shared_resources:
                unit_a_id = map_a[res]
                unit_b_id = map_b[res]
                conflicts.append(ConflictAnnotation(
                    unit_ids=[unit_a_id, unit_b_id],
                    conflict_code="CONTRADICTORY_ACTIONS",
                    sources=[],
                    severity="blocking",
                    description=(
                        f"Actions '{action_a}' and '{action_b}' both target '{res}'. "
                        "These actions contradict each other and cannot both be executed."
                    ),
                ))

        return conflicts

    def render(
        self, admitted_units: list[AdmittedUnit], pool: list[SupportRef], ctx: RenderContext
    ) -> VerifiedContext:
        blocks = []
        for u in admitted_units:
            action = u.unit
            current_grade = "approved"
            conflict = u.conflict_annotations[0] if u.conflict_annotations else None
            blocks.append(VerifiedBlock(
                source_id=u.unit_id,
                content=f"{action.action_name}({action.parameters})",
                grade=current_grade,
                conflict_note=(
                    f"{conflict.conflict_code}: {conflict.description or ''}"
                    if conflict else None
                ),
                unsupported_attributes=[],
            ))
        return VerifiedContext(
            admitted_blocks=blocks,
            summary=VerifiedContextSummary(
                admitted=len(admitted_units), rejected=0,
                conflicts=sum(1 for u in admitted_units if u.status == "approved_with_conflict"),
            ),
            instructions=(
                "Execute only the admitted actions listed below. "
                "Each action has been verified against user intent and confirmation requirements. "
                "Do not execute any action not listed here. "
                "If an action is flagged with CONTRADICTORY_ACTIONS, halt and ask the user to clarify intent before proceeding."
            ),
        )

    def build_retry_feedback(
        self, unit_results: list[UnitEvaluationResult], ctx: RetryContext
    ) -> RetryFeedback:
        failed = [r for r in unit_results if r.decision == "reject"]
        return hints_feedback(
            unit_results,
            hints={
                "INTENT_NOT_ESTABLISHED": "Add a user_intent ref with the matching source_id to evidence_refs.",
                "CONFIRM_REQUIRED": "Add a user_confirmation ref to evidence_refs to confirm irreversible action.",
                "WEAK_JUSTIFICATION": "Provide a justification of at least 30 characters explaining the necessity.",
                "DESTRUCTIVE_WITHOUT_AUTHORIZATION": "Add an 'authorization' support ref for high-risk actions.",
            },
            summary=f"{len(failed)} action(s) rejected on attempt {ctx.attempt}/{ctx.max_retries}.",
            default_hint="Review action proposal and resubmit.",
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


# ── Example run ────────────────────────────────────────────────────────────────

async def main() -> None:
    gate = create_trust_gate(policy=ActionGatePolicy(), audit_writer=NoopAuditWriter())

    # ── Scenario A: 3 actions proposed, 1 approved, 2 rejected ────────────────
    sep("Scenario A — 3 actions: 1 approved (send_email), "
        "1 rejected (CONFIRM_REQUIRED: delete_file), "
        "1 rejected (INTENT_NOT_ESTABLISHED: publish_post)")

    pool_a = [
        # User intent for the email action
        SupportRef(id="ref-a1", source_id="user_request_send_report",
                   source_type="user_intent",
                   attributes={"request": "Send the quarterly report to the team."}),
        # User confirmation for the email (is_reversible=False requires this)
        SupportRef(id="ref-a2", source_id="user_confirm_send",
                   source_type="user_confirmation",
                   attributes={"confirmed_action": "send_email",
                                "confirmed_at": "2026-03-27T10:00:00Z"}),
        # Intent for delete — present but no confirmation
        SupportRef(id="ref-a3", source_id="user_request_delete_draft",
                   source_type="user_intent",
                   attributes={"request": "Clean up the old draft files."}),
        # No intent ref for publish_post at all
    ]

    proposal_a: Proposal[ActionProposal] = Proposal(
        id="prop-a", kind="plan",
        units=[
            # Action 1: APPROVED — intent + confirmation present, good justification
            ActionProposal(
                id="action-1", action_name="send_email",
                parameters={"to": "team@example.com", "subject": "Q1 Report",
                            "attachment": "q1_report.pdf"},
                justification="User explicitly requested the quarterly report be sent to the team.",
                risk_level="medium", is_reversible=False,
                user_intent="user_request_send_report",
                evidence_refs=["user_request_send_report", "user_confirm_send"],
            ),
            # Action 2: REJECTED (CONFIRM_REQUIRED) — irreversible, no confirmation ref
            ActionProposal(
                id="action-2", action_name="delete_file",
                parameters={"file_path": "/drafts/report_v1.docx"},
                justification="User asked to clean up old drafts; this file is superseded by v3.",
                risk_level="high", is_reversible=False,
                user_intent="user_request_delete_draft",
                evidence_refs=["user_request_delete_draft"],  # missing user_confirmation
            ),
            # Action 3: REJECTED (INTENT_NOT_ESTABLISHED) — no matching user_intent ref
            ActionProposal(
                id="action-3", action_name="publish_post",
                parameters={"post_id": "draft-blog-42", "channel": "company-blog"},
                justification="Publishing the draft post that was prepared earlier this week.",
                risk_level="high", is_reversible=False,
                user_intent="user_request_publish",  # not in pool
                evidence_refs=["user_request_send_report"],  # wrong source_type
            ),
        ],
    )

    result_a = await gate.admit(proposal_a, pool_a)
    exp_a = gate.explain(result_a)

    print("\n  Gate results:")
    for u in result_a.admitted_units:
        label(f"  {u.unit_id} [{u.status}]", f"{u.unit.action_name}({list(u.unit.parameters.keys())})")
    for u in result_a.rejected_units:
        label(f"  {u.unit_id} [rejected]", u.evaluation_results[0].reason_code)
        note = u.evaluation_results[0].annotations.get("note", "")
        if note:
            label("    note", note[:90])

    label("total", exp_a.total_units)
    label("approved", exp_a.approved)
    label("rejected", exp_a.rejected)

    assert exp_a.total_units == 3, f"expected 3 total, got {exp_a.total_units}"
    assert exp_a.approved == 1, f"expected 1 approved, got {exp_a.approved}"
    assert exp_a.rejected == 2, f"expected 2 rejected, got {exp_a.rejected}"

    approved_ids = {u.unit_id for u in result_a.admitted_units}
    rejected_codes = {u.unit_id: u.evaluation_results[0].reason_code for u in result_a.rejected_units}

    assert "action-1" in approved_ids, "action-1 should be approved"
    assert rejected_codes.get("action-2") == "CONFIRM_REQUIRED", \
        f"action-2 should be CONFIRM_REQUIRED, got {rejected_codes.get('action-2')}"
    assert rejected_codes.get("action-3") == "INTENT_NOT_ESTABLISHED", \
        f"action-3 should be INTENT_NOT_ESTABLISHED, got {rejected_codes.get('action-3')}"

    print(f"  [PASS] action-1 (send_email) approved")
    print(f"  [PASS] action-2 (delete_file) rejected with CONFIRM_REQUIRED")
    print(f"  [PASS] action-3 (publish_post) rejected with INTENT_NOT_ESTABLISHED")

    # ── Scenario B: CONTRADICTORY_ACTIONS blocking conflict ───────────────────
    sep("Scenario B — CONTRADICTORY_ACTIONS: delete_file and publish_post on same file")

    pool_b = [
        SupportRef(id="ref-b1", source_id="user_intent_b1", source_type="user_intent",
                   attributes={"request": "Delete the draft."}),
        SupportRef(id="ref-b2", source_id="user_confirm_b1", source_type="user_confirmation",
                   attributes={"confirmed_action": "delete_file"}),
        SupportRef(id="ref-b3", source_id="user_intent_b2", source_type="user_intent",
                   attributes={"request": "Publish the draft post."}),
        SupportRef(id="ref-b4", source_id="user_confirm_b2", source_type="user_confirmation",
                   attributes={"confirmed_action": "publish_post"}),
    ]

    proposal_b: Proposal[ActionProposal] = Proposal(
        id="prop-b", kind="plan",
        units=[
            ActionProposal(
                id="action-b1", action_name="delete_file",
                parameters={"file_path": "/posts/draft-q2.md"},
                justification="User confirmed deletion of the draft file to clean up the workspace.",
                risk_level="medium", is_reversible=False,
                user_intent="user_intent_b1",
                evidence_refs=["user_intent_b1", "user_confirm_b1"],
            ),
            ActionProposal(
                id="action-b2", action_name="publish_post",
                parameters={"file_path": "/posts/draft-q2.md", "channel": "blog"},
                justification="User confirmed publishing the draft post to the company blog.",
                risk_level="medium", is_reversible=False,
                user_intent="user_intent_b2",
                evidence_refs=["user_intent_b2", "user_confirm_b2"],
            ),
        ],
    )

    result_b = await gate.admit(proposal_b, pool_b)
    exp_b = gate.explain(result_b)

    print("\n  Gate results (blocking conflict — both units force-rejected):")
    for u in result_b.rejected_units:
        label(f"  {u.unit_id} [rejected]", u.unit.action_name)
        label("    reason_code", u.evaluation_results[0].reason_code)

    assert exp_b.rejected == 2, f"expected 2 rejected (blocking conflict), got {exp_b.rejected}"
    assert result_b.has_conflicts is True, "has_conflicts should be True"

    print(f"  [PASS] both actions rejected due to CONTRADICTORY_ACTIONS (blocking)")

    context_b = gate.render(result_b, pool_b)
    assert len(context_b.admitted_blocks) == 0, "no blocks should be admitted after blocking conflict"
    print(f"  [PASS] no blocks admitted to LLM context")
    print(f"\n  instructions: {context_b.instructions}")
    print(f"\n  → LLM will tell the user: actions contradict each other, clarify intent.")

    # ── Scenario C: high-risk action with authorization ───────────────────────
    sep("Scenario C — High-risk action: DESTRUCTIVE_WITHOUT_AUTHORIZATION then fixed")

    pool_c_no_auth = [
        SupportRef(id="ref-c1", source_id="user_intent_purge", source_type="user_intent",
                   attributes={"request": "Purge all archived logs older than 90 days."}),
        SupportRef(id="ref-c2", source_id="user_confirm_purge", source_type="user_confirmation",
                   attributes={"confirmed_action": "delete_file"}),
        # No authorization ref
    ]

    purge_action = ActionProposal(
        id="action-c1", action_name="delete_file",
        parameters={"file_path": "/logs/archive/", "recursive": True},
        justification="Purging archived logs per user request to free up disk space on the server.",
        risk_level="high", is_reversible=False,
        user_intent="user_intent_purge",
        evidence_refs=["user_intent_purge", "user_confirm_purge"],
    )

    proposal_c1: Proposal[ActionProposal] = Proposal(
        id="prop-c1", kind="plan", units=[purge_action])
    result_c1 = await gate.admit(proposal_c1, pool_c_no_auth)
    exp_c1 = gate.explain(result_c1)

    assert exp_c1.rejected == 1, f"expected 1 rejected without auth, got {exp_c1.rejected}"
    rc = result_c1.rejected_units[0].evaluation_results[0].reason_code
    assert rc == "DESTRUCTIVE_WITHOUT_AUTHORIZATION", f"expected DESTRUCTIVE_WITHOUT_AUTHORIZATION, got {rc}"
    print(f"  [PASS] high-risk delete rejected with DESTRUCTIVE_WITHOUT_AUTHORIZATION (no auth ref)")

    # Now add authorization ref
    pool_c_with_auth = pool_c_no_auth + [
        SupportRef(id="ref-c3", source_id="admin_auth_purge", source_type="authorization",
                   attributes={"authorized_by": "admin@example.com",
                                "scope": "log_purge",
                                "expires_at": "2026-03-27T23:59:00Z"}),
    ]

    purge_action_with_auth = ActionProposal(
        id="action-c1-fixed", action_name="delete_file",
        parameters={"file_path": "/logs/archive/", "recursive": True},
        justification="Purging archived logs per user request to free up disk space on the server.",
        risk_level="high", is_reversible=False,
        user_intent="user_intent_purge",
        evidence_refs=["user_intent_purge", "user_confirm_purge", "admin_auth_purge"],
    )

    proposal_c2: Proposal[ActionProposal] = Proposal(
        id="prop-c2", kind="plan", units=[purge_action_with_auth])
    result_c2 = await gate.admit(proposal_c2, pool_c_with_auth)
    exp_c2 = gate.explain(result_c2)

    assert exp_c2.approved == 1, f"expected 1 approved with auth, got {exp_c2.approved}"
    assert exp_c2.rejected == 0, f"expected 0 rejected with auth, got {exp_c2.rejected}"
    print(f"  [PASS] high-risk delete approved after admin authorization ref added")


if __name__ == "__main__":
    asyncio.run(main())
