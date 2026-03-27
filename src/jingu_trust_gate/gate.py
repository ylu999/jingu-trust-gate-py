"""
GateRunner — admission pipeline: validate → bind+evaluate → conflicts → admit.
"""

from __future__ import annotations

import uuid
from typing import Generic, Optional, TypeVar

from .audit import AuditWriter, build_audit_entry
from .policy import GatePolicy
from .types import (
    AdmissionResult,
    AdmittedUnit,
    ConflictAnnotation,
    Proposal,
    SupportRef,
    UnitEvaluationResult,
    UnitStatus,
    UnitWithSupport,
)

TUnit = TypeVar("TUnit")


class GateRunner(Generic[TUnit]):
    def __init__(
        self,
        policy: GatePolicy[TUnit],
        audit_writer: Optional[AuditWriter] = None,
    ) -> None:
        self._policy = policy
        self._audit_writer = audit_writer

    async def run(
        self,
        proposal: Proposal[TUnit],
        support_pool: list[SupportRef],
    ) -> AdmissionResult[TUnit]:
        audit_id = str(uuid.uuid4())
        proposal_context = {
            "proposal_id": proposal.id,
            "proposal_kind": proposal.kind,
        }

        # Step 1: Structure validation
        structure_result = self._policy.validate_structure(proposal)
        if not structure_result.valid:
            structure_rejected = [
                _build_admitted_unit(
                    unit=unit,
                    unit_id=_get_unit_id(unit, i),
                    eval_result=UnitEvaluationResult(
                        unit_id=_get_unit_id(unit, i),
                        decision="reject",
                        reason_code="STRUCTURE_INVALID",
                    ),
                    conflict_annotations=[],
                    support_ids=[],
                )
                for i, unit in enumerate(proposal.units)
            ]
            entry = build_audit_entry(
                audit_id=audit_id,
                proposal=proposal,
                all_units=structure_rejected,
                gate_results=[structure_result],
                unit_support_map={},
            )
            if self._audit_writer:
                await self._audit_writer.append(entry)
            return AdmissionResult(
                proposal_id=proposal.id,
                admitted_units=[],
                rejected_units=structure_rejected,
                has_conflicts=False,
                audit_id=audit_id,
                retry_attempts=1,
            )

        # Step 2: Bind support + evaluate each unit
        unit_support_map: dict[str, list[str]] = {}
        eval_triples: list[tuple[TUnit, UnitEvaluationResult, list[str]]] = []

        for unit in proposal.units:
            bound = self._policy.bind_support(unit, support_pool)
            eval_result = self._policy.evaluate_unit(bound, proposal_context)
            unit_support_map[eval_result.unit_id] = bound.support_ids
            eval_triples.append((unit, eval_result, bound.support_ids))

        # Step 3: Conflict detection
        units_with_support: list[UnitWithSupport[TUnit]] = [
            UnitWithSupport(
                unit=unit,
                support_ids=support_ids,
                support_refs=[s for s in support_pool if s.id in support_ids],
            )
            for unit, _, support_ids in eval_triples
        ]
        conflict_annotations = self._policy.detect_conflicts(
            units_with_support, support_pool
        )

        # Step 4: Build AdmittedUnit[] — blocking conflicts force-reject
        blocking_ids: set[str] = {
            uid
            for ann in conflict_annotations
            if ann.severity == "blocking"
            for uid in ann.unit_ids
        }

        all_units: list[AdmittedUnit] = []
        for unit, eval_result, support_ids in eval_triples:
            if eval_result.unit_id in blocking_ids and eval_result.decision != "reject":
                eval_result = UnitEvaluationResult(
                    unit_id=eval_result.unit_id,
                    decision="reject",
                    reason_code="BLOCKING_CONFLICT",
                )
            all_units.append(
                _build_admitted_unit(
                    unit=unit,
                    unit_id=eval_result.unit_id,
                    eval_result=eval_result,
                    conflict_annotations=conflict_annotations,
                    support_ids=support_ids,
                )
            )

        admitted = [u for u in all_units if u.status != "rejected"]
        rejected = [u for u in all_units if u.status == "rejected"]

        # Write audit (GateRunner internal — distinct from GatePolicy step numbering)
        all_gate_results = [
            structure_result,
            *[ev for _, ev, _ in eval_triples],
            {"kind": "conflict", "conflict_annotations": conflict_annotations},
        ]
        entry = build_audit_entry(
            audit_id=audit_id,
            proposal=proposal,
            all_units=all_units,
            gate_results=all_gate_results,
            unit_support_map=unit_support_map,
        )
        if self._audit_writer:
            await self._audit_writer.append(entry)

        return AdmissionResult(
            proposal_id=proposal.id,
            admitted_units=admitted,
            rejected_units=rejected,
            has_conflicts=len(conflict_annotations) > 0,
            audit_id=audit_id,
            retry_attempts=1,
        )


def _get_unit_id(unit: object, fallback_index: int) -> str:
    if isinstance(unit, dict):
        return str(unit.get("id", f"unit-{fallback_index}"))
    return str(getattr(unit, "id", f"unit-{fallback_index}"))


def _build_admitted_unit(
    unit: object,
    unit_id: str,
    eval_result: UnitEvaluationResult,
    conflict_annotations: list[ConflictAnnotation],
    support_ids: list[str],
) -> AdmittedUnit:
    relevant_conflicts = [
        ann for ann in conflict_annotations if unit_id in ann.unit_ids
    ]

    if eval_result.decision == "reject":
        status: UnitStatus = "rejected"
    elif eval_result.decision == "downgrade":
        status = "downgraded"
    elif relevant_conflicts:
        status = "approved_with_conflict"
    else:
        status = "approved"

    applied_grades: list[str] = []
    if eval_result.new_grade:
        applied_grades.append(eval_result.new_grade)

    return AdmittedUnit(
        unit=unit,
        unit_id=unit_id,
        status=status,
        applied_grades=applied_grades,
        evaluation_results=[eval_result],
        conflict_annotations=relevant_conflicts,
        support_ids=support_ids,
    )
