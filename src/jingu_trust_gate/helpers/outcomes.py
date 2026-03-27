"""
Outcome builders for UnitEvaluationResult.

These are the canonical way to construct gate decisions inside a policy.
Using these instead of hand-building dicts ensures consistent shape and
avoids typos in field names.

Contract:
  - approve()   — unit passes all checks
  - reject()    — unit must not be admitted
  - downgrade() — unit is admitted with reduced grade and flagged attributes

These are value constructors only. They contain no logic.
"""

from __future__ import annotations

from jingu_trust_gate.types import UnitEvaluationResult


def approve(unit_id: str, *, reason_code: str = "OK") -> UnitEvaluationResult:
    """Unit passes. Use as the default return at the end of evaluate_unit()."""
    return UnitEvaluationResult(unit_id=unit_id, decision="approve", reason_code=reason_code)


def reject(
    unit_id: str,
    reason_code: str,
    *,
    note: str | None = None,
    **extra_annotations: object,
) -> UnitEvaluationResult:
    """Unit is rejected and will not be admitted.

    Args:
        unit_id:    The unit being evaluated.
        reason_code: Machine-readable failure code (e.g. "MISSING_CONTEXT").
        note:        Optional human-readable explanation added to annotations.
        **extra_annotations: Any additional annotation keys to attach.
    """
    annotations: dict[str, object] = {**extra_annotations}
    if note is not None:
        annotations["note"] = note
    return UnitEvaluationResult(
        unit_id=unit_id,
        decision="reject",
        reason_code=reason_code,
        annotations=annotations or None,
    )


def downgrade(
    unit_id: str,
    reason_code: str,
    new_grade: str,
    *,
    note: str | None = None,
    **extra_annotations: object,
) -> UnitEvaluationResult:
    """Unit is admitted with a reduced grade and flagged unsupported attributes.

    Args:
        unit_id:    The unit being evaluated.
        reason_code: Machine-readable reason for the downgrade.
        new_grade:   The grade to apply after downgrading (e.g. "speculative").
        note:        Optional human-readable explanation.
        **extra_annotations: Any additional annotation keys.
    """
    annotations: dict[str, object] = {**extra_annotations}
    if note is not None:
        annotations["note"] = note
    return UnitEvaluationResult(
        unit_id=unit_id,
        decision="downgrade",
        reason_code=reason_code,
        new_grade=new_grade,
        annotations=annotations or None,
    )
