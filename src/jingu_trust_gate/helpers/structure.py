"""
Structure validation helpers.

Thin helpers for the boilerplate that appears at the top of every
validate_structure() implementation: empty proposal check, missing id,
empty required text fields.

What these helpers do NOT do:
- No schema inference or reflection
- No required-field declarations
- No domain-specific field names
  (caller always passes field name and value explicitly)
"""

from __future__ import annotations

from jingu_trust_gate.types import Proposal, StructureError


def empty_proposal_errors(proposal: Proposal) -> list[StructureError]:
    """Return [StructureError] if the proposal has no units, else []."""
    if not proposal.units:
        return [StructureError(field="units", reason_code="EMPTY_PROPOSAL")]
    return []


def missing_id_errors(units: list, *, id_attr: str = "id") -> list[StructureError]:
    """Return one StructureError per unit whose id field is empty or missing.

    Args:
        units: list of unit objects (any type).
        id_attr: attribute name to check (default "id").
    """
    errors: list[StructureError] = []
    for unit in units:
        value = getattr(unit, id_attr, None)
        if not value or not str(value).strip():
            errors.append(StructureError(field=id_attr, reason_code="MISSING_UNIT_ID"))
    return errors


def missing_text_field_errors(
    units: list,
    field: str,
    *,
    reason_code: str,
    id_attr: str = "id",
) -> list[StructureError]:
    """Return one StructureError per unit whose text field is empty or missing.

    Args:
        units: list of unit objects.
        field: attribute name to validate.
        reason_code: reason_code to set on the error.
        id_attr: attribute used to identify the unit in the error message.

    Example:
        errors.extend(missing_text_field_errors(
            proposal.units, "description", reason_code="EMPTY_DESCRIPTION"
        ))
    """
    errors: list[StructureError] = []
    for unit in units:
        value = getattr(unit, field, None)
        if not value or not str(value).strip():
            unit_id = getattr(unit, id_attr, "?")
            errors.append(StructureError(
                field=field,
                reason_code=reason_code,
                message=f"unit {unit_id}: {field} is empty or missing",
            ))
    return errors
