"""
Conflict annotator utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import AdmittedUnit, ConflictAnnotation


@dataclass
class ConflictSurface:
    unit_id: str
    status: str  # "approved_with_conflict"
    conflict_code: str
    conflicting_support_ids: list[str]
    description: Optional[str] = None


def surface_conflicts(
    admitted_units: list[AdmittedUnit],
    annotations: list[ConflictAnnotation],
) -> list[ConflictSurface]:
    """
    Return ConflictSurface for each admitted unit that has a conflict annotation.
    Pure function — does not mutate inputs.
    """
    surfaces = []
    for unit in admitted_units:
        if unit.status != "approved_with_conflict":
            continue
        annotation = next((a for a in annotations if unit.unit_id in a.unit_ids), None)
        if annotation is None:
            continue
        surfaces.append(ConflictSurface(
            unit_id=unit.unit_id,
            status="approved_with_conflict",
            conflict_code=annotation.conflict_code,
            conflicting_support_ids=annotation.sources,
            description=annotation.description,
        ))
    return surfaces


def group_conflicts_by_code(
    annotations: list[ConflictAnnotation],
) -> dict[str, list[ConflictAnnotation]]:
    """Group ConflictAnnotations by conflict_code."""
    groups: dict[str, list[ConflictAnnotation]] = {}
    for ann in annotations:
        groups.setdefault(ann.conflict_code, []).append(ann)
    return groups


def has_conflicts(annotations: list[ConflictAnnotation]) -> bool:
    return len(annotations) > 0
