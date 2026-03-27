"""
BaseRenderer — default render implementation.

Converts admitted units into VerifiedContext (input for LLM API).
Does NOT generate user-facing text — that is the LLM's responsibility.
"""

from __future__ import annotations

from typing import Callable, Optional

from .types import (
    AdmittedUnit,
    ConflictAnnotation,
    RenderContext,
    SupportRef,
    VerifiedBlock,
    VerifiedContext,
    VerifiedContextSummary,
)


class BaseRenderer:
    def render(
        self,
        admitted_units: list[AdmittedUnit],
        support_pool: list[SupportRef],
        context: RenderContext,
        extract_content: Callable[[object, list[SupportRef]], str],
    ) -> VerifiedContext:
        blocks: list[VerifiedBlock] = []

        for admitted in admitted_units:
            unit_support = [s for s in support_pool if s.id in admitted.support_ids]
            content = extract_content(admitted.unit, unit_support)

            block = VerifiedBlock(
                source_id=admitted.unit_id,
                content=content,
                grade=(
                    admitted.applied_grades[-1]
                    if admitted.applied_grades else None
                ),
                conflict_note=(
                    _build_conflict_note(admitted.conflict_annotations)
                    if admitted.status == "approved_with_conflict" else None
                ),
            )
            blocks.append(block)

        conflicts = sum(1 for u in admitted_units if u.status == "approved_with_conflict")

        return VerifiedContext(
            admitted_blocks=blocks,
            summary=VerifiedContextSummary(
                admitted=len(blocks),
                rejected=0,  # patched by TrustGate.render()
                conflicts=conflicts,
            ),
        )


def _build_conflict_note(annotations: list[ConflictAnnotation]) -> str:
    if not annotations:
        return "conflicting information detected"
    return "; ".join(
        a.description or f"conflict detected ({a.conflict_code}): sources {', '.join(a.sources)}"
        for a in annotations
    )
