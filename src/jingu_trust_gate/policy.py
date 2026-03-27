"""
GatePolicy abstract base class — the interface every policy must implement.
Port of src/types/policy.ts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .types import (
    AdmittedUnit,
    ConflictAnnotation,
    Proposal,
    RenderContext,
    RetryContext,
    RetryFeedback,
    StructureValidationResult,
    SupportRef,
    UnitEvaluationResult,
    UnitWithSupport,
    VerifiedContext,
)

TUnit = TypeVar("TUnit")


class GatePolicy(ABC, Generic[TUnit]):
    """
    Policy interface — implement all 6 methods to define admission logic.

    The gate engine calls these methods in order:
      1. validate_structure  — proposal-level structural check
      2. bind_support        — which SupportRefs apply to this unit
      3. evaluate_unit       — unit-level semantic evaluation
      4. detect_conflicts    — cross-unit conflict detection
      5. render              — admitted units → VerifiedContext for LLM input
      6. build_retry_feedback — structured feedback when gate rejects units
    """

    @abstractmethod
    def validate_structure(
        self, proposal: Proposal[TUnit]
    ) -> StructureValidationResult:
        """Step 1: validate Proposal structure (proposal-level)."""

    @abstractmethod
    def bind_support(
        self, unit: TUnit, support_pool: list[SupportRef]
    ) -> UnitWithSupport[TUnit]:
        """Step 2: bind support to each unit."""

    @abstractmethod
    def evaluate_unit(
        self,
        unit_with_support: UnitWithSupport[TUnit],
        context: dict,  # {"proposal_id": str, "proposal_kind": str}
    ) -> UnitEvaluationResult:
        """Step 3: evaluate each unit against its bound support."""

    @abstractmethod
    def detect_conflicts(
        self,
        units: list[UnitWithSupport[TUnit]],
        support_pool: list[SupportRef],
    ) -> list[ConflictAnnotation]:
        """Step 4: detect cross-unit conflicts."""

    @abstractmethod
    def render(
        self,
        admitted_units: list[AdmittedUnit],
        support_pool: list[SupportRef],
        context: RenderContext,
    ) -> VerifiedContext:
        """Step 5: render admitted units → VerifiedContext for LLM API input."""

    @abstractmethod
    def build_retry_feedback(
        self,
        unit_results: list[UnitEvaluationResult],
        context: RetryContext,
    ) -> RetryFeedback:
        """Step 6: build structured retry feedback from gate results."""
