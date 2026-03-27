"""
Core types for jingu-trust-gate Python SDK.
All core types — plain dataclasses, no runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Optional, TypeVar

TUnit = TypeVar("TUnit")

# ── Proposal ──────────────────────────────────────────────────────────────────

ProposalKind = Literal["response", "mutation", "plan", "classification"]


@dataclass
class Proposal(Generic[TUnit]):
    id: str
    kind: ProposalKind
    units: list[TUnit]
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Support ───────────────────────────────────────────────────────────────────

@dataclass
class SupportRef:
    id: str           # system-internal ID — used in support_ids (audit traceability)
    source_type: str
    source_id: str    # business ID — used in evidence_refs matching (policy's bind_support)
    confidence: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    retrieved_at: Optional[str] = None  # ISO 8601


@dataclass
class UnitWithSupport(Generic[TUnit]):
    unit: TUnit
    support_ids: list[str]       # IDs of bound SupportRefs (for audit traceability)
    support_refs: list[SupportRef]  # full SupportRef objects (for attribute inspection)


# ── Gate results ──────────────────────────────────────────────────────────────

@dataclass
class StructureError:
    field: str
    reason_code: str
    message: Optional[str] = None


@dataclass
class StructureValidationResult:
    kind: Literal["structure"] = "structure"
    valid: bool = True
    errors: list[StructureError] = field(default_factory=list)


UnitDecision = Literal["approve", "downgrade", "reject"]


@dataclass
class UnitEvaluationResult:
    kind: Literal["unit"] = "unit"
    unit_id: str = ""
    decision: UnitDecision = "approve"
    reason_code: str = "OK"
    new_grade: Optional[str] = None  # only when decision == "downgrade"
    annotations: dict[str, Any] = field(default_factory=dict)


ConflictSeverity = Literal["informational", "blocking"]


@dataclass
class ConflictAnnotation:
    unit_ids: list[str]
    conflict_code: str
    sources: list[str]  # SupportRef IDs involved
    severity: ConflictSeverity = "informational"
    description: Optional[str] = None


# ── Admission ─────────────────────────────────────────────────────────────────

UnitStatus = Literal["approved", "downgraded", "rejected", "approved_with_conflict"]


@dataclass
class AdmittedUnit(Generic[TUnit]):
    unit: TUnit
    unit_id: str
    status: UnitStatus
    applied_grades: list[str]
    evaluation_results: list[UnitEvaluationResult]
    conflict_annotations: list[ConflictAnnotation] = field(default_factory=list)
    support_ids: list[str] = field(default_factory=list)


@dataclass
class AdmissionResult(Generic[TUnit]):
    proposal_id: str
    admitted_units: list[AdmittedUnit]
    rejected_units: list[AdmittedUnit]
    has_conflicts: bool
    audit_id: str
    retry_attempts: int = 1


# ── Renderer ──────────────────────────────────────────────────────────────────

@dataclass
class VerifiedBlock:
    source_id: str
    content: str
    grade: Optional[str] = None
    conflict_note: Optional[str] = None
    unsupported_attributes: list[str] = field(default_factory=list)


@dataclass
class VerifiedContextSummary:
    admitted: int
    rejected: int
    conflicts: int


@dataclass
class VerifiedContext:
    admitted_blocks: list[VerifiedBlock]
    summary: VerifiedContextSummary
    instructions: Optional[str] = None


@dataclass
class RenderContext:
    user_locale: Optional[str] = None
    channel_type: Optional[Literal["chat", "api", "notification"]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateExplanation:
    total_units: int
    approved: int
    downgraded: int
    conflicts: int
    rejected: int
    retry_attempts: int
    gate_reason_codes: list[str]


# ── Retry ─────────────────────────────────────────────────────────────────────

@dataclass
class RetryError:
    reason_code: str
    unit_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryFeedback:
    summary: str
    errors: list[RetryError]


@dataclass
class RetryConfig:
    max_retries: int = 3
    retry_on_decisions: list[UnitDecision] = field(
        default_factory=lambda: ["reject"]
    )


@dataclass
class RetryContext:
    attempt: int
    max_retries: int
    proposal_id: str


# ── Audit ─────────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    audit_id: str
    timestamp: str   # ISO 8601
    proposal_id: str
    proposal_kind: ProposalKind
    total_units: int
    approved_count: int
    downgrade_count: int
    rejected_count: int
    conflict_count: int
    unit_support_map: dict[str, list[str]]  # unit_id → support_ids
    gate_results: list[Any]
    retry_attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
