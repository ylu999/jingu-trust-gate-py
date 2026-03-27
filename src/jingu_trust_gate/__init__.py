"""
jingu-trust-gate — deterministic trust gate for LLM systems.
Python SDK: pip install jingu-trust-gate
"""

from .audit import AuditEntry, AuditWriter, FileAuditWriter, create_default_audit_writer
from .trust_gate import TrustGate, TrustGateConfig, create_trust_gate
from .policy import GatePolicy
from .types import (
    AdmissionResult,
    AdmittedUnit,
    ConflictAnnotation,
    GateExplanation,
    Proposal,
    RenderContext,
    RetryConfig,
    RetryContext,
    RetryError,
    RetryFeedback,
    SupportRef,
    StructureError,
    StructureValidationResult,
    UnitEvaluationResult,
    UnitWithSupport,
    VerifiedBlock,
    VerifiedContext,
    VerifiedContextSummary,
)

__version__ = "0.1.0"

__all__ = [
    # Harness
    "create_trust_gate",
    "TrustGate",
    "TrustGateConfig",
    # Policy
    "GatePolicy",
    # Types
    "Proposal",
    "SupportRef",
    "UnitWithSupport",
    "StructureError",
    "StructureValidationResult",
    "UnitEvaluationResult",
    "ConflictAnnotation",
    "AdmittedUnit",
    "AdmissionResult",
    "VerifiedBlock",
    "VerifiedContext",
    "VerifiedContextSummary",
    "RenderContext",
    "GateExplanation",
    "RetryFeedback",
    "RetryError",
    "RetryConfig",
    "RetryContext",
    # Audit
    "AuditEntry",
    "AuditWriter",
    "FileAuditWriter",
    "create_default_audit_writer",
]
