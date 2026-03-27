"""
jingu-trust-gate — deterministic trust gate for LLM systems.
Python SDK: pip install jingu-trust-gate
"""

from .audit import AuditEntry, AuditWriter, FileAuditWriter, create_default_audit_writer
from .harness import Harness, HarnessConfig, create_harness
from .policy import HarnessPolicy
from .types import (
    AdmissionResult,
    AdmittedUnit,
    ConflictAnnotation,
    HarnessExplanation,
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
    "create_harness",
    "Harness",
    "HarnessConfig",
    # Policy
    "HarnessPolicy",
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
    "HarnessExplanation",
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
