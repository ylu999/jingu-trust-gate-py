"""
Ergonomic helpers for common policy boilerplate.

These helpers eliminate repetitive patterns that appear across GatePolicy
implementations. They are optional — every helper can be replaced with
equivalent inline code.

Design constraints (see ARCHITECTURE.md):
- No domain semantics (no risk levels, no justification schemas, no grade rules)
- No approve/downgrade/reject logic
- No required fields or mandatory schemas
- Thin functions only, no base classes or mixins
"""

from jingu_trust_gate.helpers.support import (
    has_support_type,
    find_support_by_type,
    filter_support_by_type,
    has_support_attr,
    find_support_by_attr,
    filter_support,
)
from jingu_trust_gate.helpers.structure import (
    empty_proposal_errors,
    missing_id_errors,
    missing_text_field_errors,
)
from jingu_trust_gate.helpers.feedback import (
    hints_feedback,
)

__all__ = [
    # support.py
    "has_support_type",
    "find_support_by_type",
    "filter_support_by_type",
    "has_support_attr",
    "find_support_by_attr",
    "filter_support",
    # structure.py
    "empty_proposal_errors",
    "missing_id_errors",
    "missing_text_field_errors",
    # feedback.py
    "hints_feedback",
]
