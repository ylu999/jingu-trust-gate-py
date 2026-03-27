"""
Retry feedback helpers.

The hints-dict pattern appears identically in every build_retry_feedback()
implementation: map reason_code → human hint, build RetryError list, wrap
in RetryFeedback.  hints_feedback() eliminates that boilerplate.

What this helper does NOT do:
- No predefined hints or reason codes
- No default policy for what constitutes a failure
- The caller owns the hints dict entirely
"""

from __future__ import annotations

from jingu_trust_gate.types import RetryError, RetryFeedback, UnitEvaluationResult


def hints_feedback(
    unit_results: list[UnitEvaluationResult],
    hints: dict[str, str],
    *,
    summary: str,
    default_hint: str = "Review proposal and resubmit.",
) -> RetryFeedback:
    """Build a RetryFeedback from rejected/downgraded unit results and a hints dict.

    Only results with decision != "approve" are included as errors.

    Args:
        unit_results: all UnitEvaluationResult from the gate run.
        hints: mapping from reason_code to a human-readable correction hint.
        summary: top-level summary string for the RetryFeedback.
        default_hint: fallback hint for reason codes not in hints.

    Example:
        return hints_feedback(
            unit_results,
            hints={
                "MISSING_CONTEXT": "Add the required context ref to the support pool.",
                "WEAK_JUSTIFICATION": "Expand the justification to explain why this step is necessary.",
            },
            summary=f"{len(failed)} step(s) need correction",
        )
    """
    errors = [
        RetryError(
            unit_id=r.unit_id,
            reason_code=r.reason_code,
            details={"hint": hints.get(r.reason_code, default_hint)},
        )
        for r in unit_results
        if r.decision != "approve"
    ]
    return RetryFeedback(summary=summary, errors=errors)
