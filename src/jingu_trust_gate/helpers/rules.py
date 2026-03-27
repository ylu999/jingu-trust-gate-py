"""
Rule combinators for evaluate_unit() implementations.

These combinators operate on already-evaluated check results, not on callables.
Each check function in your policy should return:
  - None  if the unit passes the check or the check does not apply
  - UnitEvaluationResult with decision != "approve" if evaluation should stop

A check must NOT return an approve() result. That is the caller's responsibility.
Returning approve() from a check is a contract violation and will raise ValueError.

Typical usage:

    from jingu_trust_gate.helpers.outcomes import approve, reject, downgrade
    from jingu_trust_gate.helpers.rules import first_failing

    def evaluate_unit(self, uws, ctx):
        result = first_failing([
            check_intent(uws),
            check_confirmation(uws),
            check_authorization(uws),
        ])
        return result or approve(uws.unit.id)
"""

from __future__ import annotations

from collections.abc import Sequence

from jingu_trust_gate.types import UnitEvaluationResult


def first_failing(
    results: Sequence[UnitEvaluationResult | None],
) -> UnitEvaluationResult | None:
    """Return the first result with decision != 'approve', or None if all pass.

    Args:
        results: A sequence of check outcomes. Each element is either:
                 - None: the check passed or did not apply
                 - UnitEvaluationResult with decision "reject" or "downgrade"

    Returns:
        The first non-None result, or None if all checks passed.

    Raises:
        ValueError: If any non-None result has decision == "approve".
                    Checks must not produce approve results — that is the
                    caller's responsibility after all checks pass.
    """
    for result in results:
        if result is None:
            continue
        if result.decision == "approve":
            raise ValueError(
                f"check returned an approve result for unit '{result.unit_id}' "
                f"(reason_code='{result.reason_code}'). "
                "Check functions must return None to signal pass, not approve(). "
                "Only the caller (evaluate_unit) should produce the final approve."
            )
        return result
    return None
