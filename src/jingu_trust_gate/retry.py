"""
Retry loop — port of src/retry/retry-loop.ts.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Generic, Optional, TypeVar

from .audit import AuditWriter
from .gate import GateRunner
from .policy import HarnessPolicy
from .types import (
    AdmissionResult,
    Proposal,
    RetryConfig,
    RetryContext,
    RetryFeedback,
    SupportRef,
    UnitEvaluationResult,
)

TUnit = TypeVar("TUnit")

# LLMInvoker: takes prompt + optional feedback, returns Proposal
LLMInvoker = Callable[
    [str, Optional[RetryFeedback]],
    Awaitable[Proposal],
]

_DEFAULT_RETRY_CONFIG = RetryConfig(max_retries=3, retry_on_decisions=["reject"])


async def run_with_retry(
    invoker: LLMInvoker,
    support: list[SupportRef],
    policy: HarnessPolicy,
    prompt: str,
    config: Optional[RetryConfig] = None,
    audit_writer: Optional[AuditWriter] = None,
) -> tuple[AdmissionResult, int]:
    """
    Semantic-level retry loop.
    Returns (final AdmissionResult, total attempts).
    """
    cfg = config or _DEFAULT_RETRY_CONFIG
    runner = GateRunner(policy, audit_writer)
    last_result: Optional[AdmissionResult] = None
    attempts = 0

    for attempt in range(cfg.max_retries + 1):
        attempts = attempt + 1

        feedback: Optional[RetryFeedback] = None
        if attempt > 0 and last_result is not None:
            feedback = _build_feedback(last_result, policy, attempt, cfg)

        proposal = await invoker(prompt, feedback)
        last_result = await runner.run(proposal, support)

        all_results = _collect_unit_results(last_result)
        if not _needs_retry(all_results, cfg.retry_on_decisions):
            break
        if attempt >= cfg.max_retries:
            break

    final = AdmissionResult(
        proposal_id=last_result.proposal_id,
        admitted_units=last_result.admitted_units,
        rejected_units=last_result.rejected_units,
        has_conflicts=last_result.has_conflicts,
        audit_id=last_result.audit_id,
        retry_attempts=attempts,
    )
    return final, attempts


def _collect_unit_results(result: AdmissionResult) -> list[UnitEvaluationResult]:
    out = []
    for u in result.admitted_units + result.rejected_units:
        out.extend(u.evaluation_results)
    return out


def _needs_retry(
    results: list[UnitEvaluationResult],
    retry_on: list[str],
) -> bool:
    return any(r.decision in retry_on for r in results)


def _build_feedback(
    result: AdmissionResult,
    policy: HarnessPolicy,
    attempt: int,
    cfg: RetryConfig,
) -> RetryFeedback:
    all_results = _collect_unit_results(result)
    retryable = [r for r in all_results if r.decision in cfg.retry_on_decisions]
    ctx = RetryContext(
        attempt=attempt,
        max_retries=cfg.max_retries,
        proposal_id=result.proposal_id,
    )
    return policy.build_retry_feedback(retryable, ctx)
