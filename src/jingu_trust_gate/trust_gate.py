"""
TrustGate — top-level entry point. Port of src/trust-gate.ts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, Optional, TypeVar

from .audit import AuditWriter, create_default_audit_writer
from .gate import GateRunner
from .policy import GatePolicy
from .retry import LLMInvoker, run_with_retry
from .types import (
    AdmissionResult,
    GateExplanation,
    Proposal,
    RenderContext,
    RetryConfig,
    SupportRef,
    VerifiedContext,
)

TUnit = TypeVar("TUnit")


@dataclass
class TrustGateConfig(Generic[TUnit]):
    policy: GatePolicy[TUnit]
    audit_writer: Optional[AuditWriter] = None
    retry: Optional[RetryConfig] = None
    # Optional: how to extract content from TUnit for base renderer
    extract_content: Optional[Callable[[object, list[SupportRef]], str]] = None


class TrustGate(Generic[TUnit]):
    def __init__(self, config: TrustGateConfig[TUnit]) -> None:
        self._policy = config.policy
        self._audit_writer = config.audit_writer or create_default_audit_writer()
        self._retry = config.retry
        self._runner: GateRunner[TUnit] = GateRunner(self._policy, self._audit_writer)

    async def admit(
        self,
        proposal: Proposal[TUnit],
        support: list[SupportRef],
    ) -> AdmissionResult[TUnit]:
        """
        Synchronous admission — runs Gate only, no LLM.
        Proposal must already be schema-valid.
        """
        return await self._runner.run(proposal, support)

    async def admit_with_retry(
        self,
        invoker: LLMInvoker,
        support: list[SupportRef],
        prompt: str,
    ) -> AdmissionResult[TUnit]:
        """
        Async admission with semantic retry.
        LLMInvoker encapsulates one complete LLM interaction.
        """
        result, _ = await run_with_retry(
            invoker=invoker,
            support=support,
            policy=self._policy,
            prompt=prompt,
            config=self._retry,
            audit_writer=self._audit_writer,
        )
        return result

    def render(
        self,
        result: AdmissionResult[TUnit],
        support: Optional[list[SupportRef]] = None,
        context: Optional[RenderContext] = None,
    ) -> VerifiedContext:
        """
        Render admitted units → VerifiedContext (input for LLM API).
        NOT the final user-facing text.
        """
        ctx = self._policy.render(
            result.admitted_units,
            support or [],
            context or RenderContext(),
        )
        # patch rejected count (policy.render doesn't receive rejected_units)
        ctx.summary.rejected = len(result.rejected_units)
        return ctx

    def explain(self, result: AdmissionResult[TUnit]) -> GateExplanation:
        """Read-only summary of admission result."""
        return _explain_result(result)


def create_trust_gate(
    policy: GatePolicy,
    audit_writer: Optional[AuditWriter] = None,
    retry: Optional[RetryConfig] = None,
    extract_content: Optional[Callable] = None,
) -> TrustGate:
    """Convenience constructor Python entry point — matches TypeScript createTrustGate()."""
    return TrustGate(
        TrustGateConfig(
            policy=policy,
            audit_writer=audit_writer,
            retry=retry,
            extract_content=extract_content,
        )
    )


def _explain_result(result: AdmissionResult) -> GateExplanation:
    all_units = result.admitted_units + result.rejected_units
    reason_codes: set[str] = set()
    for unit in all_units:
        for ev in unit.evaluation_results:
            reason_codes.add(ev.reason_code)

    return GateExplanation(
        total_units=len(all_units),
        approved=sum(1 for u in result.admitted_units if u.status == "approved"),
        downgraded=sum(1 for u in result.admitted_units if u.status == "downgraded"),
        conflicts=sum(
            1 for u in result.admitted_units if u.status == "approved_with_conflict"
        ),
        rejected=len(result.rejected_units),
        retry_attempts=result.retry_attempts,
        gate_reason_codes=sorted(reason_codes),
    )
