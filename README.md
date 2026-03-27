# jingu-trust-gate

**LLM output is untrusted input. jingu-trust-gate decides what is allowed to become trusted system state.**

Python SDK for [jingu-trust-gate](https://github.com/ylu999/jingu-trust-gate) — deterministic admission control layer for LLM systems.

## Install

```bash
pip install jingu-trust-gate
```

Requires Python 3.11+. Zero runtime dependencies.

## The problem

LLMs do not distinguish between what is known and what is guessed. In a RAG pipeline, this means the LLM can assert facts not in your data, over-specify beyond what evidence supports, or silently resolve conflicting sources — with no way to detect or audit it.

jingu-trust-gate inserts a deterministic gate between LLM output and your trusted context. Only claims provably supported by evidence are allowed through. Every decision is written to an audit log.

## Quick start

```python
import asyncio
from dataclasses import dataclass
from jingu_trust_gate import (
    create_trust_gate, GatePolicy, Proposal, SupportRef,
    UnitWithSupport, UnitEvaluationResult, AdmittedUnit,
    VerifiedContext, VerifiedContextSummary, VerifiedBlock,
    StructureValidationResult, ConflictAnnotation,
    RetryFeedback, RenderContext, RetryContext,
    AuditEntry, AuditWriter,
)

@dataclass
class MyClaim:
    id: str
    text: str
    grade: str
    evidence_refs: list[str]

class MyPolicy(GatePolicy[MyClaim]):
    def validate_structure(self, proposal):
        return StructureValidationResult(valid=len(proposal.units) > 0, errors=[])

    def bind_support(self, unit, pool):
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws, ctx):
        if uws.unit.grade == "proven" and not uws.support_ids:
            return UnitEvaluationResult(unit_id=uws.unit.id, decision="reject", reason_code="MISSING_EVIDENCE")
        return UnitEvaluationResult(unit_id=uws.unit.id, decision="approve", reason_code="OK")

    def detect_conflicts(self, units, pool):
        return []

    def render(self, admitted_units, pool, ctx):
        blocks = [VerifiedBlock(source_id=u.unit_id, content=u.unit.text) for u in admitted_units]
        return VerifiedContext(
            admitted_blocks=blocks,
            summary=VerifiedContextSummary(admitted=len(blocks), rejected=0, conflicts=0),
        )

    def build_retry_feedback(self, unit_results, ctx):
        failed = [r for r in unit_results if r.decision == "reject"]
        return RetryFeedback(
            summary=f"{len(failed)} claim(s) rejected",
            errors=[],
        )

class NoopAuditWriter(AuditWriter):
    async def append(self, entry: AuditEntry) -> None:
        pass

async def main():
    gate = create_trust_gate(policy=MyPolicy(), audit_writer=NoopAuditWriter())

    support_pool = [
        SupportRef(id="ref-1", source_id="doc-1", source_type="observation", attributes={}),
    ]
    proposal = Proposal(
        id="prop-1", kind="response",
        units=[
            MyClaim(id="u1", text="Fact with evidence", grade="proven", evidence_refs=["doc-1"]),
            MyClaim(id="u2", text="Hallucinated fact",  grade="proven", evidence_refs=[]),
        ],
    )

    result  = await gate.admit(proposal, support_pool)
    context = gate.render(result)   # VerifiedContext → pass to LLM API
    summary = gate.explain(result)  # GateExplanation(approved, downgraded, rejected, ...)

    print(f"approved={summary.approved}, rejected={summary.rejected}")
    # approved=1, rejected=1

asyncio.run(main())
```

## GatePolicy interface

Implement all six methods. None may call an LLM.

| Method | What it does |
|--------|-------------|
| `validate_structure` | Is the proposal well-formed? (required fields, non-empty, etc.) |
| `bind_support` | Which evidence from the pool applies to this claim? |
| `evaluate_unit` | Should this claim be approved, downgraded, or rejected? |
| `detect_conflicts` | Do any claims contradict each other? |
| `render` | Serialize admitted claims into `VerifiedContext`. |
| `build_retry_feedback` | When gate rejects, what structured feedback should the LLM receive? |

## Unit status

| Status | Meaning | Gate action |
|--------|---------|-------------|
| `approved` | Claim has evidence, nothing over-asserted | Passes through |
| `downgraded` | Claim more specific than evidence supports | Admitted with reduced grade + `unsupported_attributes` |
| `rejected` | No evidence, or categorically unsafe | Blocked — never reaches LLM context |
| `approved_with_conflict` | Has evidence but contradicts another claim | Admitted with `conflict_note` |

`blocking` conflicts force-reject all involved units — `admitted_blocks` is empty, LLM receives only `instructions`. `informational` conflicts admit both with `conflict_note`.

## Adapters

`VerifiedContext` is abstract. Implement `ContextAdapter[T]` to convert it to your LLM API's wire format:

```python
from jingu_trust_gate import ContextAdapter, VerifiedContext

class MyAdapter(ContextAdapter[list[dict]]):
    def adapt(self, context: VerifiedContext) -> list[dict]:
        return [{"role": "user", "content": b.content} for b in context.admitted_blocks]
```

Reference implementations for Claude, OpenAI, and Gemini are in [`examples/adapter_examples.py`](examples/adapter_examples.py).

## Narrative demo

A self-contained walkthrough of the full pipeline — same domain (household memory assistant) used in the TypeScript reference demo:

```bash
python demo/demo.py
```

Covers all 6 scenarios: happy path, missing evidence, over-specificity, conflict detection (informational + blocking), semantic retry loop, and all three adapters (Claude / OpenAI / Gemini).

## Examples

Five runnable domain policies in `examples/`. Each shows a real scenario and what the gate catches that a plain RAG pipeline would not.

| File | Domain | Key reason codes |
|------|--------|-----------------|
| [`medical_symptom_policy.py`](examples/medical_symptom_policy.py) | Health assistant | `DIAGNOSIS_UNCONFIRMED`, `TREATMENT_NOT_ADVISED`, `OVER_CERTAIN` |
| [`legal_contract_policy.py`](examples/legal_contract_policy.py) | Contract review | `TERM_NOT_IN_EVIDENCE`, `OVER_SPECIFIC_FIGURE`, `SCOPE_EXCEEDED` |
| [`hpc_diagnostic_policy.py`](examples/hpc_diagnostic_policy.py) | GPU cluster SRE | `UNSUPPORTED_SEVERITY`, `UNSUPPORTED_SCOPE`, `OVER_SPECIFIC_METRIC` |
| [`ecommerce_catalog_policy.py`](examples/ecommerce_catalog_policy.py) | Product chatbot | `UNSUPPORTED_FEATURE`, `OVER_SPECIFIC_STOCK`, `STOCK_CONFLICT` |
| [`bi_analytics_policy.py`](examples/bi_analytics_policy.py) | BI assistant | `VALUE_MISMATCH`, `PERIOD_MISMATCH`, `DIMENSION_MISMATCH`, `METRIC_CONFLICT` |

```bash
python examples/medical_symptom_policy.py
python examples/legal_contract_policy.py
python examples/hpc_diagnostic_policy.py
python examples/ecommerce_catalog_policy.py
python examples/bi_analytics_policy.py
```

## Three iron laws

1. **Gate Engine: zero LLM calls** — all four gate steps are deterministic code. No AI judging AI.
2. **Policy is injected** — the gate core has zero business logic. Domain rules live entirely in `GatePolicy`.
3. **Every admission is audited** — append-only JSONL at `.jingu-trust-gate/audit.jsonl`.

## TypeScript SDK

The [TypeScript SDK](https://github.com/ylu999/jingu-trust-gate) (`npm install jingu-trust-gate`) is the reference implementation. Both SDKs are API-compatible — the same `GatePolicy` design, same pipeline, same type names.

## License

MIT
