# jingu-trust-gate

**AI can propose anything. Only verified results are accepted.**

```
AI  →  propose
           ↓
        verify
           ↓
    accept / reject
```

LLMs are proposal generators, not sources of truth. They produce confident output whether or not it is correct. jingu-trust-gate is the layer that decides which proposals are allowed to become system state — by checking each one against evidence before it is admitted.

Nothing passes through unless it can be proven. Every decision is audited.

Python SDK for [jingu-trust-gate](https://github.com/ylu999/jingu-trust-gate). Requires Python 3.11+. Zero runtime dependencies.

## Install

```bash
pip install jingu-trust-gate
```

## The problem

LLMs do not distinguish between what is known and what is guessed. This creates the same failure mode across every LLM use case:

| Use case | What the LLM proposes | What can go wrong |
|---|---|---|
| RAG / Q&A | Claims about retrieved data | Asserts facts not in your evidence |
| Agent planning | Next steps to execute | Proposes steps that lack required context |
| Tool calls | Function calls to make | Calls tools redundantly or without user intent |
| Action execution | Irreversible actions | Acts without authorization or confirmation |

In each case, the LLM output flows directly into system state with no deterministic check. Once an incorrect output is accepted, it is indistinguishable from a correct one — and there is no reproducible way to audit the failure.

jingu-trust-gate inserts a deterministic gate between LLM output and your system state. Only proposals that are provably supported by evidence are allowed through. Every decision is written to an audit log.

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
from jingu_trust_gate.helpers import approve, reject

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
            return reject(uws.unit.id, "MISSING_EVIDENCE")
        return approve(uws.unit.id)

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

## SupportRef — not just evidence

`SupportRef` is the unit of context that a proposal unit can be bound to. `source_type` is a free string — you define the semantics for your domain.

The same mechanism works for any context that needs to constrain what an LLM or agent is allowed to assert or do:

| `source_type` value | What it represents | Typical domain |
|---|---|---|
| `"document"` / `"observation"` | Retrieved RAG evidence | Knowledge base Q&A |
| `"prerequisite"` | A condition that must be true before a step can run | Agent planning |
| `"system_state"` | Current runtime state (queue depth, error count, flag value) | SRE / ops agents |
| `"user_intent"` / `"explicit_request"` | A statement the user actually made | Tool call / action gate |
| `"user_confirmation"` | Explicit user approval for a risky action | High-risk action gate |
| `"prior_result"` / `"tool_output"` | Output from a previous tool call | Multi-step agents |
| `"permission"` / `"authorization"` | A capability or role grant | Authority enforcement |
| `"finding"` | A concluded fact from earlier reasoning | Research agents |

Your `bind_support()` and `evaluate_unit()` filter and check by `source_type`. For example:

```python
from jingu_trust_gate.helpers import approve, reject, downgrade, first_failing

# Tool call gate: reject if no "explicit_request" in support
def evaluate_unit(self, uws, ctx):
    return first_failing([
        None if any(s.source_type == "explicit_request" for s in uws.support_refs)
             else reject(uws.unit.id, "INTENT_NOT_ESTABLISHED"),
    ]) or approve(uws.unit.id)

# Action gate: require "user_confirmation" for high-risk irreversible actions
def evaluate_unit(self, uws, ctx):
    return first_failing([
        reject(uws.unit.id, "CONFIRM_REQUIRED")
        if uws.unit.risk_level == "high" and not uws.unit.is_reversible
           and not any(s.source_type == "user_confirmation" for s in uws.support_refs)
        else None,
    ]) or approve(uws.unit.id)

# Agent step gate: reject if required context not in support pool
def evaluate_unit(self, uws, ctx):
    return first_failing([
        reject(uws.unit.id, "MISSING_CONTEXT")
        if uws.unit.grade == "required" and not uws.support_ids
        else None,
    ]) or approve(uws.unit.id)
```

See `examples/tool_call_policy.py`, `examples/action_gate_policy.py`, and `examples/agent_step_policy.py` for complete working implementations.

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

Eight runnable domain policies in `examples/`. Five cover RAG/data grounding; three show the gate applied to agent steps, tool calls, and irreversible actions.

| File | Domain | Key reason codes |
|------|--------|-----------------|
| [`medical_symptom_policy.py`](examples/medical_symptom_policy.py) | Health assistant | `DIAGNOSIS_UNCONFIRMED`, `TREATMENT_NOT_ADVISED`, `OVER_CERTAIN` |
| [`legal_contract_policy.py`](examples/legal_contract_policy.py) | Contract review | `TERM_NOT_IN_EVIDENCE`, `OVER_SPECIFIC_FIGURE`, `SCOPE_EXCEEDED` |
| [`hpc_diagnostic_policy.py`](examples/hpc_diagnostic_policy.py) | GPU cluster SRE | `UNSUPPORTED_SEVERITY`, `UNSUPPORTED_SCOPE`, `OVER_SPECIFIC_METRIC` |
| [`ecommerce_catalog_policy.py`](examples/ecommerce_catalog_policy.py) | Product chatbot | `UNSUPPORTED_FEATURE`, `OVER_SPECIFIC_STOCK`, `STOCK_CONFLICT` |
| [`bi_analytics_policy.py`](examples/bi_analytics_policy.py) | BI assistant | `VALUE_MISMATCH`, `PERIOD_MISMATCH`, `DIMENSION_MISMATCH`, `METRIC_CONFLICT` |
| [`agent_step_policy.py`](examples/agent_step_policy.py) | Research agent steps | `MISSING_CONTEXT`, `INSUFFICIENT_FINDINGS`, `WEAK_JUSTIFICATION`, `REDUNDANT_STEP` |
| [`tool_call_policy.py`](examples/tool_call_policy.py) | LLM tool call gate | `REDUNDANT_CALL`, `INTENT_NOT_ESTABLISHED`, `WEAK_JUSTIFICATION`, `MISSING_EXPECTED_VALUE` |
| [`action_gate_policy.py`](examples/action_gate_policy.py) | Irreversible action gate | `CONFIRM_REQUIRED`, `DESTRUCTIVE_WITHOUT_AUTHORIZATION`, `SCOPE_EXCEEDED`, `CONTRADICTORY_ACTIONS` |

```bash
python examples/medical_symptom_policy.py
python examples/legal_contract_policy.py
python examples/hpc_diagnostic_policy.py
python examples/ecommerce_catalog_policy.py
python examples/bi_analytics_policy.py
python examples/agent_step_policy.py
python examples/tool_call_policy.py
python examples/action_gate_policy.py
```

## Three iron laws

1. **Gate Engine: zero LLM calls** — all four gate steps are deterministic code. No AI judging AI.
2. **Policy is injected** — the gate core has zero business logic. Domain rules live entirely in `GatePolicy`.
3. **Every admission is audited** — append-only JSONL at `.jingu-trust-gate/audit.jsonl`.

## TypeScript SDK

The [TypeScript SDK](https://github.com/ylu999/jingu-trust-gate) (`npm install jingu-trust-gate`) is the reference implementation. Both SDKs are API-compatible — the same `GatePolicy` design, same pipeline, same type names.

## Changelog

### 0.1.9
- `jingu_trust_gate.helpers` module: `approve()`, `reject()`, `downgrade()` outcome builders; `first_failing()` combinator; `has_support_type()`, `find_support_by_type()` etc. support queries; `empty_proposal_errors()`, `missing_id_errors()`, `missing_text_field_errors()` structure helpers; `hints_feedback()` feedback builder
- All three agent/tool/action example policies refactored to use helpers — `evaluate_unit` now reads `first_failing([...checks]) or approve(id)` throughout
- `ARCHITECTURE.md` added: three-layer model, mechanism vs semantics boundary, what helpers must not become

### 0.1.8
- Three new example policies: `agent_step_policy.py` (research agent step gate), `tool_call_policy.py` (LLM tool call gate), `action_gate_policy.py` (irreversible action gate)
- README: added `SupportRef — not just evidence` section with `source_type` semantics table and code patterns for tool-call, action, and agent-step gates
- README: expanded examples section to cover all 8 example policies

### 0.1.7
- `demo/demo.py` added: narrative walkthrough of all 6 scenarios mirroring the TypeScript demo

### 0.1.6
- Code quality audit across all source files: fixed stale comments, corrected caller attribution in `GatePolicy` docstrings, improved type precision (`Literal["approved_with_conflict"]`)
- Fixed `approved_count` double-counting `approved_with_conflict` units in audit log
- `AuditEntry.downgrade_count` renamed to `downgraded_count`; JSONL key updated to `"downgradedCount"` (aligns with TypeScript SDK)
- Added `demo/demo.py` — narrative walkthrough of all 6 scenarios mirroring the TypeScript demo

### 0.1.5
- Adapter implementations (Claude, OpenAI, Gemini) moved from core to `examples/adapter_examples.py`; only `ContextAdapter` interface remains in the public API
- README rewritten with full quick start, GatePolicy interface table, examples table, adapters section

### 0.1.4
- Five example domain policies: medical, legal, HPC, e-commerce, BI analytics

### 0.1.3
- Initial public release
- Full retry loop with typed `RetryFeedback`
- File audit writer (append-only JSONL)

## License

MIT
