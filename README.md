# jingu-trust-gate

A deterministic admission layer between your LLM and your system. Every claim an LLM proposes is checked against evidence before it is allowed to affect state or trigger actions.

```
LLM output  →  gate.admit()  →  VerifiedContext  →  your system
```

Not a guardrails framework. Not output validation. An evidence-grounded admission boundary — deterministic, zero LLM calls, fully audited.

Python SDK for [jingu-trust-gate](https://github.com/ylu999/jingu-trust-gate). Requires Python 3.11+. Zero runtime dependencies.

---

## Two failure modes that every AI system eventually hits

### 1. Agent does things you never asked for

User says: `"Order more milk."`

Agent proposes:

```
order_milk              — user asked for this         → should run
delete_old_list         — agent decided on its own    → should NOT run
send_notification_email — agent decided on its own    → should NOT run
```

Without jingu-trust-gate: all three execute.

With jingu-trust-gate:

```
order_milk              → ACCEPT   (has explicit_request evidence)
delete_old_list         → REJECT   (INTENT_NOT_ESTABLISHED)
send_notification_email → REJECT   (INTENT_NOT_ESTABLISHED)
```

### 2. System remembers things you never said

User says: `"We're running low on milk."`

LLM proposes writing to memory:

```json
{ "milk_stock": "low", "user_prefers_brand": "Oatly", "weekly_budget": "$50" }
```

The user never mentioned Oatly. The user never mentioned $50.

Without jingu-trust-gate: all three are stored. The system now treats those guesses as permanent facts — shaping every future recommendation, shopping list, and budget calculation. There is no automatic correction.

With jingu-trust-gate:

```
milk_stock = "low"           → ACCEPT   (verbatim in user statement)
user_prefers_brand = "Oatly" → REJECT   (INFERRED_NOT_STATED)
weekly_budget = "$50"        → REJECT   (INFERRED_NOT_STATED)
```

State after gate: `{ "milk_stock": "low" }`

The two hallucinated facts are blocked at the boundary. They are never stored. They cannot corrupt future queries.

---

The gate does not make the model smarter. It makes the system honest about what it actually knows.

```bash
python demo/aha_moment_demo.py   # two scenarios above, with pacing
python demo/demo.py              # full 8-scenario walkthrough
```

---

## Where it fits in your stack

```
Your retrieval system / event source
            ↓
      support pool        ← the evidence you have
            ↓
       LLM call           ← proposes claims referencing that evidence
            ↓
     gate.admit()         ← deterministic, zero LLM, fully audited
       step 1: validate_structure()  — is the proposal well-formed?
       step 2: bind_support()        — which evidence applies to each claim?
       step 3: evaluate_unit()       — does the claim stay within what evidence supports?
       step 4: detect_conflicts()    — do any claims contradict each other?
            ↓
    AdmissionResult       ← every claim labeled: approved / downgraded / rejected
            ↓
    VerifiedContext        ← only grounded claims reach downstream
            ↓
   Your system / DB / API
```

All domain logic (what counts as "grounded") lives in your `GatePolicy`. The gate core is a fixed pipeline with zero business logic embedded.

---

## Install

```bash
pip install jingu-trust-gate
```

## Quick start

```python
import asyncio
from dataclasses import dataclass
from jingu_trust_gate import (
    create_trust_gate, GatePolicy, Proposal, SupportRef,
    UnitWithSupport, UnitEvaluationResult, AdmittedUnit,
    VerifiedContext, VerifiedContextSummary, VerifiedBlock,
    StructureValidationResult, RetryFeedback, AuditEntry, AuditWriter,
)
from jingu_trust_gate.helpers import approve, reject

@dataclass
class Claim:
    id: str
    text: str
    grade: str           # "proven" | "derived"
    evidence_refs: list[str]

# All domain logic lives in GatePolicy. The gate core has none.
class MyPolicy(GatePolicy):
    def validate_structure(self, proposal):
        return StructureValidationResult(valid=len(proposal.units) > 0, errors=[])

    def bind_support(self, unit, pool):
        matched = [s for s in pool if s.source_id in unit.evidence_refs]
        return UnitWithSupport(unit=unit, support_ids=[s.id for s in matched], support_refs=matched)

    def evaluate_unit(self, uws, ctx):
        if uws.unit.grade == "proven" and not uws.support_ids:
            return reject(uws.unit.id, "MISSING_EVIDENCE")
        return approve(uws.unit.id)

    def detect_conflicts(self, units, pool): return []

    def render(self, admitted_units, pool, ctx):
        blocks = [VerifiedBlock(source_id=u.unit_id, content=u.unit.text) for u in admitted_units]
        return VerifiedContext(
            admitted_blocks=blocks,
            summary=VerifiedContextSummary(admitted=len(blocks), rejected=0, conflicts=0),
        )

    def build_retry_feedback(self, unit_results, ctx):
        failed = [r for r in unit_results if r.decision == "reject"]
        return RetryFeedback(summary=f"{len(failed)} rejected", errors=[])

class NoopAuditWriter(AuditWriter):
    async def append(self, entry: AuditEntry) -> None: pass

async def main():
    gate = create_trust_gate(policy=MyPolicy(), audit_writer=NoopAuditWriter())
    support_pool = [SupportRef(id="ref-1", source_id="doc-1", source_type="observation", attributes={})]
    proposal = Proposal(id="prop-1", kind="response", units=[
        Claim(id="u1", text="Fact with evidence", grade="proven", evidence_refs=["doc-1"]),
        Claim(id="u2", text="Hallucinated fact",  grade="proven", evidence_refs=[]),
    ])

    result  = await gate.admit(proposal, support_pool)
    context = gate.render(result)   # VerifiedContext → pass to LLM API
    summary = gate.explain(result)  # GateExplanation(approved, rejected, ...)

    # What came through:
    for block in context.admitted_blocks:
        print(f"admitted: {block.source_id!r}  {block.content!r}")
    # admitted: 'u1'  'Fact with evidence'

    # What was blocked (and why):
    for u in result.rejected_units:
        print(f"rejected: {u.unit_id!r}  reason={u.evaluation_results[0].reason_code!r}")
    # rejected: 'u2'  reason='MISSING_EVIDENCE'

asyncio.run(main())
```

## Three iron laws

1. **Zero LLM calls in the gate** — all four steps are deterministic code. No AI judging AI. The same input always produces the same admission decision.

2. **Policy is injected, not embedded** — the gate core has zero domain logic. Every business rule lives in your `GatePolicy`. Swap the policy, the gate stays identical.

3. **Every admission is audited** — append-only JSONL at `.jingu-trust-gate/audit.jsonl`. Every claim's fate is on record with its `audit_id`, reason code, and timestamp.

## This is not a guardrails framework

Guardrails frameworks check whether output is **safe or well-formed** — they block toxic content, enforce schemas, detect PII. That is a different problem.

jingu-trust-gate checks whether each **proposal is actually supported by evidence**. It does not care whether output is polite or syntactically valid. It cares whether what the LLM proposes can be proven correct before it becomes system state.

| System | Question it answers | When it runs |
|--------|---|---|
| Guardrails AI | Is the output safe? | after generation |
| NeMo Guardrails | Is the bot on-topic? | at conversation level |
| RAG / grounding | Did retrieval find relevant docs? | before generation |
| DeepEval | How often does the model hallucinate? | offline, in eval |
| **jingu-trust-gate** | **Is this proposal allowed to become state?** | **at every admission, deterministically** |

To our knowledge, existing systems validate outputs, evaluate models, or retrieve evidence — but do not provide a deterministic admission boundary that enforces what claims are allowed to be treated as true at runtime.

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

See `examples/actions/tool_call_policy.py` and `examples/actions/action_gate_policy.py` for complete working implementations.

## Adapters

`VerifiedContext` is abstract. Implement `ContextAdapter[T]` to convert it to your LLM API's wire format:

```python
from jingu_trust_gate import ContextAdapter, VerifiedContext

class MyAdapter(ContextAdapter[list[dict]]):
    def adapt(self, context: VerifiedContext) -> list[dict]:
        return [{"role": "user", "content": b.content} for b in context.admitted_blocks]
```

Reference implementations for Claude, OpenAI, and Gemini are in [`examples/integration/adapter_examples.py`](examples/integration/adapter_examples.py).

## Narrative demo

```bash
python demo/demo.py        # full 8-scenario walkthrough
```

Covers 8 scenarios: happy path, missing evidence, over-specificity, conflict detection (informational + blocking), semantic retry loop, all three adapters (Claude / OpenAI / Gemini), agent action gate (`INTENT_NOT_ESTABLISHED`, `CONFIRM_REQUIRED`), and preventing memory corruption (`INFERRED_NOT_STATED`, state drift).

## Examples

Nine runnable domain policies in `examples/`, organized into four use-case categories.

### answers/ — gate what the LLM claims in a response

| File | Domain | Key reason codes |
|------|--------|-----------------|
| [`answers/medical_symptom_policy.py`](examples/answers/medical_symptom_policy.py) | Health assistant | `DIAGNOSIS_UNCONFIRMED`, `TREATMENT_NOT_ADVISED`, `OVER_CERTAIN` |
| [`answers/legal_contract_policy.py`](examples/answers/legal_contract_policy.py) | Contract review | `TERM_NOT_IN_EVIDENCE`, `OVER_SPECIFIC_FIGURE`, `SCOPE_EXCEEDED` |

### actions/ — gate what the LLM agent is allowed to do

| File | Domain | Key reason codes |
|------|--------|-----------------|
| [`actions/tool_call_policy.py`](examples/actions/tool_call_policy.py) | LLM tool call gate | `REDUNDANT_CALL`, `INTENT_NOT_ESTABLISHED`, `WEAK_JUSTIFICATION`, `MISSING_EXPECTED_VALUE` |
| [`actions/action_gate_policy.py`](examples/actions/action_gate_policy.py) | Irreversible action gate | `CONFIRM_REQUIRED`, `DESTRUCTIVE_WITHOUT_AUTHORIZATION`, `SCOPE_EXCEEDED`, `CONTRADICTORY_ACTIONS` |

### state/ — gate what the LLM is allowed to write into persistent state

| File | Domain | Key reason codes |
|------|--------|-----------------|
| [`state/memory_update_policy.py`](examples/state/memory_update_policy.py) | Personal memory assistant | `SOURCE_UNVERIFIED`, `INFERRED_NOT_STATED`, `SCOPE_VIOLATION` |
| [`state/fact_write_policy.py`](examples/state/fact_write_policy.py) | Knowledge base write gate | `UNSOURCED`, `OVER_SPECIFIC`, `LOW_CONFIDENCE_SOURCE`, `CONFLICTING_VALUES` |

### integration/ — audit logging, retry loops, adapters

| File | What it shows |
|------|--------------|
| [`integration/audit_writer_example.py`](examples/integration/audit_writer_example.py) | `FileAuditWriter` — JSONL audit log (Law 3) |
| [`integration/downgrade_retry_example.py`](examples/integration/downgrade_retry_example.py) | `retry_on_decisions=["downgrade"]` — retry when claims are downgraded |
| [`integration/adapter_examples.py`](examples/integration/adapter_examples.py) | `ContextAdapter` for Claude, OpenAI, Gemini |

```bash
# answers
python examples/answers/medical_symptom_policy.py
python examples/answers/legal_contract_policy.py

# actions
python examples/actions/tool_call_policy.py
python examples/actions/action_gate_policy.py

# state
python examples/state/memory_update_policy.py
python examples/state/fact_write_policy.py

# integration
python examples/integration/audit_writer_example.py
python examples/integration/downgrade_retry_example.py
python examples/integration/adapter_examples.py
```

## TypeScript SDK

The [TypeScript SDK](https://github.com/ylu999/jingu-trust-gate) (`npm install jingu-trust-gate`) is the reference implementation. Both SDKs are API-compatible — the same `GatePolicy` design, same pipeline, same type names.

## Changelog

### 0.1.11
- `demo/demo.py` — added Scenario 7 (Agent Action Gate: `INTENT_NOT_ESTABLISHED`, `CONFIRM_REQUIRED`) and Scenario 8 (Preventing Memory Corruption: `INFERRED_NOT_STATED`, state drift). Now 8 scenarios.
- README: hero section rewritten with concrete before/after examples for both failure modes. Quick start and Three iron laws moved to top.

### 0.1.10
- Examples reorganized into four use-case categories: `answers/`, `actions/`, `state/`, `integration/`
- Removed: `ecommerce_catalog_policy.py`, `hpc_diagnostic_policy.py`, `bi_analytics_policy.py`, `agent_step_policy.py` (overlapping patterns)
- New: `state/memory_update_policy.py` — personal memory write gate (`SOURCE_UNVERIFIED`, `INFERRED_NOT_STATED`, `SCOPE_VIOLATION`)
- New: `state/fact_write_policy.py` — KB fact write gate (`UNSOURCED`, `OVER_SPECIFIC`, `LOW_CONFIDENCE_SOURCE`, `CONFLICTING_VALUES`)
- New: `integration/audit_writer_example.py` — `FileAuditWriter` usage and JSONL log verification (Law 3)
- New: `integration/downgrade_retry_example.py` — `retry_on_decisions=["downgrade"]` pattern with `RetryFeedback` walkthrough
- Each subdirectory has a `README.md` with mental model and use-case guide

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
