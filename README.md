# jingu-trust-gate

**LLM output is untrusted input. jingu-trust-gate decides what is allowed to become trusted system state.**

Python SDK for [jingu-trust-gate](https://github.com/ylu999/jingu-trust-gate) — deterministic admission control layer for LLM systems.

## Install

```bash
pip install jingu-trust-gate
```

## Quick start

```python
from jingu_trust_gate import create_harness, HarnessPolicy, Proposal, SupportRef

class MyPolicy(HarnessPolicy[MyClaim]):
    def validate_structure(self, proposal): ...
    def bind_support(self, unit, pool): ...
    def evaluate_unit(self, unit_with_support, ctx): ...
    def detect_conflicts(self, units, pool): ...
    def render(self, admitted_units, pool, ctx): ...
    def build_retry_feedback(self, unit_results, ctx): ...

harness = create_harness(policy=MyPolicy())
result = await harness.admit(proposal, support_pool)
context = harness.render(result)
```

See `examples/medical_symptom_policy.py` for a complete working example.

## License

MIT
