"""
Context adapters — convert VerifiedContext into LLM API wire formats.

Only the abstract interface lives here. Concrete implementations for
Claude, OpenAI, Gemini, etc. belong in your application code.
See examples/adapter_examples.py for reference implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .types import VerifiedContext

TOutput = TypeVar("TOutput")


class ContextAdapter(ABC, Generic[TOutput]):
    """
    Convert VerifiedContext into the wire format expected by a specific LLM API.

    gate.render() always outputs VerifiedContext (abstract semantic structure).
    The adapter serializes that into whatever the target API needs.

    Implement this interface in your application code for each LLM provider you use.
    See examples/adapter_examples.py for Claude, OpenAI, and Gemini reference implementations.
    """

    @abstractmethod
    def adapt(self, context: VerifiedContext) -> TOutput: ...
