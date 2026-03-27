"""
Reference adapter implementations for jingu-trust-gate.

These show how to implement ContextAdapter for Claude, OpenAI, and Gemini.
Copy and adapt as needed for your application — they are NOT part of the
core SDK.

Usage:
    from examples.adapter_examples import ClaudeContextAdapter, OpenAIContextAdapter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from jingu_trust_gate import ContextAdapter
from jingu_trust_gate.types import VerifiedBlock, VerifiedContext


# ── Claude ─────────────────────────────────────────────────────────────────────

@dataclass
class ClaudeSearchResultBlock:
    """Simplified representation of a Claude API search_result block."""
    type: str = "search_result"
    source: str = ""
    title: str = ""
    content: list[dict] = field(default_factory=list)
    citations: Optional[dict] = None


@dataclass
class ClaudeAdapterOptions:
    citations: bool = True
    source_prefix: str = ""


class ClaudeContextAdapter(ContextAdapter[list[ClaudeSearchResultBlock]]):
    """
    Converts VerifiedContext → Claude API search_result blocks.

    Each admitted block becomes one search_result block.
    Downgraded grade and conflict notes are appended to the text content
    so Claude sees them as contextual caveats.

    Usage:
        adapter = ClaudeContextAdapter()
        blocks = adapter.adapt(verified_ctx)
        # Pass blocks as tool_result content or top-level user message content
    """

    def __init__(self, options: Optional[ClaudeAdapterOptions] = None) -> None:
        opts = options or ClaudeAdapterOptions()
        self._citations = opts.citations
        self._source_prefix = opts.source_prefix

    def adapt(self, context: VerifiedContext) -> list[ClaudeSearchResultBlock]:
        return [self._block_to_search_result(b) for b in context.admitted_blocks]

    def _block_to_search_result(self, block: VerifiedBlock) -> ClaudeSearchResultBlock:
        parts = [block.content]
        if block.grade:
            parts.append(f"[Evidence grade: {block.grade}]")
        if block.unsupported_attributes:
            parts.append(f"[Not supported by evidence: {', '.join(block.unsupported_attributes)}]")
        if block.conflict_note:
            parts.append(f"[Conflict: {block.conflict_note}]")
        return ClaudeSearchResultBlock(
            type="search_result",
            source=f"{self._source_prefix}{block.source_id}",
            title=block.source_id,
            content=[{"type": "text", "text": "\n".join(parts)}],
            citations={"enabled": self._citations},
        )


# ── OpenAI ─────────────────────────────────────────────────────────────────────

@dataclass
class OpenAIChatMessage:
    """OpenAI chat message — tool or user role."""
    role: str
    content: str
    tool_call_id: Optional[str] = None


@dataclass
class OpenAIAdapterOptions:
    mode: str = "user"          # "tool" | "user"
    tool_call_id: Optional[str] = None
    block_separator: str = "\n\n---\n\n"


class OpenAIContextAdapter(ContextAdapter[OpenAIChatMessage]):
    """
    Converts VerifiedContext → OpenAI chat message.

    OpenAI does not have a native search_result block type; verified content
    is serialised as plain text with semantic caveats inline.

    Usage (tool mode):
        adapter = OpenAIContextAdapter(OpenAIAdapterOptions(mode="tool", tool_call_id=call_id))
        msg = adapter.adapt(verified_ctx)

    Usage (user mode):
        adapter = OpenAIContextAdapter()
        msg = adapter.adapt(verified_ctx)
    """

    def __init__(self, options: Optional[OpenAIAdapterOptions] = None) -> None:
        opts = options or OpenAIAdapterOptions()
        self._mode = opts.mode
        self._tool_call_id = opts.tool_call_id
        self._block_separator = opts.block_separator

    def adapt(self, context: VerifiedContext) -> OpenAIChatMessage:
        parts = [self._block_to_text(b) for b in context.admitted_blocks]
        content = self._block_separator.join(parts)
        if self._mode == "tool":
            return OpenAIChatMessage(
                role="tool",
                content=content,
                tool_call_id=self._tool_call_id or "",
            )
        return OpenAIChatMessage(role="user", content=content)

    def _block_to_text(self, block: VerifiedBlock) -> str:
        lines = [f"[{block.source_id}] {block.content}"]
        if block.grade:
            lines.append(f"Evidence grade: {block.grade}")
        if block.unsupported_attributes:
            lines.append(f"Not supported by evidence: {', '.join(block.unsupported_attributes)}")
        if block.conflict_note:
            lines.append(f"Conflict: {block.conflict_note}")
        return "\n".join(lines)


# ── Gemini ─────────────────────────────────────────────────────────────────────

@dataclass
class GeminiTextPart:
    text: str


@dataclass
class GeminiContent:
    """Gemini API Content object (one turn in the conversation)."""
    role: str
    parts: list[GeminiTextPart]


@dataclass
class GeminiAdapterOptions:
    role: str = "user"          # "user" | "function"


class GeminiContextAdapter(ContextAdapter[GeminiContent]):
    """
    Converts VerifiedContext → Gemini API Content object.

    Usage:
        adapter = GeminiContextAdapter()
        content = adapter.adapt(verified_ctx)

        result = model.generate_content(contents=[
            content,
            {"role": "user", "parts": [{"text": user_query}]},
        ])
    """

    def __init__(self, options: Optional[GeminiAdapterOptions] = None) -> None:
        opts = options or GeminiAdapterOptions()
        self._role = opts.role

    def adapt(self, context: VerifiedContext) -> GeminiContent:
        if not context.admitted_blocks:
            return GeminiContent(
                role=self._role,
                parts=[GeminiTextPart(text="[No verified context available]")],
            )
        return GeminiContent(
            role=self._role,
            parts=[GeminiTextPart(text=self._block_to_text(b)) for b in context.admitted_blocks],
        )

    def _block_to_text(self, block: VerifiedBlock) -> str:
        lines = [f"[{block.source_id}] {block.content}"]
        if block.grade:
            lines.append(f"Evidence grade: {block.grade}")
        if block.unsupported_attributes:
            lines.append(f"Not supported by evidence: {', '.join(block.unsupported_attributes)}")
        if block.conflict_note:
            lines.append(f"Conflict: {block.conflict_note}")
        return "\n".join(lines)
