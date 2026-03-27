"""
Support ref query helpers.

Thin wrappers around common `source_type` and `attributes` filter patterns.
Every function here is equivalent to a one- or two-line list comprehension —
the value is consistency and readability, not hidden logic.

What these helpers do NOT do:
- No semantic rules (e.g. "proven requires two supports")
- No grade or risk checks
- No approve/reject decisions
"""

from __future__ import annotations

from typing import Any, Callable

from jingu_trust_gate.types import SupportRef


def has_support_type(refs: list[SupportRef], source_type: str) -> bool:
    """Return True if any ref has the given source_type."""
    return any(s.source_type == source_type for s in refs)


def find_support_by_type(refs: list[SupportRef], source_type: str) -> SupportRef | None:
    """Return the first ref with the given source_type, or None."""
    return next((s for s in refs if s.source_type == source_type), None)


def filter_support_by_type(refs: list[SupportRef], source_type: str) -> list[SupportRef]:
    """Return all refs with the given source_type."""
    return [s for s in refs if s.source_type == source_type]


def has_support_attr(refs: list[SupportRef], key: str, value: Any) -> bool:
    """Return True if any ref has attributes[key] == value."""
    return any(s.attributes.get(key) == value for s in refs)


def find_support_by_attr(refs: list[SupportRef], key: str, value: Any) -> SupportRef | None:
    """Return the first ref where attributes[key] == value, or None."""
    return next((s for s in refs if s.attributes.get(key) == value), None)


def filter_support(refs: list[SupportRef], predicate: Callable[[SupportRef], bool]) -> list[SupportRef]:
    """Return all refs matching an arbitrary predicate.

    Use this when the built-in helpers don't cover your filter logic:

        matched = filter_support(pool, lambda s: s.source_type == "finding" and s.attributes.get("verified"))
    """
    return [s for s in refs if predicate(s)]
