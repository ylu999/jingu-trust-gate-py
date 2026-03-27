"""
Audit writer — port of src/audit/audit-log.ts and audit-entry.ts.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from .types import AdmittedUnit, AuditEntry, Proposal


class AuditWriter(ABC):
    @abstractmethod
    async def append(self, entry: AuditEntry) -> None: ...


class FileAuditWriter(AuditWriter):
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    async def append(self, entry: AuditEntry) -> None:
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        line = json.dumps(_entry_to_dict(entry)) + "\n"
        with open(self._file_path, "a", encoding="utf-8") as f:
            f.write(line)


def create_default_audit_writer() -> FileAuditWriter:
    return FileAuditWriter(".jingu-trust-gate/audit.jsonl")


def build_audit_entry(
    *,
    audit_id: str,
    proposal: Proposal,
    all_units: list[AdmittedUnit],
    gate_results: list,
    unit_support_map: dict[str, list[str]],
    retry_attempts: int = 1,
) -> AuditEntry:
    approved = sum(1 for u in all_units if u.status in ("approved", "approved_with_conflict"))
    downgraded = sum(1 for u in all_units if u.status == "downgraded")
    rejected = sum(1 for u in all_units if u.status == "rejected")
    conflicts = sum(1 for u in all_units if u.status == "approved_with_conflict")

    return AuditEntry(
        audit_id=audit_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        proposal_id=proposal.id,
        proposal_kind=proposal.kind,
        total_units=len(all_units),
        approved_count=approved,
        downgrade_count=downgraded,
        rejected_count=rejected,
        conflict_count=conflicts,
        unit_support_map=unit_support_map,
        gate_results=gate_results,
        retry_attempts=retry_attempts,
    )


def _entry_to_dict(entry: AuditEntry) -> dict:
    return {
        "auditId": entry.audit_id,
        "timestamp": entry.timestamp,
        "proposalId": entry.proposal_id,
        "proposalKind": entry.proposal_kind,
        "totalUnits": entry.total_units,
        "approvedCount": entry.approved_count,
        "downgradeCount": entry.downgrade_count,
        "rejectedCount": entry.rejected_count,
        "conflictCount": entry.conflict_count,
        "unitSupportMap": entry.unit_support_map,
        "retryAttempts": entry.retry_attempts,
        "metadata": entry.metadata,
    }
