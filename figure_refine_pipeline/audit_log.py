"""Per-paper audit trail: every field/sample's reconciled value, confidence,
and excluded outlier draws (with their raw evidence text). Never silently
discard an outlier draw's reasoning -- see
docs/superpowers/specs/2026-07-28-t3-figure-pipeline-redesign-design.md,
"Audit trail" -- an excluded outlier's evidence text is exactly what
previously revealed the Paper12 P2/P3 strategy-switching bug.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_audit_log(paper_dir: Path, field_name: str, sample_results: dict) -> None:
    log_path = paper_dir / "audit_log.json"
    entries = json.loads(log_path.read_text()) if log_path.exists() else []
    # Keyed by (field_name, sample_id) so a rerun against the same
    # paper_dir replaces a field/sample's prior entry instead of
    # accumulating duplicate, potentially-conflicting entries for it across
    # runs -- previously this just appended unconditionally, so re-running
    # the pipeline against an existing output directory left stale entries
    # from the old run mixed in with the new ones, with no way to tell
    # which one was current.
    by_key = {(entry["field_name"], entry["sample_id"]): entry for entry in entries}
    for sample_id, result in sample_results.items():
        by_key[(field_name, sample_id)] = {
            "field_name": field_name,
            "sample_id": sample_id,
            "value": result.value,
            "confidence": result.confidence,
            "majority_fraction": result.majority_fraction,
            "outliers": result.outliers,
        }
    log_path.write_text(json.dumps(list(by_key.values()), indent=2), encoding="utf-8")
