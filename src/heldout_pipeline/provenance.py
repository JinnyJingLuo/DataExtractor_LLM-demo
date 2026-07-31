from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


TIER_RE = re.compile(r"\b(?:P|T|PRIORITY\s*)([123])\b", re.IGNORECASE)


@dataclass(frozen=True)
class ValidationIssue:
    row: int
    message: str


def _infer(notes: str, location: str) -> tuple[str, str]:
    text = f"{notes} {location}"
    match = TIER_RE.search(text)
    tier = f"T{match.group(1)}" if match else "NA"
    upper = text.upper()
    if tier == "T3" or "FIGURE" in upper or "FIG." in upper:
        return tier, "paper_figure"
    if tier == "T2" or "CALCULAT" in upper or "EQUATION" in upper or "EQ." in upper:
        return tier, "derived_from_paper"
    if tier == "T1":
        return tier, "paper_table" if "TABLE" in upper else "paper_text"
    return "NA", "unknown"


def _is_external_reference(field_name: str, notes: str, external_fields: set[str]) -> bool:
    if field_name not in external_fields:
        return False
    normalized = notes.casefold()
    explicit_external = any(
        marker in normalized
        for marker in (
            "standard value",
            "external",
            "assumed",
            "room temperature",
            "not reported",
        )
    )
    return explicit_external or not TIER_RE.search(notes)


def prepare_provenance(
    predictions: pd.DataFrame,
    evidence: pd.DataFrame,
    external_reference_fields: set[str],
) -> pd.DataFrame:
    evidence = evidence.copy()
    if evidence.empty:
        evidence = pd.DataFrame(
            columns=["paper_id", "field_name", "source_location", "notes"]
        )
    for column in ["paper_id", "field_name", "source_location", "notes"]:
        if column not in evidence.columns:
            evidence[column] = ""
    evidence = evidence.drop_duplicates(["paper_id", "field_name"], keep="last")
    merged = predictions[["paper_id", "sample_id", "field_name"]].merge(
        evidence[["paper_id", "field_name", "source_location", "notes"]],
        on=["paper_id", "field_name"],
        how="left",
    )
    rows = []
    for _, row in merged.iterrows():
        location = "" if pd.isna(row["source_location"]) else str(row["source_location"])
        notes = "" if pd.isna(row["notes"]) else str(row["notes"])
        if _is_external_reference(row["field_name"], notes, external_reference_fields):
            tier, source = "NA", "external_reference"
        else:
            tier, source = _infer(notes, location)
        rows.append(
            {
                "paper_id": row["paper_id"],
                "sample_id": row["sample_id"],
                "field_name": row["field_name"],
                "tier": tier,
                "provenance": source,
                "evidence_location": location,
                "evidence_text": notes,
                "review_status": "needs_review",
            }
        )
    return pd.DataFrame(rows)


def validate_provenance(records: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for index, row in records.iterrows():
        if row.get("tier") not in {"T1", "T2", "T3", "NA"}:
            issues.append(ValidationIssue(int(index), "invalid tier"))
        if row.get("provenance") == "unknown":
            issues.append(ValidationIssue(int(index), "unknown provenance requires review"))
    return issues
