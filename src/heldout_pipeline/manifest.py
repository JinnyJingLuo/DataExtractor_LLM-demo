from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    pdf_path: Path
    split: str
    include: bool
    selection_note: str


def _parse_bool(value: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ManifestError(f"row {row_number}: invalid include value {value!r}")


def load_manifest(path: Path) -> list[PaperRecord]:
    path = Path(path)
    required = {"paper_id", "pdf_path", "split", "include", "selection_note"}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ManifestError(f"missing manifest columns: {sorted(missing)}")
        raw_rows = list(reader)

    records: list[PaperRecord] = []
    seen: set[str] = set()
    for row_number, row in enumerate(raw_rows, start=2):
        paper_id = row["paper_id"].strip()
        if not paper_id:
            raise ManifestError(f"row {row_number}: paper_id is required")
        if paper_id in seen:
            raise ManifestError(f"duplicate paper_id: {paper_id}")
        seen.add(paper_id)

        split = row["split"].strip().lower()
        if split not in {"development", "heldout"}:
            raise ManifestError(f"row {row_number}: invalid split {split!r}")
        include = _parse_bool(row["include"], row_number)
        note = row["selection_note"].strip()
        if split == "heldout" and include and not note:
            raise ManifestError(f"row {row_number}: heldout selection_note is required")

        raw_pdf = Path(row["pdf_path"].strip()).expanduser()
        pdf_path = raw_pdf if raw_pdf.is_absolute() else path.parent / raw_pdf
        records.append(
            PaperRecord(
                paper_id=paper_id,
                pdf_path=pdf_path.resolve(),
                split=split,
                include=include,
                selection_note=note,
            )
        )

    for record in records:
        if record.include and not record.pdf_path.is_file():
            raise ManifestError(
                f"included PDF for {record.paper_id} does not exist: {record.pdf_path}"
            )
    return records


def select_papers(records: Iterable[PaperRecord], split: str) -> list[PaperRecord]:
    if split not in {"development", "heldout"}:
        raise ManifestError(f"invalid split {split!r}")
    return [record for record in records if record.include and record.split == split]
