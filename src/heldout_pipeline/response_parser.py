from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


class ParseError(ValueError):
    pass


@dataclass
class ParsedResponse:
    extracted_data: pd.DataFrame
    evidence_source: pd.DataFrame
    report: dict


def _split_row(line: str) -> list[str]:
    marker = "\u0000PIPE\u0000"
    protected = line.strip().strip("|").replace(r"\|", marker)
    return [cell.strip().replace(marker, "|") for cell in protected.split("|")]


def _is_separator(line: str) -> bool:
    cells = _split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _table_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            current.append(line)
        else:
            if len(current) >= 2 and _is_separator(current[1]):
                blocks.append(current)
            current = []
    if len(current) >= 2 and _is_separator(current[1]):
        blocks.append(current)
    return blocks


def _parse_block(lines: list[str]) -> pd.DataFrame:
    headers = _split_row(lines[0])
    if len(headers) != len(set(headers)):
        raise ParseError(f"duplicate headers: {headers}")
    rows = []
    for line in lines[2:]:
        values = _split_row(line)
        if len(values) != len(headers):
            raise ParseError(
                f"row has {len(values)} cells but header has {len(headers)}: {line}"
            )
        rows.append(values)
    return pd.DataFrame(rows, columns=headers)


def parse_response(text: str, required_columns: Sequence[str]) -> ParsedResponse:
    blocks = _table_blocks(text)
    if len(blocks) < 2:
        raise ParseError(f"expected two Markdown tables, found {len(blocks)}")
    extracted = _parse_block(blocks[0])
    evidence = _parse_block(blocks[1])
    missing = [column for column in required_columns if column not in extracted.columns]
    if missing:
        raise ParseError(f"missing required Table 1 columns: {missing}")
    expected_evidence = {"Column Name", "Source Location", "Notes"}
    evidence_missing = expected_evidence - set(evidence.columns)
    if evidence_missing:
        raise ParseError(f"missing required Table 2 columns: {sorted(evidence_missing)}")
    report = {
        "valid": True,
        "extracted_rows": len(extracted),
        "evidence_rows": len(evidence),
        "unexpected_columns": [
            column for column in extracted.columns if column not in required_columns
        ],
    }
    return ParsedResponse(extracted, evidence, report)


def write_parsed_artifacts(parsed: ParsedResponse, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed.extracted_data.to_csv(output_dir / "extracted_data.csv", index=False)
    parsed.evidence_source.to_csv(output_dir / "evidence_source.csv", index=False)
    (output_dir / "parse_report.json").write_text(
        json.dumps(parsed.report, indent=2), encoding="utf-8"
    )
