#!/usr/bin/env python3
"""Combine extract_paper3.py outputs (Paper1..N) into two workbooks.

extract_paper3.py writes one pair of files per paper to outputs/:
  outputs/Paper{N}_extracted_data.xlsx
  outputs/Paper{N}_evidence_source.xlsx

This script concatenates all of the extracted_data files into one
workbook and all of the evidence_source files into another, each tagged
with a "paper" column, matching the convention used by
combine_gemini_tables.py / combine_gemini_evidence.py.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

EXTRACTED_RE = re.compile(r"^Paper(\d+)_extracted_data\.xlsx$", re.IGNORECASE)
EVIDENCE_RE = re.compile(r"^Paper(\d+)_evidence_source\.xlsx$", re.IGNORECASE)


def collect_files(input_dir: Path, pattern: re.Pattern) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in input_dir.glob("*.xlsx"):
        match = pattern.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    return files


def combine(files: list[tuple[int, Path]], output_path: Path) -> None:
    frames = []
    for num, path in files:
        df = pd.read_excel(path)
        df.insert(0, "paper", num)
        df.insert(1, "source_file", path.name)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No files found to combine for {output_path}.")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_excel(output_path, index=False)
    print(f"Wrote combined workbook to {output_path} ({len(files)} papers).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine extract_paper3.py outputs into two workbooks.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs"), help="Directory containing extract_paper3.py outputs.")
    parser.add_argument(
        "--extracted-output",
        type=Path,
        default=Path("outputs") / "Extracted_Data_Combined.xlsx",
        help="Output path for combined extracted-data workbook.",
    )
    parser.add_argument(
        "--evidence-output",
        type=Path,
        default=Path("outputs") / "Evidence_Source_Combined.xlsx",
        help="Output path for combined evidence-source workbook.",
    )
    args = parser.parse_args()

    extracted_files = collect_files(args.input_dir, EXTRACTED_RE)
    evidence_files = collect_files(args.input_dir, EVIDENCE_RE)

    if not extracted_files:
        raise SystemExit(f"No *_extracted_data.xlsx files found in {args.input_dir}.")
    if not evidence_files:
        raise SystemExit(f"No *_evidence_source.xlsx files found in {args.input_dir}.")

    combine(extracted_files, args.extracted_output)
    combine(evidence_files, args.evidence_output)


if __name__ == "__main__":
    main()
