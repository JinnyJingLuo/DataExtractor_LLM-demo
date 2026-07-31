#!/usr/bin/env python3
"""Score LLM-extracted shock-physics tables against ground truth, cell by cell.

Pipeline this script expects (see main.tex, "Evaluation Metrics and Accuracy
Definitions"):
  1. extract_paper3.py extracts each paper -> outputs/Paper{N}_extracted_data.xlsx
     and outputs/Paper{N}_evidence_source.xlsx.
  2. combine_extraction_outputs.py concatenates those per-paper files into
     outputs/Extracted_Data_Combined.xlsx and outputs/Evidence_Source_Combined.xlsx
     (each row tagged with a "paper" column).
  3. This script compares the combined extracted data against
     "Ground_truth Table.xlsx" cell by cell and reports accuracy.

Scoring rules (Eqs. in main.tex):
  - Categorical fields: exact match after normalizing whitespace/case.
  - Numerical fields: correct if |x - x_gt| / max(|x_gt|, eps) < 0.005
    (eps = 1e-12), i.e. within 0.5% relative tolerance.
  - Missing entries ("-"/blank in both extracted and ground truth) count
    as correct.
  - Each field is tagged by extraction priority (T1/T2/T3), read from the
    combined evidence workbook's Notes column (looks for "(P1)"/"(P2)"/
    "(P3)"). A column defaults to T1 if no evidence tag is found.
  - Acc_{p,k}, Acc_{p,weighted}, Acc_overall, Acc_overall,k are computed
    exactly as in main.tex.

Usage:
    python evaluate_extraction.py \
        --ground-truth "Ground_truth Table.xlsx" \
        --extracted outputs/Extracted_Data_Combined.xlsx \
        --evidence outputs/Evidence_Source_Combined.xlsx \
        --output outputs/evaluation_results.xlsx
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

KEY_COLUMN = "Sample ID"
PAPER_COLUMN = "sheet"
IGNORE_COLUMNS = {"Verification", "source_file"}
EPS = 1e-12
RELATIVE_TOLERANCE = 0.005

CATEGORICAL_COLUMNS = {
    "Metal Symbol",
    "Synthesis Method",
    "Treatment",
    "Flyer Material Name",
    "Flyer Material Code",
    "Experiment Type",
    "Reference Title",
    "DOI",
}

PRIORITY_RE = re.compile(r"\(P([123])\)")
MISSING_VALUES = {"", "-", "n/a", "na", "none", "nan"}


def normalize_categorical(value: object) -> str:
    if pd.isna(value):
        text = ""
    elif isinstance(value, float) and value.is_integer():
        # Left-merges upcast all-integer columns to float64 when any row is
        # NaN (e.g. an unmatched sample or a numeric-looking code column),
        # turning 7 into 7.0. Strip that back so "7" == "7.0" compares equal.
        text = str(int(value))
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def is_missing(text: str) -> bool:
    return text in MISSING_VALUES


def normalize_key(value: object) -> str:
    """Normalize a Sample ID for row matching: case/whitespace-insensitive,
    and treats '-' and '_' as the same separator (the two data sources are
    not consistent about which one they use, e.g. "Mo_1" vs "Mo-1")."""
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"[\s\-_]+", "_", text).strip().lower()
    return text


def to_float(value: object) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def score_field(column: str, gt_value: object, ext_value: object) -> tuple[bool, str]:
    gt_text = normalize_categorical(gt_value)
    ext_text = normalize_categorical(ext_value)
    gt_missing = is_missing(gt_text)
    ext_missing = is_missing(ext_text)

    if gt_missing or ext_missing:
        if gt_missing and ext_missing:
            return True, "both missing"
        return False, "missing in one but not the other"

    if column not in CATEGORICAL_COLUMNS:
        gt_num = to_float(gt_value)
        ext_num = to_float(ext_value)
        if gt_num is not None and ext_num is not None:
            rel_error = abs(ext_num - gt_num) / max(abs(gt_num), EPS)
            correct = rel_error < RELATIVE_TOLERANCE
            return correct, f"relative error {rel_error:.4%}"

    correct = gt_text == ext_text
    return correct, "exact match" if correct else "mismatch"


def load_priority_tags(evidence: pd.DataFrame) -> dict[int, dict[str, str]]:
    """Return {paper: {column_name: 'T1'|'T2'|'T3'}} parsed from the combined evidence workbook."""
    tags: dict[int, dict[str, str]] = {}
    if not {"paper", "Column Name", "Notes"}.issubset(evidence.columns):
        return tags
    for _, row in evidence.iterrows():
        paper = int(row["paper"])
        column = str(row["Column Name"]).strip()
        match = PRIORITY_RE.search(str(row.get("Notes", "")))
        if match:
            tags.setdefault(paper, {})[column] = f"T{match.group(1)}"
    return tags


def load_ground_truth(path: Path) -> pd.DataFrame:
    gt = pd.read_excel(path, header=1)
    gt[PAPER_COLUMN] = gt[PAPER_COLUMN].astype(str).str.strip()
    gt["paper_num"] = gt[PAPER_COLUMN].str.extract(r"(\d+)").astype(int)
    return gt


def evaluate(
    ground_truth_path: Path,
    extracted_path: Path,
    evidence_path: Optional[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt = load_ground_truth(ground_truth_path)
    extracted = pd.read_excel(extracted_path)
    if "paper" not in extracted.columns:
        raise SystemExit(f"{extracted_path} has no 'paper' column; run combine_extraction_outputs.py first.")

    evidence = pd.read_excel(evidence_path) if evidence_path and evidence_path.exists() else pd.DataFrame()
    priority_tags = load_priority_tags(evidence)

    columns_to_score = [
        c for c in gt.columns if c not in (PAPER_COLUMN, KEY_COLUMN, "paper_num") and c not in IGNORE_COLUMNS
    ]

    field_rows = []
    extracted_papers = set(extracted["paper"].unique())

    for paper_num, gt_paper in gt.groupby("paper_num", sort=True):
        paper_label = f"Paper{paper_num}"

        if paper_num not in extracted_papers:
            for _, gt_row in gt_paper.iterrows():
                for column in columns_to_score:
                    field_rows.append(
                        {
                            "paper": paper_label,
                            "sample_id": gt_row[KEY_COLUMN],
                            "column": column,
                            "tier": "T1",
                            "ground_truth": gt_row[column],
                            "extracted": None,
                            "correct": False,
                            "note": "extracted data missing for this paper",
                        }
                    )
            continue

        extracted_paper = extracted[extracted["paper"] == paper_num].copy()
        tags_for_paper = priority_tags.get(paper_num, {})

        gt_paper = gt_paper.copy()
        gt_paper["_merge_key"] = gt_paper[KEY_COLUMN].map(normalize_key)
        extracted_paper["_merge_key"] = extracted_paper[KEY_COLUMN].map(normalize_key)

        merged = gt_paper.merge(
            extracted_paper,
            on="_merge_key",
            how="left",
            suffixes=("_gt", "_ext"),
        )

        for _, row in merged.iterrows():
            key_col = f"{KEY_COLUMN}_ext" if f"{KEY_COLUMN}_ext" in merged.columns else KEY_COLUMN
            row_unmatched = pd.isna(row.get(key_col))
            for column in columns_to_score:
                gt_col = f"{column}_gt" if f"{column}_gt" in merged.columns else column
                ext_col = f"{column}_ext" if f"{column}_ext" in merged.columns else column
                gt_value = row.get(gt_col)
                ext_value = row.get(ext_col)
                if row_unmatched:
                    correct, note = False, "no matching Sample ID in extracted data"
                else:
                    correct, note = score_field(column, gt_value, ext_value)
                field_rows.append(
                    {
                        "paper": paper_label,
                        "sample_id": row[f"{KEY_COLUMN}_gt" if f"{KEY_COLUMN}_gt" in merged.columns else KEY_COLUMN],
                        "column": column,
                        "tier": tags_for_paper.get(column, "T1"),
                        "ground_truth": gt_value,
                        "extracted": ext_value,
                        "correct": correct,
                        "note": note,
                    }
                )

    field_results = pd.DataFrame(field_rows)
    per_paper_summary = compute_per_paper_summary(field_results)
    overall_summary = compute_overall_summary(field_results)

    return field_results, per_paper_summary, overall_summary


def compute_per_paper_summary(field_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for paper, paper_group in field_results.groupby("paper", sort=False):
        row = {"paper": paper}
        total_correct = 0
        total_n = 0
        for tier in ["T1", "T2", "T3"]:
            tier_group = paper_group[paper_group["tier"] == tier]
            n_total = len(tier_group)
            n_correct = int(tier_group["correct"].sum())
            row[f"{tier}_n"] = n_total
            row[f"{tier}_accuracy"] = n_correct / n_total if n_total else float("nan")
            total_correct += n_correct
            total_n += n_total
        row["total_n"] = total_n
        row["weighted_accuracy"] = total_correct / total_n if total_n else float("nan")
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["paper_num"] = summary["paper"].str.extract(r"(\d+)").astype(int)
    summary = summary.sort_values("paper_num").drop(columns="paper_num").reset_index(drop=True)
    return summary


def compute_overall_summary(field_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_correct_all = int(field_results["correct"].sum())
    total_n_all = len(field_results)
    for tier in ["T1", "T2", "T3"]:
        tier_group = field_results[field_results["tier"] == tier]
        n_total = len(tier_group)
        n_correct = int(tier_group["correct"].sum())
        rows.append(
            {
                "tier": tier,
                "n_correct": n_correct,
                "n_total": n_total,
                "accuracy": n_correct / n_total if n_total else float("nan"),
            }
        )
    rows.append(
        {
            "tier": "overall_weighted",
            "n_correct": total_correct_all,
            "n_total": total_n_all,
            "accuracy": total_correct_all / total_n_all if total_n_all else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", type=Path, default=Path("Ground_truth Table.xlsx"))
    parser.add_argument("--extracted", type=Path, default=Path("outputs") / "Extracted_Data_Combined.xlsx")
    parser.add_argument("--evidence", type=Path, default=Path("outputs") / "Evidence_Source_Combined.xlsx")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "evaluation_results.xlsx")
    args = parser.parse_args()

    if not args.ground_truth.exists():
        raise SystemExit(f"Ground truth file not found: {args.ground_truth}")
    if not args.extracted.exists():
        raise SystemExit(f"Extracted data file not found: {args.extracted}. Run combine_extraction_outputs.py first.")

    field_results, per_paper_summary, overall_summary = evaluate(args.ground_truth, args.extracted, args.evidence)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        overall_summary.to_excel(writer, sheet_name="Overall Summary", index=False)
        per_paper_summary.to_excel(writer, sheet_name="Per-Paper Summary", index=False)
        field_results.to_excel(writer, sheet_name="Per-Field Results", index=False)

    print(f"Wrote evaluation results to {args.output}")
    print(overall_summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
