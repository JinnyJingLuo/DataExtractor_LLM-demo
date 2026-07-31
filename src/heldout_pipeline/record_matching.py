from __future__ import annotations

import math
import re

import pandas as pd


ANCHOR_FIELDS = [
    "Metal Symbol",
    "Synthesis Method",
    "Treatment",
    "Initial Temperature (K)",
    "Sample Thickness (mm)",
    "Sample Diameter (mm)",
    "Grain Size (µm)",
    "Initial Density (g/cm³)",
    "Impact Velocity (m/s)",
    "Longitudinal Stress at HEL (GPa)",
    "Peak Stress / Hugoniot Stress (GPa)",
    "Spall Strength (GPa)",
    "Spall Pullback Velocity (m/s)",
    "Experiment Type",
    "Gas Gun Diameter (mm)",
]


def _is_null(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().casefold() in {
        "",
        "-",
        "—",
        "–",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "not reported",
        "not available",
    }


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _number(value: object) -> float | None:
    if _is_null(value):
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value).replace(",", ""))
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _value_similarity(gt_value: object, pred_value: object) -> tuple[float | None, str]:
    if _is_null(gt_value) or _is_null(pred_value):
        return None, "missing_anchor"
    gt_number = _number(gt_value)
    pred_number = _number(pred_value)
    if gt_number is not None and pred_number is not None:
        if gt_number == pred_number:
            return 1.0, "numeric_exact"
        relative_error = abs(pred_number - gt_number) / max(abs(gt_number), 1e-12)
        if relative_error <= 0.005:
            return 1.0, "numeric_under_0.5pct"
        if relative_error <= 0.02:
            return 0.95, "numeric_under_2pct"
        if relative_error <= 0.05:
            return 0.80, "numeric_under_5pct"
        if relative_error <= 0.10:
            return 0.60, "numeric_under_10pct"
        if relative_error <= 0.20:
            return 0.30, "numeric_under_20pct"
        return 0.0, "numeric_over_20pct"
    if _text(gt_value) == _text(pred_value):
        return 1.0, "text_exact"
    return 0.0, "text_mismatch"


def _records(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["paper_id", "sample_id"])
    return (
        frame[frame["field_name"].isin(ANCHOR_FIELDS)]
        .pivot_table(
            index=["paper_id", "sample_id"],
            columns="field_name",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )


def _pair_score(gt_row: pd.Series, pred_row: pd.Series) -> tuple[float, int, str]:
    scores: list[float] = []
    reasons: list[str] = []
    for field in ANCHOR_FIELDS:
        if field not in gt_row.index or field not in pred_row.index:
            continue
        score, reason = _value_similarity(gt_row[field], pred_row[field])
        if score is None:
            continue
        scores.append(score)
        reasons.append(f"{field}:{reason}")
    if not scores:
        return 0.0, 0, "no_comparable_anchor_fields"
    return float(sum(scores) / len(scores)), len(scores), "; ".join(reasons[:8])


def match_records(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    min_score: float = 0.55,
    min_anchors: int = 2,
) -> pd.DataFrame:
    gt_records = _records(ground_truth)
    pred_records = _records(predictions)
    rows: list[dict[str, object]] = []
    for paper_id in sorted(
        set(gt_records.get("paper_id", pd.Series(dtype=str)))
        | set(pred_records.get("paper_id", pd.Series(dtype=str)))
    ):
        gt_paper = gt_records[gt_records["paper_id"].eq(paper_id)].copy()
        pred_paper = pred_records[pred_records["paper_id"].eq(paper_id)].copy()
        gt_ids = set(gt_paper["sample_id"])
        pred_ids = set(pred_paper["sample_id"])
        exact_ids = sorted(gt_ids & pred_ids)
        for sample_id in exact_ids:
            rows.append(
                {
                    "paper_id": paper_id,
                    "ground_truth_sample_id": sample_id,
                    "prediction_sample_id": sample_id,
                    "record_status": "exact_match",
                    "match_score": 1.0,
                    "anchor_count": None,
                    "match_notes": "sample_id_exact",
                }
            )

        gt_unmatched = gt_paper[~gt_paper["sample_id"].isin(exact_ids)]
        pred_unmatched = pred_paper[~pred_paper["sample_id"].isin(exact_ids)]
        candidates: list[dict[str, object]] = []
        for _, gt_row in gt_unmatched.iterrows():
            for _, pred_row in pred_unmatched.iterrows():
                score, anchor_count, notes = _pair_score(gt_row, pred_row)
                if score >= min_score and anchor_count >= min_anchors:
                    candidates.append(
                        {
                            "paper_id": paper_id,
                            "ground_truth_sample_id": gt_row["sample_id"],
                            "prediction_sample_id": pred_row["sample_id"],
                            "match_score": score,
                            "anchor_count": anchor_count,
                            "match_notes": notes,
                        }
                    )
        matched_gt: set[str] = set()
        matched_pred: set[str] = set()
        for candidate in sorted(
            candidates,
            key=lambda row: (-float(row["match_score"]), -int(row["anchor_count"])),
        ):
            gt_id = str(candidate["ground_truth_sample_id"])
            pred_id = str(candidate["prediction_sample_id"])
            if gt_id in matched_gt or pred_id in matched_pred:
                continue
            matched_gt.add(gt_id)
            matched_pred.add(pred_id)
            rows.append({**candidate, "record_status": "record_matched"})

        for sample_id in sorted(gt_ids - set(exact_ids) - matched_gt):
            rows.append(
                {
                    "paper_id": paper_id,
                    "ground_truth_sample_id": sample_id,
                    "prediction_sample_id": pd.NA,
                    "record_status": "ground_truth_only",
                    "match_score": 0.0,
                    "anchor_count": 0,
                    "match_notes": "no_prediction_record_above_threshold",
                }
            )
        for sample_id in sorted(pred_ids - set(exact_ids) - matched_pred):
            rows.append(
                {
                    "paper_id": paper_id,
                    "ground_truth_sample_id": pd.NA,
                    "prediction_sample_id": sample_id,
                    "record_status": "prediction_only",
                    "match_score": 0.0,
                    "anchor_count": 0,
                    "match_notes": "no_ground_truth_record_above_threshold",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "paper_id",
            "ground_truth_sample_id",
            "prediction_sample_id",
            "record_status",
            "match_score",
            "anchor_count",
            "match_notes",
        ],
    )


def _apply_mapping(frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or mapping.empty:
        return frame
    merged = frame.merge(
        mapping,
        left_on=["paper_id", "sample_id"],
        right_on=["paper_id", "prediction_sample_id"],
        how="left",
    )
    remapped = merged["ground_truth_sample_id"].notna()
    merged.loc[remapped, "sample_id"] = merged.loc[remapped, "ground_truth_sample_id"]
    return merged.drop(columns=["prediction_sample_id", "ground_truth_sample_id"])


def apply_record_matching(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    provenance: pd.DataFrame,
    *,
    min_score: float = 0.55,
    min_anchors: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review = match_records(
        ground_truth,
        predictions,
        min_score=min_score,
        min_anchors=min_anchors,
    )
    mapping = review[
        review["record_status"].isin(["record_matched"])
        & review["prediction_sample_id"].notna()
        & review["ground_truth_sample_id"].notna()
    ][["paper_id", "prediction_sample_id", "ground_truth_sample_id"]]
    remapped_predictions = _apply_mapping(predictions, mapping)
    remapped_provenance = _apply_mapping(provenance, mapping)
    duplicates = remapped_predictions.duplicated(
        ["paper_id", "sample_id", "field_name"],
        keep=False,
    )
    if duplicates.any():
        keys = (
            remapped_predictions.loc[duplicates, ["paper_id", "sample_id", "field_name"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"record matching creates duplicate prediction keys: {keys[:5]}")
    return remapped_predictions, remapped_provenance, review
