from __future__ import annotations

import math
import re

import pandas as pd

from .config import EvaluationConfig


KEYS = ["paper_id", "sample_id", "field_name"]


class EvaluationError(ValueError):
    pass


def _reject_duplicate_keys(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        return
    duplicates = frame.duplicated(KEYS, keep=False)
    if duplicates.any():
        keys = frame.loc[duplicates, KEYS].drop_duplicates().to_dict("records")
        raise EvaluationError(f"duplicate {label} keys: {keys[:5]}")


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _is_null(value: object, config: EvaluationConfig) -> bool:
    return _text(value) in config.null_values


def _number(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _loose_number(value: object) -> float | None:
    if pd.isna(value):
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value).strip())
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _relative_error(ground_truth: float, prediction: float, config: EvaluationConfig) -> float:
    return abs(prediction - ground_truth) / max(abs(ground_truth), config.numeric_floor)


def _matches(field: str, ground_truth: object, prediction: object, config: EvaluationConfig) -> bool:
    if field in config.categorical_fields:
        return _text(ground_truth) == _text(prediction)
    gt_number = _number(ground_truth)
    pred_number = _number(prediction)
    if gt_number is not None and pred_number is not None:
        absolute_error = abs(pred_number - gt_number)
        relative_error = _relative_error(gt_number, pred_number, config)
        return (
            absolute_error <= config.absolute_tolerance
            or relative_error <= config.relative_tolerance
        )
    return _text(ground_truth) == _text(prediction)


def _field_type(field: str, config: EvaluationConfig) -> str:
    return "categorical" if field in config.categorical_fields else "numeric"


def _score_value(
    field: str,
    ground_truth: object,
    prediction: object,
    outcome: str,
    config: EvaluationConfig,
) -> tuple[float, str]:
    if outcome == "correct_null":
        return 1.0, "both_missing"
    if outcome in {"missing_extraction", "false_extraction"}:
        return 0.0, outcome
    if field in config.categorical_fields:
        if _text(ground_truth) == _text(prediction):
            return 1.0, "text_exact_match"
        return 0.0, "text_mismatch"

    gt_number = _number(ground_truth)
    pred_number = _number(prediction)
    if gt_number is not None and pred_number is not None:
        error = _relative_error(gt_number, pred_number, config)
        if error <= config.relative_tolerance:
            return 1.0, "perfect_under_0.5pct"
        if error <= 0.02:
            return 0.95, "rounding_under_2pct"
        if error <= 0.05:
            return 0.80, "within_5pct"
        if error <= 0.10:
            return 0.60, "within_10pct"
        if error <= 0.20:
            return 0.30, "within_20pct"
        return 0.0, "major_deviation_over_20pct"

    gt_loose = _loose_number(ground_truth)
    pred_loose = _loose_number(prediction)
    if gt_loose is not None and pred_loose is not None:
        error = _relative_error(gt_loose, pred_loose, config)
        if error <= config.relative_tolerance:
            return 0.90, "format_difference"
        if error <= 0.02:
            return 0.95, "rounding_under_2pct"
        if error <= 0.05:
            return 0.80, "within_5pct"
        if error <= 0.10:
            return 0.60, "within_10pct"
        if error <= 0.20:
            return 0.30, "within_20pct"

    if _text(ground_truth) == _text(prediction):
        return 1.0, "text_exact_match"
    return 0.0, "mismatch"


def evaluate_fields(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    provenance: pd.DataFrame,
    config: EvaluationConfig,
) -> pd.DataFrame:
    _reject_duplicate_keys(ground_truth, "ground-truth")
    _reject_duplicate_keys(predictions, "prediction")
    _reject_duplicate_keys(provenance, "provenance")
    ground_truth = ground_truth[
        ~ground_truth["field_name"].isin(config.ignored_fields)
    ].copy()
    predictions = predictions[
        ~predictions["field_name"].isin(config.ignored_fields)
    ].copy()
    merged = ground_truth.merge(
        predictions,
        on=KEYS,
        how="outer",
        suffixes=("_ground_truth", "_prediction"),
        indicator=True,
    )
    if "value_ground_truth" not in merged.columns:
        merged["value_ground_truth"] = pd.NA
    if "value_prediction" not in merged.columns:
        merged["value_prediction"] = pd.NA

    provenance_columns = KEYS + ["tier", "provenance", "review_status"]
    if provenance.empty:
        provenance = pd.DataFrame(columns=provenance_columns)
    merged = merged.merge(
        provenance[[column for column in provenance_columns if column in provenance.columns]],
        on=KEYS,
        how="left",
    )
    merged["tier"] = merged.get("tier", pd.Series(index=merged.index, dtype=object)).fillna("NA")
    merged["provenance"] = merged.get(
        "provenance", pd.Series(index=merged.index, dtype=object)
    ).fillna("unknown")
    merged["review_status"] = merged.get(
        "review_status", pd.Series(index=merged.index, dtype=object)
    ).fillna("needs_review")

    outcomes: list[str] = []
    scores: list[float] = []
    score_reasons: list[str] = []
    for _, row in merged.iterrows():
        gt_null = row["_merge"] == "right_only" or _is_null(row["value_ground_truth"], config)
        pred_null = row["_merge"] == "left_only" or _is_null(row["value_prediction"], config)
        if gt_null and pred_null:
            outcome = "correct_null"
        elif gt_null:
            outcome = "false_extraction"
        elif pred_null:
            outcome = "missing_extraction"
        elif _matches(
            row["field_name"], row["value_ground_truth"], row["value_prediction"], config
        ):
            outcome = "correct_value"
        else:
            outcome = "wrong_value"
        outcomes.append(outcome)
        score, score_reason = _score_value(
            row["field_name"],
            row["value_ground_truth"],
            row["value_prediction"],
            outcome,
            config,
        )
        scores.append(score)
        score_reasons.append(score_reason)
    merged["outcome"] = outcomes
    merged["score"] = scores
    merged["score_reason"] = score_reasons
    merged["field_type"] = merged["field_name"].map(lambda field: _field_type(field, config))
    merged["include_primary"] = merged["provenance"] != "external_reference"
    return merged.drop(columns=["_merge"])
