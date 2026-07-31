import pandas as pd
import pytest

from heldout_pipeline.config import EvaluationConfig
from heldout_pipeline.evaluator import EvaluationError, evaluate_fields


CONFIG = EvaluationConfig(
    relative_tolerance=0.005,
    absolute_tolerance=0.0,
    numeric_floor=1e-12,
    null_values=frozenset({"", "-", "null", "none", "nan"}),
    categorical_fields=frozenset({"Treatment"}),
    external_reference_fields=frozenset(),
    allowed_tiers=frozenset({"T1", "T2", "T3", "NA"}),
    allowed_provenance=frozenset({"paper_text", "unknown", "external_reference"}),
)


def long(rows):
    return pd.DataFrame(rows, columns=["paper_id", "sample_id", "field_name", "value"])


def provenance(rows):
    return pd.DataFrame(
        rows,
        columns=["paper_id", "sample_id", "field_name", "tier", "provenance", "review_status"],
    )


def test_absent_prediction_column_is_missing_not_false_correct():
    gt = long([("P1", "a", "Value", 10.0)])
    pred = long([])
    result = evaluate_fields(gt, pred, provenance([]), CONFIG)
    assert result.iloc[0]["outcome"] == "missing_extraction"


def test_extra_prediction_sample_is_false_extraction():
    gt = long([("P1", "a", "Value", 10.0)])
    pred = long([("P1", "a", "Value", 10.0), ("P1", "extra", "Value", 12.0)])
    result = evaluate_fields(gt, pred, provenance([]), CONFIG)
    assert set(result["outcome"]) == {"correct_value", "false_extraction"}


def test_all_five_outcomes():
    gt = long(
        [
            ("P1", "a", "correct", 10.0),
            ("P1", "a", "wrong", 10.0),
            ("P1", "a", "missing", 10.0),
            ("P1", "a", "false", "-"),
            ("P1", "a", "null", "-"),
        ]
    )
    pred = long(
        [
            ("P1", "a", "correct", 10.04),
            ("P1", "a", "wrong", 11.0),
            ("P1", "a", "false", 3.0),
            ("P1", "a", "null", "-"),
        ]
    )
    result = evaluate_fields(gt, pred, provenance([]), CONFIG)
    assert set(result["outcome"]) == {
        "correct_value",
        "wrong_value",
        "missing_extraction",
        "false_extraction",
        "correct_null",
    }


def test_categorical_is_exact_after_case_and_whitespace_only():
    gt = long([("P1", "a", "Treatment", "Annealed"), ("P1", "b", "Treatment", "Annealed")])
    pred = long([("P1", "a", "Treatment", " annealed "), ("P1", "b", "Treatment", "Annealing")])
    result = evaluate_fields(gt, pred, provenance([]), CONFIG).set_index("sample_id")
    assert result.loc["a", "outcome"] == "correct_value"
    assert result.loc["b", "outcome"] == "wrong_value"


def test_external_reference_is_excluded_from_primary_metric_flag():
    gt = long([("P1", "a", "Density", 2.7)])
    pred = long([("P1", "a", "Density", 2.7)])
    prov = provenance([("P1", "a", "Density", "NA", "external_reference", "approved")])
    result = evaluate_fields(gt, pred, prov, CONFIG)
    assert bool(result.iloc[0]["include_primary"]) is False


def test_verification_output_column_is_not_scored():
    gt = long([("P1", "a", "Value", 10.0)])
    pred = long(
        [
            ("P1", "a", "Value", 10.0),
            ("P1", "a", "Verification", "Verified"),
        ]
    )
    result = evaluate_fields(gt, pred, provenance([]), CONFIG)
    assert result["field_name"].tolist() == ["Value"]


def test_duplicate_provenance_key_is_rejected():
    gt = long([("P1", "a", "Value", 10.0)])
    pred = long([("P1", "a", "Value", 10.0)])
    prov = provenance(
        [
            ("P1", "a", "Value", "T1", "paper_text", "approved"),
            ("P1", "a", "Value", "T2", "unknown", "needs_review"),
        ]
    )
    with pytest.raises(EvaluationError, match="duplicate provenance"):
        evaluate_fields(gt, pred, prov, CONFIG)


def test_marks_numeric_and_categorical_field_types():
    gt = long(
        [
            ("P1", "a", "Value", 10.0),
            ("P1", "a", "Treatment", "Annealed"),
        ]
    )
    pred = long(
        [
            ("P1", "a", "Value", 10.0),
            ("P1", "a", "Treatment", "Annealed"),
        ]
    )

    result = evaluate_fields(gt, pred, provenance([]), CONFIG).set_index("field_name")

    assert result.loc["Value", "field_type"] == "numeric"
    assert result.loc["Treatment", "field_type"] == "categorical"


def test_numeric_graded_scores_follow_project_rubric():
    gt = long(
        [
            ("P1", "a", "perfect", 100.0),
            ("P1", "a", "rounding", 100.0),
            ("P1", "a", "within5", 100.0),
            ("P1", "a", "within10", 100.0),
            ("P1", "a", "within20", 100.0),
            ("P1", "a", "major", 100.0),
            ("P1", "a", "missing", 100.0),
        ]
    )
    pred = long(
        [
            ("P1", "a", "perfect", 100.4),
            ("P1", "a", "rounding", 101.0),
            ("P1", "a", "within5", 104.0),
            ("P1", "a", "within10", 109.0),
            ("P1", "a", "within20", 119.0),
            ("P1", "a", "major", 121.0),
        ]
    )

    result = evaluate_fields(gt, pred, provenance([]), CONFIG).set_index("field_name")

    assert result.loc["perfect", "score"] == 1.0
    assert result.loc["rounding", "score"] == 0.95
    assert result.loc["within5", "score"] == 0.80
    assert result.loc["within10", "score"] == 0.60
    assert result.loc["within20", "score"] == 0.30
    assert result.loc["major", "score"] == 0.0
    assert result.loc["missing", "score"] == 0.0
    assert result.loc["rounding", "score_reason"] == "rounding_under_2pct"


def test_numeric_format_difference_gets_partial_credit():
    gt = long([("P1", "a", "Value", "0.4")])
    pred = long([("P1", "a", "Value", "~0.4")])

    result = evaluate_fields(gt, pred, provenance([]), CONFIG).iloc[0]

    assert result["outcome"] == "wrong_value"
    assert result["score"] == 0.90
    assert result["score_reason"] == "format_difference"


def test_categorical_scores_are_binary_exact_after_mechanical_normalization():
    gt = long(
        [
            ("P1", "a", "Treatment", "Annealed"),
            ("P1", "b", "Treatment", "Annealed"),
        ]
    )
    pred = long(
        [
            ("P1", "a", "Treatment", " annealed "),
            ("P1", "b", "Treatment", "Annealing"),
        ]
    )

    result = evaluate_fields(gt, pred, provenance([]), CONFIG).set_index("sample_id")

    assert result.loc["a", "score"] == 1.0
    assert result.loc["a", "score_reason"] == "text_exact_match"
    assert result.loc["b", "score"] == 0.0
    assert result.loc["b", "score_reason"] == "text_mismatch"
