import pandas as pd

from heldout_pipeline.metrics import summarize_metrics


def test_metrics_use_information_extraction_denominators():
    outcomes = pd.DataFrame(
        {
            "split": ["heldout"] * 5,
            "outcome": [
                "correct_value",
                "wrong_value",
                "missing_extraction",
                "false_extraction",
                "correct_null",
            ],
            "include_primary": [True] * 5,
            "score": [1.0, 0.6, 0.0, 0.0, 1.0],
        }
    )
    row = summarize_metrics(outcomes, ["split"]).iloc[0]
    assert row["precision"] == 1 / 3
    assert row["recall"] == 1 / 3
    assert row["f1"] == 1 / 3
    assert row["strict_accuracy_all"] == 2 / 5
    assert row["strict_accuracy_non_null"] == 1 / 3
    assert row["accuracy_all"] == 2.6 / 5
    assert row["accuracy_non_null"] == 1.6 / 3


def test_metrics_keep_development_and_heldout_separate():
    outcomes = pd.DataFrame(
        {
            "split": ["development", "heldout"],
            "outcome": ["correct_value", "wrong_value"],
            "include_primary": [True, True],
            "score": [1.0, 0.6],
        }
    )
    result = summarize_metrics(outcomes, ["split"]).set_index("split")
    assert result.loc["development", "accuracy_non_null"] == 1.0
    assert result.loc["heldout", "accuracy_non_null"] == 0.6
