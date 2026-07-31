import json

import pandas as pd

from heldout_pipeline.ai_evaluator import (
    build_evaluation_prompt,
    hybrid_numeric_ai_evaluate,
    parse_ai_scores,
)


def test_build_evaluation_prompt_contains_rubric_and_rows():
    rows = pd.DataFrame(
        [
            {
                "row_id": 7,
                "paper_id": "P1",
                "sample_id": "S1",
                "field_name": "Impact Velocity (m/s)",
                "ground_truth": "100",
                "prediction": "~101",
            }
        ]
    )

    prompt = build_evaluation_prompt(rows)

    assert "0.95" in prompt
    assert "0.90" in prompt
    assert "Return only JSON" in prompt
    assert '"row_id":7' in prompt


def test_parse_ai_scores_accepts_json_array():
    text = json.dumps(
        [
            {
                "row_id": 1,
                "score": 0.95,
                "reason": "rounding",
                "normalized_ground_truth": "100",
                "normalized_prediction": "101",
            }
        ]
    )

    parsed = parse_ai_scores(text)

    assert parsed == [
        {
            "row_id": 1,
            "ai_score": 0.95,
            "ai_reason": "rounding",
            "ai_normalized_ground_truth": "100",
            "ai_normalized_prediction": "101",
        }
    ]


def test_hybrid_numeric_ai_evaluate_only_sends_ambiguous_numeric_wrong_values():
    comparisons = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "sample_id": "S1",
                "field_name": "Metal Symbol",
                "field_type": "categorical",
                "ground_truth": "Cu",
                "prediction": "Al",
                "outcome": "wrong_value",
                "score": 0.0,
                "score_reason": "text_mismatch",
            },
            {
                "paper_id": "P1",
                "sample_id": "S1",
                "field_name": "Impact Velocity (m/s)",
                "field_type": "numeric",
                "ground_truth": "100",
                "prediction": "~101",
                "outcome": "wrong_value",
                "score": 0.0,
                "score_reason": "mismatch",
            },
            {
                "paper_id": "P1",
                "sample_id": "S1",
                "field_name": "Spall Strength (GPa)",
                "field_type": "numeric",
                "ground_truth": "-",
                "prediction": "-",
                "outcome": "correct_null",
                "score": 1.0,
                "score_reason": "both_missing",
            },
        ]
    )
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return json.dumps(
            [
                {
                    "row_id": 0,
                    "score": 0.95,
                    "reason": "rounding",
                    "normalized_ground_truth": "100",
                    "normalized_prediction": "101",
                }
            ]
        )

    result = hybrid_numeric_ai_evaluate(comparisons, fake_generate, batch_size=10)

    assert len(prompts) == 1
    assert result["ai_final_score"].tolist() == [0.95, 1.0]
    assert result["ai_evaluation_source"].tolist() == ["llm", "deterministic"]


def test_all_mode_sends_every_numeric_row_to_ai():
    comparisons = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "sample_id": "S1",
                "field_name": "Impact Velocity (m/s)",
                "field_type": "numeric",
                "ground_truth": "100",
                "prediction": "100",
                "outcome": "correct_value",
                "score": 1.0,
                "score_reason": "perfect_under_0.5pct",
            },
            {
                "paper_id": "P1",
                "sample_id": "S1",
                "field_name": "Spall Strength (GPa)",
                "field_type": "numeric",
                "ground_truth": "-",
                "prediction": "-",
                "outcome": "correct_null",
                "score": 1.0,
                "score_reason": "both_missing",
            },
        ]
    )
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return json.dumps(
            [
                {
                    "row_id": 0,
                    "score": 1.0,
                    "reason": "exact",
                    "normalized_ground_truth": "100",
                    "normalized_prediction": "100",
                },
                {
                    "row_id": 1,
                    "score": 1.0,
                    "reason": "both missing",
                    "normalized_ground_truth": "null",
                    "normalized_prediction": "null",
                },
            ]
        )

    result = hybrid_numeric_ai_evaluate(
        comparisons,
        fake_generate,
        batch_size=10,
        mode="all",
    )

    assert len(prompts) == 1
    assert result["ai_evaluation_source"].tolist() == ["llm", "llm"]
