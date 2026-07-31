import json

import pandas as pd

from heldout_pipeline.reporting import build_experiment_overview, summarize_run_metadata


def test_summarizes_tokens_runtime_retries_and_failures(tmp_path):
    for split, paper, success in [
        ("development", "D1", True),
        ("heldout", "H1", False),
    ]:
        directory = tmp_path / split / paper
        directory.mkdir(parents=True)
        (directory / "response_metadata.json").write_text(
            json.dumps(
                {
                    "paper_id": paper,
                    "split": split,
                    "model_id": "gemini-test",
                    "success": success,
                    "input_tokens": 100 if success else None,
                    "output_tokens": 20 if success else None,
                    "total_tokens": 120 if success else None,
                    "duration_seconds": 3.5,
                    "retries": 1,
                    "estimated_cost_usd": None,
                }
            )
        )
    result = summarize_run_metadata(tmp_path)
    assert len(result) == 2
    assert result.loc[result["split"] == "heldout", "failure_rate"].iloc[0] == 1.0
    assert result.loc[result["split"] == "development", "total_tokens"].iloc[0] == 120


def test_overview_keeps_three_experiments_separate():
    api = pd.DataFrame(
        {
            "split": ["development", "heldout"],
            "papers": [30, 10],
            "successful_runs": [30, 9],
        }
    )
    result = build_experiment_overview(api, original_chatbox_papers=30)
    assert result["experiment"].tolist() == [
        "Original chatbox development evaluation",
        "API development evaluation",
        "API held-out evaluation",
    ]
