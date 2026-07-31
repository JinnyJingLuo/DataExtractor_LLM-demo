from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def summarize_run_metadata(artifact_root: Path) -> pd.DataFrame:
    rows = []
    for path in Path(artifact_root).glob("*/*/response_metadata.json"):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        return pd.DataFrame(
            columns=[
                "split",
                "model_id",
                "papers",
                "successful_runs",
                "failed_runs",
                "failure_rate",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "duration_seconds",
                "retries",
                "estimated_cost_usd",
            ]
        )
    frame = pd.DataFrame(rows)
    summaries = []
    for (split, model_id), group in frame.groupby(["split", "model_id"], dropna=False):
        successes = int(group["success"].fillna(False).astype(bool).sum())
        papers = len(group)
        summaries.append(
            {
                "split": split,
                "model_id": model_id,
                "papers": papers,
                "successful_runs": successes,
                "failed_runs": papers - successes,
                "failure_rate": (papers - successes) / papers if papers else 0.0,
                "input_tokens": group.get("input_tokens", pd.Series(dtype=float)).sum(
                    min_count=1
                ),
                "output_tokens": group.get("output_tokens", pd.Series(dtype=float)).sum(
                    min_count=1
                ),
                "total_tokens": group.get("total_tokens", pd.Series(dtype=float)).sum(
                    min_count=1
                ),
                "duration_seconds": group["duration_seconds"].sum(),
                "retries": group["retries"].sum(),
                "estimated_cost_usd": group.get(
                    "estimated_cost_usd", pd.Series(dtype=float)
                ).sum(min_count=1),
            }
        )
    return pd.DataFrame(summaries).sort_values("split").reset_index(drop=True)


def build_experiment_overview(
    api_summary: pd.DataFrame,
    original_chatbox_papers: int = 30,
) -> pd.DataFrame:
    rows = [
        {
            "experiment": "Original chatbox development evaluation",
            "split": "development",
            "interface": "chatbox",
            "papers": original_chatbox_papers,
            "successful_runs": original_chatbox_papers,
        }
    ]
    for split, label in [
        ("development", "API development evaluation"),
        ("heldout", "API held-out evaluation"),
    ]:
        selected = api_summary[api_summary["split"] == split]
        rows.append(
            {
                "experiment": label,
                "split": split,
                "interface": "API",
                "papers": int(selected["papers"].sum()) if not selected.empty else 0,
                "successful_runs": int(selected["successful_runs"].sum())
                if not selected.empty
                else 0,
            }
        )
    return pd.DataFrame(rows)
