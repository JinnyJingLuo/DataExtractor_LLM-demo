from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import summarize_metrics


AI_SCORE_COLUMNS = [
    "ai_score",
    "ai_reason",
    "ai_normalized_ground_truth",
    "ai_normalized_prediction",
    "ai_evaluation_source",
]


def build_evaluation_prompt(rows: pd.DataFrame) -> str:
    payload = []
    for row in rows.to_dict("records"):
        payload.append(
            {
                "row_id": int(row["row_id"]),
                "paper_id": str(row.get("paper_id", "")),
                "sample_id": str(row.get("sample_id", "")),
                "field_name": str(row.get("field_name", "")),
                "ground_truth": "" if pd.isna(row.get("ground_truth")) else str(row.get("ground_truth")),
                "prediction": "" if pd.isna(row.get("prediction")) else str(row.get("prediction")),
            }
        )
    compact_rows = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""You are evaluating numeric materials-data extraction results.

Use this rubric:
1.00 = Perfect Match: identical or <0.5% relative difference.
0.95 = Rounding: <2% relative difference, such as 0.43 vs 0.429.
0.90 = Format Diff only, such as 0.4 vs ~0.4 or equivalent scientific notation.
0.80 = Within Tolerance: <5% relative difference.
0.60 = Moderate: <10% relative difference.
0.30 = Major Deviation: <20% relative difference.
0.00 = Mismatch/Missing: >20% relative difference, missing, or not comparable.

Rules:
- Evaluate only the ground_truth and prediction values for the provided numeric field.
- Treat unit-equivalent numeric forms as equivalent only when the field name makes the unit clear.
- Treat uncertainty notation by comparing central values, e.g. 4.3±0.2 uses 4.3.
- Treat approximate notation, ranges, and scientific notation leniently according to the rubric.
- Treat blank, "-", "N/A", "not reported", and "null" as missing.
- If both values are missing/null, score 1.00.
- If one value is missing/null and the other is not, score 0.00.
- Do not reward a value that is clearly for a different physical quantity.

Return only JSON. Return a JSON array with one object per input row:
[
  {{
    "row_id": 123,
    "score": 0.95,
    "reason": "short reason",
    "normalized_ground_truth": "normalized numeric value or null",
    "normalized_prediction": "normalized numeric value or null"
  }}
]

Rows:
{compact_rows}
"""


def _extract_json_array(text: str) -> list[Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start < 0 or end < start:
            raise
        data = json.loads(stripped[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("AI evaluator response must be a JSON array")
    return data


def parse_ai_scores(text: str) -> list[dict[str, object]]:
    data = _extract_json_array(text)
    rows: list[dict[str, object]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("AI evaluator row must be an object")
        row_id = int(item["row_id"])
        score = float(item["score"])
        if score < 0 or score > 1:
            raise ValueError(f"AI score out of range for row_id={row_id}: {score}")
        rows.append(
            {
                "row_id": row_id,
                "ai_score": score,
                "ai_reason": str(item.get("reason", "")),
                "ai_normalized_ground_truth": str(
                    item.get("normalized_ground_truth", "")
                ),
                "ai_normalized_prediction": str(
                    item.get("normalized_prediction", "")
                ),
            }
        )
    return rows


def _needs_ai_review(row: pd.Series, mode: str) -> bool:
    if row.get("field_type") != "numeric":
        return False
    if mode == "all":
        return True
    if mode != "hybrid":
        raise ValueError(f"unsupported AI evaluation mode: {mode}")
    return row.get("outcome") == "wrong_value"


def hybrid_numeric_ai_evaluate(
    comparisons: pd.DataFrame,
    generate_text: Callable[[str], str],
    *,
    batch_size: int = 25,
    mode: str = "hybrid",
) -> pd.DataFrame:
    numeric = comparisons[comparisons["field_type"].eq("numeric")].copy()
    numeric = numeric.reset_index(drop=True)
    numeric["row_id"] = numeric.index
    numeric["ai_score"] = numeric["score"].fillna(0).astype(float)
    numeric["ai_reason"] = numeric["score_reason"].fillna("").astype(str)
    numeric["ai_normalized_ground_truth"] = numeric["ground_truth"].fillna("").astype(str)
    numeric["ai_normalized_prediction"] = numeric["prediction"].fillna("").astype(str)
    numeric["ai_evaluation_source"] = "deterministic"

    review = numeric[numeric.apply(lambda row: _needs_ai_review(row, mode), axis=1)].copy()
    ai_rows: list[dict[str, object]] = []
    total_review_rows = len(review)
    if total_review_rows:
        print(
            f"AI evaluation started: mode={mode}, rows={total_review_rows}, "
            f"batch_size={batch_size}",
            flush=True,
        )
    for start in range(0, len(review), batch_size):
        batch = review.iloc[start : start + batch_size]
        if batch.empty:
            continue
        batch_number = start // batch_size + 1
        completed_before = min(start, total_review_rows)
        print(
            f"AI evaluation batch {batch_number}: "
            f"starting rows {completed_before + 1}-"
            f"{min(start + len(batch), total_review_rows)} of {total_review_rows}",
            flush=True,
        )
        response_text = generate_text(build_evaluation_prompt(batch))
        ai_rows.extend(parse_ai_scores(response_text))
        print(
            f"AI evaluation progress: completed "
            f"{min(start + len(batch), total_review_rows)}/{total_review_rows} rows",
            flush=True,
        )

    if ai_rows:
        ai_frame = pd.DataFrame(ai_rows).set_index("row_id")
        for row_id, row in ai_frame.iterrows():
            mask = numeric["row_id"].eq(row_id)
            numeric.loc[mask, "ai_score"] = float(row["ai_score"])
            numeric.loc[mask, "ai_reason"] = row["ai_reason"]
            numeric.loc[mask, "ai_normalized_ground_truth"] = row[
                "ai_normalized_ground_truth"
            ]
            numeric.loc[mask, "ai_normalized_prediction"] = row[
                "ai_normalized_prediction"
            ]
            numeric.loc[mask, "ai_evaluation_source"] = "llm"

    numeric["ai_final_score"] = numeric["ai_score"].astype(float)
    return numeric.drop(columns=["row_id"])


def summarize_ai_evaluation(ai_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outcomes = ai_rows.copy()
    outcomes["score"] = outcomes["ai_final_score"]
    return {
        "overall_metrics": summarize_metrics(outcomes, ["split"]),
        "paper_metrics": summarize_metrics(outcomes, ["paper_id"]),
        "field_metrics": summarize_metrics(outcomes, ["field_name"]),
        "tier_metrics_unfiltered": summarize_metrics(
            outcomes[outcomes["tier"].astype(str).ne("NA")],
            ["tier"],
        ),
        "ai_source_metrics": summarize_metrics(outcomes, ["ai_evaluation_source"]),
    }


def write_ai_evaluation_outputs(ai_rows: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ai_rows.to_csv(output_dir / "ai_numeric_field_scores.csv", index=False)
    metrics = summarize_ai_evaluation(ai_rows)
    for name, frame in metrics.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    with pd.ExcelWriter(output_dir / "ai_numeric_evaluation.xlsx", engine="openpyxl") as writer:
        for name, frame in metrics.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        ai_rows.to_excel(writer, sheet_name="ai_numeric_field_scores", index=False)
