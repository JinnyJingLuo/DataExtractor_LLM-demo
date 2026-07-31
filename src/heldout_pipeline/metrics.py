from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


OUTCOMES = [
    "correct_value",
    "wrong_value",
    "missing_extraction",
    "false_extraction",
    "correct_null",
]

METRIC_COLUMNS = [
    *OUTCOMES,
    "support",
    "precision",
    "recall",
    "f1",
    "accuracy_all",
    "accuracy_non_null",
    "strict_accuracy_all",
    "strict_accuracy_non_null",
]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _summarize(group: pd.DataFrame) -> dict[str, float | int]:
    counts = group["outcome"].value_counts()
    values = {name: int(counts.get(name, 0)) for name in OUTCOMES}
    precision_denominator = (
        values["correct_value"] + values["wrong_value"] + values["false_extraction"]
    )
    recall_denominator = (
        values["correct_value"] + values["wrong_value"] + values["missing_extraction"]
    )
    precision = _ratio(values["correct_value"], precision_denominator)
    recall = _ratio(values["correct_value"], recall_denominator)
    f1 = _ratio(2 * precision * recall, precision + recall)
    support = sum(values.values())
    if "score" in group.columns:
        score = group["score"].fillna(0).astype(float)
        scored_non_null = group[
            group["outcome"].isin(
                ["correct_value", "wrong_value", "missing_extraction"]
            )
        ]["score"].fillna(0).astype(float)
        accuracy_all = _ratio(float(score.sum()), support)
        accuracy_non_null = _ratio(float(scored_non_null.sum()), recall_denominator)
    else:
        accuracy_all = _ratio(values["correct_value"] + values["correct_null"], support)
        accuracy_non_null = _ratio(values["correct_value"], recall_denominator)
    return {
        **values,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_all": accuracy_all,
        "accuracy_non_null": accuracy_non_null,
        "strict_accuracy_all": _ratio(
            values["correct_value"] + values["correct_null"],
            support,
        ),
        "strict_accuracy_non_null": _ratio(
            values["correct_value"],
            recall_denominator,
        ),
    }


def summarize_metrics(outcomes: pd.DataFrame, group_by: Sequence[str]) -> pd.DataFrame:
    primary = outcomes[outcomes["include_primary"].astype(bool)].copy()
    rows = []
    if group_by:
        if primary.empty:
            return pd.DataFrame(columns=[*group_by, *METRIC_COLUMNS])
        grouper = group_by[0] if len(group_by) == 1 else list(group_by)
        for key, group in primary.groupby(grouper, dropna=False, sort=True):
            keys = (key,) if len(group_by) == 1 else key
            row = dict(zip(group_by, keys))
            row.update(_summarize(group))
            rows.append(row)
    else:
        rows.append(_summarize(primary))
    return pd.DataFrame(rows)
