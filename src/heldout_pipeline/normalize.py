from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


class NormalizationError(ValueError):
    pass


KEY_COLUMNS = ["paper_id", "sample_id", "field_name"]


def normalize_sample_id(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[\s\-_]+", "_", str(value).strip().casefold()).strip("_")


def wide_to_long(frame: pd.DataFrame, paper_col: str, sample_col: str) -> pd.DataFrame:
    missing = {paper_col, sample_col} - set(frame.columns)
    if missing:
        raise NormalizationError(f"missing identifier columns: {sorted(missing)}")
    value_columns = [column for column in frame.columns if column not in {paper_col, sample_col}]
    result = frame.melt(
        id_vars=[paper_col, sample_col],
        value_vars=value_columns,
        var_name="field_name",
        value_name="value",
    ).rename(columns={paper_col: "paper_id", sample_col: "sample_id"})
    result["paper_id"] = result["paper_id"].astype(str).str.strip()
    result["sample_id"] = result["sample_id"].map(normalize_sample_id)
    result["field_name"] = result["field_name"].astype(str).str.strip()
    duplicates = result.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        keys = result.loc[duplicates, KEY_COLUMNS].drop_duplicates().to_dict("records")
        raise NormalizationError(f"duplicate normalized keys: {keys[:5]}")
    return result


def load_ground_truth(path: Path, allowed_papers: set[str]) -> pd.DataFrame:
    frame = pd.read_excel(path, header=1)
    if "sheet" not in frame.columns or "Sample ID" not in frame.columns:
        raise NormalizationError("ground truth requires 'sheet' and 'Sample ID' columns")
    frame["sheet"] = frame["sheet"].astype(str).str.strip()
    frame = frame[frame["sheet"].isin(allowed_papers)].copy()
    return wide_to_long(frame, "sheet", "Sample ID")


def load_predictions(artifact_root: Path, records) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in records:
        path = Path(artifact_root) / record.split / record.paper_id / "extracted_data.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "paper_id", record.paper_id)
        frames.append(wide_to_long(frame, "paper_id", "Sample ID"))
    if not frames:
        return pd.DataFrame(columns=KEY_COLUMNS + ["value"])
    combined = pd.concat(frames, ignore_index=True)
    duplicates = combined.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        raise NormalizationError("duplicate prediction keys across artifacts")
    return combined
