from pathlib import Path

import pandas as pd
import pytest

from heldout_pipeline.normalize import (
    NormalizationError,
    load_ground_truth,
    normalize_sample_id,
    wide_to_long,
)


def test_loads_existing_ground_truth_excel_layout(tmp_path):
    path = tmp_path / "ground_truth.xlsx"
    frame = pd.DataFrame(
        {
            "sheet": ["Paper1", "Heldout001"],
            "Sample ID": ["A-1", "B_1"],
            "Treatment": ["Annealed", "-"],
            "Value": [10.0, 20.0],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, startrow=1)

    result = load_ground_truth(path, {"Heldout001"})

    assert set(result["paper_id"]) == {"Heldout001"}
    assert set(result["field_name"]) == {"Treatment", "Value"}


def test_normalize_sample_id_unifies_separators():
    assert normalize_sample_id(" Mo- 1 ") == "mo_1"
    assert normalize_sample_id("Mo_1") == "mo_1"


def test_wide_to_long_rejects_duplicate_keys():
    frame = pd.DataFrame(
        {
            "paper": ["P1", "P1"],
            "Sample ID": ["A-1", "A_1"],
            "Value": [1, 2],
        }
    )
    with pytest.raises(NormalizationError, match="duplicate"):
        wide_to_long(frame, "paper", "Sample ID")


def test_wide_to_long_preserves_extra_prediction_sample():
    frame = pd.DataFrame(
        {"paper": ["P1", "P1"], "Sample ID": ["A", "EXTRA"], "Value": [1, 2]}
    )
    result = wide_to_long(frame, "paper", "Sample ID")
    assert set(result["sample_id"]) == {"a", "extra"}
