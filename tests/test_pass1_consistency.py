import json

import pandas as pd
import pytest

from run_figure_refine import summarize_pass1_consistency


def test_pass1_consistency_reports_agreement_fractions():
    first = pd.DataFrame(
        [
            {"Sample ID": "S1", "Spall Strength (GPa)": 1.0, "Treatment": "Annealed"},
            {"Sample ID": "S2", "Spall Strength (GPa)": 2.0, "Treatment": "Annealed"},
        ]
    )
    second = pd.DataFrame(
        [
            {"Sample ID": "S1", "Spall Strength (GPa)": 1.0, "Treatment": "Annealed"},
            {"Sample ID": "S2", "Spall Strength (GPa)": 2.2, "Treatment": "Annealed"},
        ]
    )
    third = pd.DataFrame(
        [
            {"Sample ID": "S1", "Spall Strength (GPa)": 1.0, "Treatment": "Annealed"},
            {"Sample ID": "S2", "Spall Strength (GPa)": 2.0, "Treatment": "Aged"},
        ]
    )

    rows = summarize_pass1_consistency(first, [first, second, third])
    s1_spall = rows[
        rows["sample_id"].eq("S1") & rows["field_name"].eq("Spall Strength (GPa)")
    ].iloc[0]
    s2_spall = rows[
        rows["sample_id"].eq("S2") & rows["field_name"].eq("Spall Strength (GPa)")
    ].iloc[0]
    s2_treatment = rows[
        rows["sample_id"].eq("S2") & rows["field_name"].eq("Treatment")
    ].iloc[0]

    assert s1_spall["agreement_fraction"] == pytest.approx(1.0)
    assert s1_spall["confidence"] == "high"
    assert s2_spall["agreement_fraction"] == pytest.approx(2 / 3)
    assert s2_spall["confidence"] == "medium"
    assert s2_treatment["agreement_fraction"] == pytest.approx(2 / 3)
    assert json.loads(s2_treatment["observed_values_json"]) == [
        "Annealed",
        "Annealed",
        "Aged",
    ]


def test_pass1_consistency_matches_samples_across_hyphen_underscore_variants():
    # Observed on a real Paper12 run: draw 1 wrote "VA0_300", draws 2 and 3
    # wrote "VA0-300" -- same sample, same value, different separator. Before
    # normalization this made every sample look "missing" from every other
    # draw and reported false low-confidence for the entire table.
    first = pd.DataFrame([{"Sample ID": "VA0_300", "Longitudinal Stress at HEL (GPa)": 3.349}])
    second = pd.DataFrame([{"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": 3.349}])
    third = pd.DataFrame([{"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": 3.349}])

    rows = summarize_pass1_consistency(first, [first, second, third])
    row = rows.iloc[0]

    assert row["sample_id"] == "VA0_300"  # reports the original, unnormalized form
    assert row["compared_draws"] == 3
    assert row["agreement_count"] == 3
    assert row["confidence"] == "high"
    assert "<MISSING_SAMPLE>" not in json.loads(row["observed_values_json"])


def test_pass1_consistency_treats_missing_records_as_low_confidence():
    first = pd.DataFrame([{"Sample ID": "S1", "Spall Strength (GPa)": 1.0}])
    second = pd.DataFrame([{"Sample ID": "S2", "Spall Strength (GPa)": 1.0}])

    rows = summarize_pass1_consistency(first, [first, second])
    row = rows.iloc[0]

    assert row["agreement_count"] == 1
    assert row["compared_draws"] == 2
    assert row["agreement_fraction"] == pytest.approx(0.5)
    assert row["confidence"] == "medium"
    assert json.loads(row["observed_values_json"]) == ["1.0", "<MISSING_SAMPLE>"]


def test_pass1_consistency_treats_numeric_formatting_as_agreement():
    first = pd.DataFrame([{"Sample ID": "S1", "Value": "1"}])
    second = pd.DataFrame([{"Sample ID": "S1", "Value": "1.0"}])
    third = pd.DataFrame([{"Sample ID": "S1", "Value": "1.004"}])

    rows = summarize_pass1_consistency(first, [first, second, third])
    row = rows.iloc[0]

    assert row["agreement_count"] == 3
    assert row["agreement_fraction"] == pytest.approx(1.0)
    assert row["confidence"] == "high"


def test_pass1_consistency_normalizes_text_case_and_spaces():
    first = pd.DataFrame([{"Sample ID": "S1", "Treatment": "Annealed"}])
    second = pd.DataFrame([{"Sample ID": "S1", "Treatment": " annealed "}])

    rows = summarize_pass1_consistency(first, [first, second])
    row = rows.iloc[0]

    assert row["agreement_count"] == 2
    assert row["confidence"] == "high"
