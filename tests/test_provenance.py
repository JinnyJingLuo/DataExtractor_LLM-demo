import pandas as pd

from heldout_pipeline.provenance import prepare_provenance


def predictions():
    return pd.DataFrame(
        [
            {"paper_id": "P1", "sample_id": "a", "field_name": "Treatment", "value": "Annealed"},
            {"paper_id": "P1", "sample_id": "a", "field_name": "Initial Density (g/cm3)", "value": 2.7},
            {"paper_id": "P1", "sample_id": "a", "field_name": "Unknown", "value": 3},
        ]
    )


def test_prepares_tiers_without_defaulting_unknown_to_t1():
    evidence = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "field_name": "Treatment",
                "source_location": "Table 1",
                "notes": "DIRECT (P1): reported",
            }
        ]
    )

    result = prepare_provenance(
        predictions(),
        evidence,
        external_reference_fields={"Initial Density (g/cm3)"},
    ).set_index("field_name")

    assert result.loc["Treatment", "tier"] == "T1"
    assert result.loc["Unknown", "tier"] == "NA"
    assert result.loc["Unknown", "provenance"] == "unknown"
    assert result.loc["Initial Density (g/cm3)", "provenance"] == "external_reference"


def test_parses_equation_and_figure_evidence():
    evidence = pd.DataFrame(
        [
            {"paper_id": "P1", "field_name": "A", "source_location": "Eq. 2", "notes": "CALCULATED (P2)"},
            {"paper_id": "P1", "field_name": "B", "source_location": "Fig. 4", "notes": "FIGURE (P3)"},
        ]
    )
    pred = pd.DataFrame(
        [
            {"paper_id": "P1", "sample_id": "s", "field_name": "A", "value": 1},
            {"paper_id": "P1", "sample_id": "s", "field_name": "B", "value": 2},
        ]
    )
    result = prepare_provenance(pred, evidence, set()).set_index("field_name")
    assert result.loc["A", "provenance"] == "derived_from_paper"
    assert result.loc["B", "provenance"] == "paper_figure"


def test_directly_reported_density_remains_paper_extraction():
    pred = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "sample_id": "s",
                "field_name": "Initial Density (g/cm3)",
                "value": 2.7,
            }
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "field_name": "Initial Density (g/cm3)",
                "source_location": "Table 1",
                "notes": "DIRECT (P1): density reported by the paper",
            }
        ]
    )
    row = prepare_provenance(
        pred, evidence, {"Initial Density (g/cm3)"}
    ).iloc[0]
    assert row["tier"] == "T1"
    assert row["provenance"] == "paper_table"
