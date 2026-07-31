import pandas as pd

from run_figure_refine import filter_t1_resolved_fields, find_direct_fields


def test_find_direct_fields_detects_priority1_evidence():
    evidence_rows = [
        {
            "Column Name": "Impact Velocity (m/s)",
            "Source Location": "Table 1",
            "Notes": "DIRECT (P1): tabulated in the paper",
        },
        {
            "Column Name": "Spall Strength (GPa)",
            "Source Location": "Figure 5",
            "Notes": "FIGURE (P3): read from plot",
        },
    ]

    assert find_direct_fields(evidence_rows) == {"Impact Velocity (m/s)"}


def test_t1_gate_removes_direct_field_only_when_all_values_present():
    figure_fields = {
        "Impact Velocity (m/s)": "2",
        "Spall Strength (GPa)": "5",
    }
    evidence_rows = [
        {
            "Column Name": "Impact Velocity (m/s)",
            "Source Location": "Table 1 and Figure 2",
            "Notes": "DIRECT (P1): tabulated; FIGURE (P3): also plotted",
        },
        {
            "Column Name": "Spall Strength (GPa)",
            "Source Location": "Figure 5",
            "Notes": "FIGURE (P3): read from plot",
        },
    ]
    extracted = pd.DataFrame(
        [
            {"Sample ID": "S1", "Impact Velocity (m/s)": 300, "Spall Strength (GPa)": 1.0},
            {"Sample ID": "S2", "Impact Velocity (m/s)": 400, "Spall Strength (GPa)": 1.2},
        ]
    )

    filtered, resolved = filter_t1_resolved_fields(figure_fields, evidence_rows, extracted)

    assert filtered == {"Spall Strength (GPa)": "5"}
    assert resolved == {"Impact Velocity (m/s)"}


def test_t1_gate_keeps_direct_field_when_some_values_missing():
    figure_fields = {"Impact Velocity (m/s)": "2"}
    evidence_rows = [
        {
            "Column Name": "Impact Velocity (m/s)",
            "Source Location": "Table 1 and Figure 2",
            "Notes": "DIRECT (P1): tabulated; FIGURE (P3): also plotted",
        }
    ]
    extracted = pd.DataFrame(
        [
            {"Sample ID": "S1", "Impact Velocity (m/s)": 300},
            {"Sample ID": "S2", "Impact Velocity (m/s)": "-"},
        ]
    )

    filtered, resolved = filter_t1_resolved_fields(figure_fields, evidence_rows, extracted)

    assert filtered == {"Impact Velocity (m/s)": "2"}
    assert resolved == set()
