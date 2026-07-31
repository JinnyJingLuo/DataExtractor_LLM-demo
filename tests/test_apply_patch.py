import pandas as pd

from reconcile import ReconciliationResult
from run_figure_refine import apply_patch


def test_apply_patch_overwrites_only_high_confidence_and_preserves_evidence():
    extracted = pd.DataFrame(
        [
            {"Sample ID": "VA0.6-300", "Longitudinal Stress at HEL (GPa)": "1.0 (pass1 guess)"},
            {"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": "1.0 (pass1 guess)"},
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "Column Name": "Longitudinal Stress at HEL (GPa)",
                "Source Location": "Page 6, Fig. 5(a) & CALCULATED",
                "Notes": "[Priority 3/2]: original pass-1 evidence text",
            }
        ]
    )
    sample_results = {
        "VA0.6-300": ReconciliationResult(
            value=3.5015, confidence="high", majority_fraction=0.8,
            outliers=[{"value": 1.02, "evidence": "visually extracted, low confidence"}],
        ),
        "VA0-300": ReconciliationResult(
            value=None, confidence="low", majority_fraction=0.4, outliers=[],
        ),
    }

    apply_patch(extracted, evidence, "Longitudinal Stress at HEL (GPa)", "5", sample_results)

    assert extracted.loc[extracted["Sample ID"] == "VA0.6-300", "Longitudinal Stress at HEL (GPa)"].iloc[0] == 3.5015
    # low-confidence sample keeps its original Pass-1 value, untouched
    assert extracted.loc[extracted["Sample ID"] == "VA0-300", "Longitudinal Stress at HEL (GPa)"].iloc[0] == "1.0 (pass1 guess)"
    assert extracted.loc[extracted["Sample ID"] == "VA0-300", "Needs Review"].iloc[0] == "Longitudinal Stress at HEL (GPa)"

    updated_note = evidence.loc[evidence["Column Name"] == "Longitudinal Stress at HEL (GPa)", "Notes"].iloc[0]
    assert "original pass-1 evidence text" in updated_note  # preserved, not replaced
    assert "HIGH-RES REVIEW" in updated_note  # new info appended
