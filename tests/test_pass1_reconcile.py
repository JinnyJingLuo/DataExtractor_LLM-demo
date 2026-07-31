import pandas as pd
import pytest

from pass1_reconcile import (
    classify_evidence_tier,
    reconcile_evidence_tiers,
    reconcile_pass1_table,
)


def test_reconcile_pass1_table_numeric_field_uses_cluster_and_reconcile():
    # Real Paper12 v3 data: draws 1-2 agree on 3.349, draw 3 is an outlier.
    selected = pd.DataFrame([{"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": "3.349"}])
    draws = [
        pd.DataFrame([{"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": "3.349"}]),
        pd.DataFrame([{"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": "3.349"}]),
        pd.DataFrame([{"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": "2.900"}]),
    ]
    reconciled, audit = reconcile_pass1_table(selected, draws)
    assert reconciled.loc[0, "Longitudinal Stress at HEL (GPa)"] == pytest.approx(3.349)
    row = audit.iloc[0]
    assert row["confidence"] == "high"
    assert row["majority_fraction"] == pytest.approx(2 / 3)


def test_reconcile_pass1_table_overrides_selected_draw_when_it_was_the_outlier():
    # Selected draw (draw #1) happened to be the bad one this time -- the
    # reconciled table must override it with the majority value, not keep it.
    selected = pd.DataFrame([{"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": "2.900"}])
    draws = [
        pd.DataFrame([{"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": "2.900"}]),
        pd.DataFrame([{"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": "3.349"}]),
        pd.DataFrame([{"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": "3.349"}]),
    ]
    reconciled, _ = reconcile_pass1_table(selected, draws)
    assert reconciled.loc[0, "Longitudinal Stress at HEL (GPa)"] == pytest.approx(3.349)


def test_reconcile_pass1_table_text_field_uses_majority_vote():
    selected = pd.DataFrame([{"Sample ID": "S2", "Treatment": "Annealed"}])
    draws = [
        pd.DataFrame([{"Sample ID": "S2", "Treatment": "Annealed"}]),
        pd.DataFrame([{"Sample ID": "S2", "Treatment": "Annealed"}]),
        pd.DataFrame([{"Sample ID": "S2", "Treatment": "Aged"}]),
    ]
    reconciled, audit = reconcile_pass1_table(selected, draws)
    assert reconciled.loc[0, "Treatment"] == "Annealed"
    assert audit.iloc[0]["confidence"] == "high"


def test_reconcile_pass1_table_single_draw_degenerates_to_that_value():
    selected = pd.DataFrame([{"Sample ID": "S1", "Hardness": "10.472"}])
    reconciled, audit = reconcile_pass1_table(selected, [selected])
    assert reconciled.loc[0, "Hardness"] == pytest.approx(10.472)
    assert audit.iloc[0]["confidence"] == "high"


def test_reconcile_pass1_table_matches_hyphen_underscore_sample_id_variants():
    selected = pd.DataFrame([{"Sample ID": "VA0_300", "Hardness": "10.0"}])
    draws = [
        pd.DataFrame([{"Sample ID": "VA0_300", "Hardness": "10.0"}]),
        pd.DataFrame([{"Sample ID": "VA0-300", "Hardness": "10.0"}]),
        pd.DataFrame([{"Sample ID": "VA0-300", "Hardness": "10.0"}]),
    ]
    reconciled, audit = reconcile_pass1_table(selected, draws)
    assert reconciled.loc[0, "Sample ID"] == "VA0_300"  # original form preserved
    assert audit.iloc[0]["confidence"] == "high"
    assert audit.iloc[0]["majority_fraction"] == pytest.approx(1.0)


def test_reconcile_pass1_table_flags_low_confidence_in_needs_review():
    selected = pd.DataFrame([{"Sample ID": "S1", "Spall Strength (GPa)": "5.0"}])
    draws = [
        pd.DataFrame([{"Sample ID": "S1", "Spall Strength (GPa)": "5.0"}]),
        pd.DataFrame([{"Sample ID": "S1", "Spall Strength (GPa)": "1.0"}]),
        pd.DataFrame([{"Sample ID": "S1", "Spall Strength (GPa)": "9.0"}]),
    ]
    reconciled, audit = reconcile_pass1_table(selected, draws)
    assert audit.iloc[0]["confidence"] == "low"
    assert "Spall Strength (GPa)" in reconciled.loc[0, "Needs Review"]


def test_reconcile_pass1_table_abstention_lowers_confidence():
    # Only 1 of 3 draws found a value at all -- should not read as "high"
    # confidence just because the lone value has no competing number.
    selected = pd.DataFrame([{"Sample ID": "S1", "Longitudinal Stress at HEL (GPa)": "1.05"}])
    draws = [
        pd.DataFrame([{"Sample ID": "S1", "Longitudinal Stress at HEL (GPa)": "1.05"}]),
        pd.DataFrame([{"Sample ID": "S1", "Longitudinal Stress at HEL (GPa)": "-"}]),
        pd.DataFrame([{"Sample ID": "S1", "Longitudinal Stress at HEL (GPa)": "-"}]),
    ]
    _, audit = reconcile_pass1_table(selected, draws)
    assert audit.iloc[0]["confidence"] == "low"
    assert audit.iloc[0]["majority_fraction"] == pytest.approx(1 / 3)


def test_classify_evidence_tier_direct():
    assert classify_evidence_tier("[Priority 1]: stated in Table I", "Table I") == "direct"


def test_classify_evidence_tier_dual_path():
    assert (
        classify_evidence_tier(
            "[Priority 3/2]: For VA0, figure. For others, calculated", "Page 6, Fig. 5(a) & CALCULATED"
        )
        == "dual-path"
    )


def test_classify_evidence_tier_calculated_only_when_no_figure_mentioned():
    assert classify_evidence_tier("[Priority 2]: E = 9BG / (3B+G)", "CALCULATED") == "calculated-only"


def test_reconcile_evidence_tiers_majority_outvotes_circular_draw():
    # Exact real scenario: 2 draws say the value is directly stated, 1 draw
    # (the buggy one) says it was calculated (circularly) from the very
    # field being computed. Majority should win, keeping the field out of
    # the T2 gate entirely rather than letting one bad draw route it there.
    draws = [
        pd.DataFrame(
            [{"Column Name": "Free Surface Velocity at HEL (m/s)", "Notes": "[Priority 1]: explicit precursor plateau value", "Source Location": "Page 4"}]
        ),
        pd.DataFrame(
            [{"Column Name": "Free Surface Velocity at HEL (m/s)", "Notes": "[Priority 2]: calculated from sigma_HEL using u_HEL = sigma_HEL / (0.5*rho0*C_l)", "Source Location": "CALCULATED"}]
        ),
        pd.DataFrame(
            [{"Column Name": "Free Surface Velocity at HEL (m/s)", "Notes": "[Priority 1]: stated in Section III.B", "Source Location": "Page 4"}]
        ),
    ]
    result = reconcile_evidence_tiers(draws)
    field = result["Free Surface Velocity at HEL (m/s)"]
    assert field["tier"] == "direct"
    assert field["confidence"] == "high"
    assert field["fraction"] == pytest.approx(2 / 3)


def test_reconcile_evidence_tiers_true_tie_is_low_confidence():
    draws = [
        pd.DataFrame([{"Column Name": "X", "Notes": "[Priority 1]: stated directly", "Source Location": "Page 1"}]),
        pd.DataFrame([{"Column Name": "X", "Notes": "[Priority 2]: calculated", "Source Location": "CALCULATED"}]),
        pd.DataFrame([{"Column Name": "X", "Notes": "[Priority 3]: read from figure", "Source Location": "Fig. 5"}]),
    ]
    result = reconcile_evidence_tiers(draws)
    assert result["X"]["tier"] is None
    assert result["X"]["confidence"] == "low"


def test_reconcile_evidence_tiers_extracts_majority_figure_number():
    draws = [
        pd.DataFrame([{"Column Name": "Y", "Notes": "[Priority 3]: read from figure", "Source Location": "Fig. 5"}]),
        pd.DataFrame([{"Column Name": "Y", "Notes": "[Priority 3]: read from figure", "Source Location": "Fig. 5"}]),
        pd.DataFrame([{"Column Name": "Y", "Notes": "[Priority 3]: read from figure", "Source Location": "Fig. 4"}]),
    ]
    result = reconcile_evidence_tiers(draws)
    assert result["Y"]["tier"] == "figure-only"
    assert result["Y"]["figure_number"] == "5"


def test_reconcile_evidence_tiers_figure_number_scoped_to_winning_tier():
    # Real Paper12 case for "Spall Pullback Velocity": 3 of 5 draws say
    # calculated-only (no figure at all -- that's what "calculated-only"
    # means by construction), but 2 minority draws (figure-only, dual-path)
    # each happen to mention "Fig 3". The old unscoped pooling attached
    # figure_number="3" to the calculated-only-tier result anyway, which
    # later made the T2-gate treat a genuinely figure-less field as if it
    # had a figure to fall back to.
    draws = [
        pd.DataFrame([{
            "Column Name": "Spall Pullback Velocity (m/s)",
            "Notes": "FIGURE (P3): evaluated from pullback waveforms",
            "Source Location": "Page 5, Figure 3a-c",
        }]),
        pd.DataFrame([{
            "Column Name": "Spall Pullback Velocity (m/s)",
            "Notes": "Priority 2 (mostly calculated): for VA0-300, visually extracted from Fig 3(a)",
            "Source Location": "Calculated",
        }]),
        pd.DataFrame([{
            "Column Name": "Spall Pullback Velocity (m/s)",
            "Notes": "Priority 2: calculated using visually extracted spall strengths",
            "Source Location": "Calculated",
        }]),
        pd.DataFrame([{
            "Column Name": "Spall Pullback Velocity (m/s)",
            "Notes": "CALCULATED (P2): formula applied",
            "Source Location": "CALCULATED",
        }]),
        pd.DataFrame([{
            "Column Name": "Spall Pullback Velocity (m/s)",
            "Notes": "CALCULATED (P2): formula applied",
            "Source Location": "CALCULATED",
        }]),
    ]
    result = reconcile_evidence_tiers(draws)
    field = result["Spall Pullback Velocity (m/s)"]
    assert field["tier"] == "calculated-only"
    assert field["figure_number"] is None
