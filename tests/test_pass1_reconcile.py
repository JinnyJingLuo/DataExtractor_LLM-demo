import pandas as pd
import pytest

from pass1_reconcile import (
    align_draws_by_position,
    classify_evidence_tier,
    reconcile_evidence_tiers,
    reconcile_pass1_table,
)


def _audit_row(audit: pd.DataFrame, field_name: str) -> pd.Series:
    # "Sample ID" is reconciled first and gets its own audit row now too, so
    # a raw positional .iloc[0] no longer reliably means "the field under
    # test" -- look up by name instead.
    return audit[audit["field_name"] == field_name].iloc[0]


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
    row = _audit_row(audit, "Longitudinal Stress at HEL (GPa)")
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
    assert _audit_row(audit, "Treatment")["confidence"] == "high"


def test_reconcile_pass1_table_single_draw_degenerates_to_that_value():
    selected = pd.DataFrame([{"Sample ID": "S1", "Hardness": "10.472"}])
    reconciled, audit = reconcile_pass1_table(selected, [selected])
    assert reconciled.loc[0, "Hardness"] == pytest.approx(10.472)
    assert _audit_row(audit, "Hardness")["confidence"] == "high"


def test_reconcile_pass1_table_sample_id_reconciled_via_majority_vote_like_any_field():
    # Sample ID is no longer a special join key with format normalization --
    # rows are aligned by position now (see the position-based tests below),
    # so Sample ID text itself is majority-voted across draws too (exact
    # match, no "-"/"_" normalization): 2 of 3 draws say "VA0-300", 1 says
    # "VA0_300" -- the literal majority wins. But it's excluded from
    # audit_rows -- it's an alignment key, not a measurement, so its own
    # agreement shouldn't count toward confidence/accuracy stats.
    selected = pd.DataFrame([{"Sample ID": "VA0_300", "Hardness": "10.0"}])
    draws = [
        pd.DataFrame([{"Sample ID": "VA0_300", "Hardness": "10.0"}]),
        pd.DataFrame([{"Sample ID": "VA0-300", "Hardness": "10.0"}]),
        pd.DataFrame([{"Sample ID": "VA0-300", "Hardness": "10.0"}]),
    ]
    reconciled, audit = reconcile_pass1_table(selected, draws)
    assert reconciled.loc[0, "Sample ID"] == "VA0-300"
    assert "Sample ID" not in audit["field_name"].values


def test_reconcile_pass1_table_aligns_rows_by_position_despite_different_id_text():
    # Real Paper13 shape: draws 1-2 name samples "1","2",... while draw 3
    # names them "Expt 1","Expt 2",... for the exact same experiments in the
    # exact same order. Position-based alignment must still reconcile all 3
    # draws together (not silently drop draw 3 because its ID text never
    # matched), and Sample ID itself should reflect the majority label.
    selected = pd.DataFrame([
        {"Sample ID": "1", "Impact Velocity (m/s)": "449"},
        {"Sample ID": "2", "Impact Velocity (m/s)": "502"},
    ])
    draws = [
        pd.DataFrame([
            {"Sample ID": "1", "Impact Velocity (m/s)": "449"},
            {"Sample ID": "2", "Impact Velocity (m/s)": "502"},
        ]),
        pd.DataFrame([
            {"Sample ID": "1", "Impact Velocity (m/s)": "449"},
            {"Sample ID": "2", "Impact Velocity (m/s)": "502"},
        ]),
        pd.DataFrame([
            {"Sample ID": "Expt 1", "Impact Velocity (m/s)": "449"},
            {"Sample ID": "Expt 2", "Impact Velocity (m/s)": "502"},
        ]),
    ]
    reconciled, audit = reconcile_pass1_table(selected, draws)
    # All 3 draws agreed on the velocity at each position -- confirms draw 3
    # wasn't silently dropped from the comparison.
    row0 = _audit_row(audit[audit["sample_id"] == "1"], "Impact Velocity (m/s)")
    assert row0["majority_fraction"] == pytest.approx(1.0)
    assert row0["confidence"] == "high"
    # Sample ID majority-votes to "1" (2 of 3 draws), not "Expt 1".
    assert reconciled.loc[0, "Sample ID"] == "1"
    assert reconciled.loc[1, "Sample ID"] == "2"


def test_align_draws_by_position_excludes_draws_with_a_different_row_count():
    # A draw with a different row count can't be safely assumed to list the
    # same shots in the same order (it may have dropped/added a row) --
    # must be excluded (None) rather than risk misaligning rows.
    selected = pd.DataFrame([{"Sample ID": "1"}, {"Sample ID": "2"}])
    same_count = pd.DataFrame([{"Sample ID": "1"}, {"Sample ID": "2"}])
    fewer_rows = pd.DataFrame([{"Sample ID": "1"}])
    aligned = align_draws_by_position(selected, [same_count, fewer_rows])
    assert aligned[0] is not None
    assert aligned[1] is None


def test_reconcile_pass1_table_flags_low_confidence_in_needs_review():
    selected = pd.DataFrame([{"Sample ID": "S1", "Spall Strength (GPa)": "5.0"}])
    draws = [
        pd.DataFrame([{"Sample ID": "S1", "Spall Strength (GPa)": "5.0"}]),
        pd.DataFrame([{"Sample ID": "S1", "Spall Strength (GPa)": "1.0"}]),
        pd.DataFrame([{"Sample ID": "S1", "Spall Strength (GPa)": "9.0"}]),
    ]
    reconciled, audit = reconcile_pass1_table(selected, draws)
    assert _audit_row(audit, "Spall Strength (GPa)")["confidence"] == "low"
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
    row = _audit_row(audit, "Longitudinal Stress at HEL (GPa)")
    assert row["confidence"] == "low"
    assert row["majority_fraction"] == pytest.approx(1 / 3)


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
