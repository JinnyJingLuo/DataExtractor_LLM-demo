import json
import pandas as pd
import pytest

from t2_gate import (
    CalculationSpec,
    detect_dual_path_fields,
    filter_unstable_inputs,
    identify_calculation,
    known_calculation_specs,
    known_confusable_specs,
    order_fields_by_dependency,
    resolve_via_t2_gate,
    safe_eval_formula,
    validate_input_provenance,
)


def test_detects_paper12_longitudinal_stress_dual_path():
    evidence_rows = [
        {
            "Column Name": "Longitudinal Stress at HEL (GPa)",
            "Source Location": "Page 6, Fig. 5(a) & CALCULATED",
            "Notes": (
                "[Priority 3/2]: For VA0, [Priority 3] visually extracted "
                "from Fig. 5(a) (~1.0 GPa); ⚠ visual extraction; type: "
                "extracted_from_figure. For VA0.6/VA5.5, [Priority 2] "
                "calculated from text u_HEL values; type: calculated_from_u_HEL"
            ),
        }
    ]
    assert detect_dual_path_fields(evidence_rows) == {
        "Longitudinal Stress at HEL (GPa)": "5"
    }


def test_ignores_pure_calculation_field_with_no_figure():
    evidence_rows = [
        {
            "Column Name": "Young's Modulus (GPa)",
            "Source Location": "CALCULATED",
            "Notes": "[Priority 2]: E = 9BG / (3B+G)",
        }
    ]
    assert detect_dual_path_fields(evidence_rows) == {}


def test_ignores_pure_figure_field_with_no_calculation():
    evidence_rows = [
        {
            "Column Name": "Strain Rate (s⁻¹)",
            "Source Location": "Page 5, Fig. 4(c)",
            "Notes": "[Priority 3]: Extracted from compressive strain rate vs stress plot",
        }
    ]
    assert detect_dual_path_fields(evidence_rows) == {}


def test_detects_full_figure_word_in_dual_path_field():
    evidence_rows = [
        {
            "Column Name": "Longitudinal Stress at HEL (GPa)",
            "Source Location": "Page 6, Figure 5(a) & CALCULATED",
            "Notes": "[Priority 2/3]: calculated where possible, otherwise FIGURE (P3)",
        }
    ]
    assert detect_dual_path_fields(evidence_rows) == {
        "Longitudinal Stress at HEL (GPa)": "5"
    }


def test_evaluates_paper12_sigma_hel_formula():
    value = safe_eval_formula(
        "0.5*rho0*c_l*u_HEL/1e9",
        {"rho0": 6110, "c_l": 6090, "u_HEL": 180},
    )
    assert value == pytest.approx(3.348891, rel=1e-4)


def test_known_longitudinal_stress_spec_uses_actual_column_names():
    columns = [
        "Sample ID",
        "Initial Density (g/cm³)",
        "Longitudinal Sound Speed (m/s)",
        "Free Surface Velocity at HEL (m/s)",
    ]
    spec = known_calculation_specs(columns)["Longitudinal Stress at HEL (GPa)"]
    assert spec == CalculationSpec(
        formula="0.5*rho0*c_l*u_HEL/1e6",
        variables={
            "rho0": "Initial Density (g/cm³)",
            "c_l": "Longitudinal Sound Speed (m/s)",
            "u_HEL": "Free Surface Velocity at HEL (m/s)",
        },
    )


def test_known_youngs_modulus_spec_uses_actual_column_names():
    columns = ["Sample ID", "Bulk Modulus (GPa)", "Shear Modulus (GPa)", "Young's Modulus (GPa)"]
    spec = known_calculation_specs(columns)["Young's Modulus (GPa)"]
    assert spec == CalculationSpec(
        formula="9*B*G/(3*B+G)",
        variables={"B": "Bulk Modulus (GPa)", "G": "Shear Modulus (GPa)"},
    )


def test_known_youngs_modulus_spec_matches_paper12_mislabeled_table_value():
    # Real Paper12 numbers: Table I reports "226 GPa" as Young's Modulus,
    # but ground truth is ~131.9 GPa -- the table actually reports
    # Longitudinal Modulus, not Young's Modulus. B=163.0, G=48.3 are the
    # actual extracted Bulk/Shear Modulus values for this paper.
    spec = known_calculation_specs(["Bulk Modulus (GPa)", "Shear Modulus (GPa)"])["Young's Modulus (GPa)"]
    value = safe_eval_formula(spec.formula, {"B": 163.0, "G": 48.3})
    assert value == pytest.approx(131.9, rel=0.01)


def test_known_confusable_specs_longitudinal_modulus_matches_paper12_numbers():
    columns = [
        "Bulk Modulus (GPa)", "Shear Modulus (GPa)",
        "Initial Density (g/cm³)", "Longitudinal Sound Speed (m/s)",
    ]
    candidates = known_confusable_specs(columns)["Young's Modulus (GPa)"]
    assert len(candidates) == 2

    values_by_formula = {c.calc_spec.formula: c for c in candidates}
    bg_candidate = values_by_formula["B+4*G/3"]
    rho_c_candidate = values_by_formula["rho0*c_l**2/1e6"]

    bg_value = safe_eval_formula(bg_candidate.calc_spec.formula, {"B": 163.0, "G": 48.3})
    rho_c_value = safe_eval_formula(rho_c_candidate.calc_spec.formula, {"rho0": 6.11, "c_l": 6090})

    # Both independently land within ~1% of the actual (mislabeled) reported
    # value of 226.0, despite using entirely different input columns.
    assert bg_value == pytest.approx(227.4, rel=1e-3)
    assert rho_c_value == pytest.approx(226.6, rel=1e-3)


def test_known_confusable_specs_empty_when_columns_missing():
    assert known_confusable_specs(["Sample ID"]) == {}


def test_known_longitudinal_stress_spec_resolves_g_per_cm3_to_gpa():
    columns = [
        "Sample ID",
        "Initial Density (g/cm³)",
        "Longitudinal Sound Speed (m/s)",
        "Free Surface Velocity at HEL (m/s)",
    ]
    calc_spec = known_calculation_specs(columns)["Longitudinal Stress at HEL (GPa)"]
    extracted = pd.DataFrame(
        [
            {
                "Sample ID": "VA0-300",
                "Initial Density (g/cm³)": 6.11,
                "Longitudinal Sound Speed (m/s)": 6090,
                "Free Surface Velocity at HEL (m/s)": 180,
            }
        ]
    )
    result = resolve_via_t2_gate(calc_spec, extracted)
    assert result["VA0-300"] == pytest.approx(3.348891, rel=1e-4)


def test_rejects_bool_literal_as_numeric_constant():
    # bool is a subclass of int in Python -- without an explicit check,
    # True/False would silently evaluate as 1/0 instead of being rejected.
    with pytest.raises(ValueError, match="unsupported constant"):
        safe_eval_formula("True + 5", {})


def test_rejects_function_calls():
    with pytest.raises(ValueError, match="unsupported"):
        safe_eval_formula("__import__('os').system('ls')", {})


def test_rejects_unknown_variable():
    with pytest.raises(ValueError, match="unknown variable"):
        safe_eval_formula("a + b", {"a": 1.0})


def test_identify_calculation_parses_formula_json():
    def fake_call_llm(api_key, model_id, contents):
        return (
            json.dumps(
                {
                    "formula": "0.5*rho0*c_l*u_HEL/1e6",
                    "variables": {
                        "rho0": "Initial Density (g/cm³)",
                        "c_l": "Longitudinal Sound Speed (m/s)",
                        "u_HEL": "Free Surface Velocity at HEL (m/s)",
                    },
                }
            ),
            {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0, "total_tokens": 15},
        )

    spec = identify_calculation(
        field_name="Longitudinal Stress at HEL (GPa)",
        evidence_note="CALCULATED (P2): sigma_HEL = 0.5 * rho0 * c_l * u_HEL",
        available_columns=["Initial Density (g/cm³)", "Longitudinal Sound Speed (m/s)", "Free Surface Velocity at HEL (m/s)"],
        api_key="fake-key",
        model_id="fake-model",
        call_llm=fake_call_llm,
    )
    assert spec == CalculationSpec(
        formula="0.5*rho0*c_l*u_HEL/1e6",
        variables={
            "rho0": "Initial Density (g/cm³)",
            "c_l": "Longitudinal Sound Speed (m/s)",
            "u_HEL": "Free Surface Velocity at HEL (m/s)",
        },
    )


def test_identify_calculation_ignores_unrelated_json_like_fragment_before_the_real_answer():
    # The old greedy r"\{.*\}" (DOTALL) regex spanned from the FIRST "{" to
    # the LAST "}" anywhere in the response -- if the model's prose happened
    # to mention an example like {"foo": "bar"} before giving the real
    # answer, it would merge both into one unparseable (or wrong) blob.
    def fake_call_llm(api_key, model_id, contents):
        return (
            'For context, a formula looks like {"foo": "bar"}. '
            + json.dumps({
                "formula": "0.5*rho0*c_l*u_HEL/1e6",
                "variables": {"rho0": "Initial Density (g/cm³)", "c_l": "Longitudinal Sound Speed (m/s)", "u_HEL": "Free Surface Velocity at HEL (m/s)"},
            }),
            {},
        )

    spec = identify_calculation(
        field_name="Longitudinal Stress at HEL (GPa)",
        evidence_note="CALCULATED (P2)",
        available_columns=["Initial Density (g/cm³)", "Longitudinal Sound Speed (m/s)", "Free Surface Velocity at HEL (m/s)"],
        api_key="fake-key",
        model_id="fake-model",
        call_llm=fake_call_llm,
    )
    # Matches the FIRST complete, balanced object -- the unrelated
    # {"foo": "bar"} example -- and returns None (no "formula" key) rather
    # than merging both fragments into a single unparseable/wrong blob, or
    # crashing, the way the old greedy regex would have.
    assert spec is None


def test_identify_calculation_parses_nested_variables_object_correctly():
    # The replacement regex must still handle the one level of nesting
    # ("variables" is itself an object) that the real response format uses.
    def fake_call_llm(api_key, model_id, contents):
        return (
            json.dumps({
                "formula": "0.5*rho0*c_l*u_HEL/1e6",
                "variables": {"rho0": "Initial Density (g/cm³)", "c_l": "Longitudinal Sound Speed (m/s)", "u_HEL": "Free Surface Velocity at HEL (m/s)"},
            }),
            {},
        )

    spec = identify_calculation(
        field_name="Longitudinal Stress at HEL (GPa)",
        evidence_note="CALCULATED (P2)",
        available_columns=["Initial Density (g/cm³)", "Longitudinal Sound Speed (m/s)", "Free Surface Velocity at HEL (m/s)"],
        api_key="fake-key",
        model_id="fake-model",
        call_llm=fake_call_llm,
    )
    assert spec.formula == "0.5*rho0*c_l*u_HEL/1e6"
    assert spec.variables == {
        "rho0": "Initial Density (g/cm³)",
        "c_l": "Longitudinal Sound Speed (m/s)",
        "u_HEL": "Free Surface Velocity at HEL (m/s)",
    }


def test_identify_calculation_returns_none_when_no_formula():
    def fake_call_llm(api_key, model_id, contents):
        return json.dumps({"formula": None}), {}

    spec = identify_calculation(
        field_name="Some Field",
        evidence_note="no closed-form relation available",
        available_columns=["Some Field", "Other Column"],
        api_key="fake-key",
        model_id="fake-model",
        call_llm=fake_call_llm,
    )
    assert spec is None


def test_identify_calculation_includes_available_columns_in_prompt():
    captured_prompt = {}

    def fake_call_llm(api_key, model_id, contents):
        captured_prompt["text"] = contents[0]
        return json.dumps({"formula": None}), {}

    identify_calculation(
        field_name="Some Field",
        evidence_note="note",
        available_columns=["Initial Density (g/cm³)", "Sample ID"],
        api_key="fake-key",
        model_id="fake-model",
        call_llm=fake_call_llm,
    )
    assert "Initial Density (g/cm³)" in captured_prompt["text"]
    assert "Sample ID" in captured_prompt["text"]


def test_resolve_via_t2_gate_computes_and_flags_missing():
    calc_spec = CalculationSpec(
        formula="0.5*rho0*c_l*u_HEL/1e9",
        variables={
            "rho0": "Initial Density (g/cm3)",
            "c_l": "Longitudinal Sound Speed (m/s)",
            "u_HEL": "Free Surface Velocity at HEL (m/s)",
        },
    )
    extracted = pd.DataFrame(
        [
            {
                "Sample ID": "VA0.6-300",
                "Initial Density (g/cm3)": 6110,
                "Longitudinal Sound Speed (m/s)": 6090,
                "Free Surface Velocity at HEL (m/s)": 180,
            },
            {
                "Sample ID": "VA0-300",
                "Initial Density (g/cm3)": 6110,
                "Longitudinal Sound Speed (m/s)": 6090,
                "Free Surface Velocity at HEL (m/s)": "-",
            },
        ]
    )
    result = resolve_via_t2_gate(calc_spec, extracted)
    assert result["VA0.6-300"] == pytest.approx(3.348891, rel=1e-4)
    assert result["VA0-300"] is None


def _hel_calc_spec():
    return CalculationSpec(
        formula="0.5*rho0*c_l*u_HEL/1e6",
        variables={
            "rho0": "Initial Density (g/cm³)",
            "c_l": "Longitudinal Sound Speed (m/s)",
            "u_HEL": "Free Surface Velocity at HEL (m/s)",
        },
    )


def test_validate_input_provenance_accepts_all_direct_inputs():
    evidence_rows = [
        {"Column Name": "Initial Density (g/cm³)", "Notes": "[Priority 1]: stated in Table I", "Source Location": "Table I"},
        {"Column Name": "Longitudinal Sound Speed (m/s)", "Notes": "DIRECT (P1): stated in Table I", "Source Location": "Table I"},
        {"Column Name": "Free Surface Velocity at HEL (m/s)", "Notes": "[Priority 1]: explicit precursor plateau value", "Source Location": "Page 4"},
    ]
    valid, reason = validate_input_provenance(_hel_calc_spec(), evidence_rows)
    assert valid is True
    assert reason is None


def test_validate_input_provenance_rejects_circular_derivation():
    # Exact real evidence text observed for Paper12 VA0: u_HEL was derived
    # FROM sigma_HEL (the target field), then the T2 gate would try to
    # compute sigma_HEL right back from it -- a loop, not a T1->T2 chain.
    evidence_rows = [
        {"Column Name": "Initial Density (g/cm³)", "Notes": "[Priority 1]: stated in Table I", "Source Location": "Table I"},
        {"Column Name": "Longitudinal Sound Speed (m/s)", "Notes": "DIRECT (P1): stated in Table I", "Source Location": "Table I"},
        {
            "Column Name": "Free Surface Velocity at HEL (m/s)",
            "Notes": "[Priority 2]: calculated from sigma_HEL using u_HEL = sigma_HEL / (0.5*rho0*C_l)",
            "Source Location": "CALCULATED",
        },
    ]
    valid, reason = validate_input_provenance(_hel_calc_spec(), evidence_rows)
    assert valid is False
    assert "Free Surface Velocity at HEL (m/s)" in reason


def test_validate_input_provenance_rejects_missing_evidence_row():
    evidence_rows = [
        {"Column Name": "Initial Density (g/cm³)", "Notes": "[Priority 1]: stated in Table I", "Source Location": "Table I"},
        {"Column Name": "Longitudinal Sound Speed (m/s)", "Notes": "DIRECT (P1): stated in Table I", "Source Location": "Table I"},
        # no evidence row at all for Free Surface Velocity at HEL (m/s)
    ]
    valid, reason = validate_input_provenance(_hel_calc_spec(), evidence_rows)
    assert valid is False
    assert "no evidence row" in reason


def _consistency_row(sample_id, field_name, confidence):
    return {
        "sample_id": sample_id,
        "field_name": field_name,
        "selected_value": "x",
        "agreement_count": 1,
        "compared_draws": 2,
        "agreement_fraction": 1.0 if confidence == "high" else 0.5,
        "confidence": confidence,
        "observed_values_json": "[]",
    }


def test_filter_unstable_inputs_noop_when_no_consistency_data():
    resolved = {"VA0-300": 3.35, "VA0.6-300": 1.04}
    assert filter_unstable_inputs(resolved, _hel_calc_spec(), None) == resolved


def test_filter_unstable_inputs_downgrades_sample_with_unstable_input():
    resolved = {"VA0-300": 1.05, "VA0.6-300": 1.04}
    consistency = pd.DataFrame(
        [
            _consistency_row("VA0-300", "Free Surface Velocity at HEL (m/s)", "low"),
            _consistency_row("VA0-300", "Initial Density (g/cm³)", "high"),
            _consistency_row("VA0.6-300", "Free Surface Velocity at HEL (m/s)", "high"),
            _consistency_row("VA0.6-300", "Initial Density (g/cm³)", "high"),
        ]
    )
    result = filter_unstable_inputs(resolved, _hel_calc_spec(), consistency)
    assert result["VA0-300"] is None
    assert result["VA0.6-300"] == pytest.approx(1.04)


def _sigma_hel_spec():
    return CalculationSpec(
        formula="0.5*rho0*c_l*u_HEL/1e6",
        variables={
            "rho0": "Initial Density (g/cm³)",
            "c_l": "Longitudinal Sound Speed (m/s)",
            "u_HEL": "Free Surface Velocity at HEL (m/s)",
        },
    )


def _tau_hel_spec():
    return CalculationSpec(
        formula="sigma_HEL*(1-2*nu)/(2*(1-nu))",
        variables={"sigma_HEL": "Longitudinal Stress at HEL (GPa)", "nu": "Poisson's Ratio"},
    )


def test_validate_input_provenance_trusts_a_field_already_resolved_by_t2_gate():
    # Real Paper12 case: "Shear Stress at HEL" depends on "Longitudinal
    # Stress at HEL", which the T2 gate itself already computed
    # deterministically earlier in the same run. Pass-1's own evidence text
    # still (correctly) says that input is "calculated", which the ordinary
    # evidence-based check would reject -- resolved_by_t2_gate lets a
    # genuinely-resolved upstream field be trusted instead.
    evidence_rows = [
        {"Column Name": "Longitudinal Stress at HEL (GPa)", "Notes": "CALCULATED (P2): ...", "Source Location": "CALCULATED"},
        {"Column Name": "Poisson's Ratio", "Notes": "[Priority 1]: stated in Table I", "Source Location": "Table I"},
    ]
    valid, reason = validate_input_provenance(
        _tau_hel_spec(),
        evidence_rows,
        field_name="Shear Stress at HEL (GPa)",
        resolved_by_t2_gate={"Longitudinal Stress at HEL (GPa)": _sigma_hel_spec()},
    )
    assert valid is True
    assert reason is None


def test_validate_input_provenance_still_rejects_a_genuine_two_step_circular_pair():
    # A depends on B, and B's own recorded formula depends on A -- a real
    # loop, not a legitimate chain. Must still be rejected even though B is
    # "resolved_by_t2_gate", because trusting it here would make both halves
    # of the loop trust each other in turn.
    a_spec = CalculationSpec(formula="b*2", variables={"b": "B"})
    b_spec = CalculationSpec(formula="a*2", variables={"a": "A"})
    evidence_rows = [{"Column Name": "B", "Notes": "CALCULATED", "Source Location": "CALCULATED"}]
    valid, reason = validate_input_provenance(
        a_spec,
        evidence_rows,
        field_name="A",
        resolved_by_t2_gate={"B": b_spec},
    )
    assert valid is False
    assert "circular" in reason


def test_order_fields_by_dependency_puts_upstream_field_first():
    # Insertion order deliberately backwards from the real dependency, to
    # prove the function reorders rather than just preserving input order.
    field_calc_specs = {
        "Shear Stress at HEL (GPa)": _tau_hel_spec(),
        "Longitudinal Stress at HEL (GPa)": _sigma_hel_spec(),
    }
    order = order_fields_by_dependency(field_calc_specs)
    assert order.index("Longitudinal Stress at HEL (GPa)") < order.index("Shear Stress at HEL (GPa)")


def test_order_fields_by_dependency_handles_fields_with_no_formula():
    field_calc_specs = {"X": None, "Y": _sigma_hel_spec()}
    order = order_fields_by_dependency(field_calc_specs)
    assert set(order) == {"X", "Y"}


def test_order_fields_by_dependency_does_not_hang_on_a_real_cycle():
    a_spec = CalculationSpec(formula="b*2", variables={"b": "B"})
    b_spec = CalculationSpec(formula="a*2", variables={"a": "A"})
    order = order_fields_by_dependency({"A": a_spec, "B": b_spec})
    assert set(order) == {"A", "B"}


def test_filter_unstable_inputs_leaves_fully_stable_samples_untouched():
    resolved = {"VA0-300": 3.35}
    consistency = pd.DataFrame(
        [
            _consistency_row("VA0-300", "Free Surface Velocity at HEL (m/s)", "high"),
            _consistency_row("VA0-300", "Initial Density (g/cm³)", "high"),
            _consistency_row("VA0-300", "Longitudinal Sound Speed (m/s)", "high"),
        ]
    )
    result = filter_unstable_inputs(resolved, _hel_calc_spec(), consistency)
    assert result == resolved
