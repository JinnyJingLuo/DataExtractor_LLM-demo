"""End-to-end coverage for a chained T2-gate calculation: one field's
formula depends on another field the T2 gate itself resolves in the same
run. Real Paper12 case: "Shear Stress at HEL" depends on "Longitudinal
Stress at HEL", which the T2 gate computes deterministically via a known
formula. Before this fix, plain dict-iteration order decided which field
got processed first -- and when "Shear Stress at HEL" happened to be
processed before its own dependency, validate_input_provenance rejected it
(Pass-1's evidence still tagged the input "calculated", not "direct"),
blanking a field ground truth actually has a real value for. This test
deliberately puts the fields in that "wrong" order in the synthetic Table 2
to prove order_fields_by_dependency -- not luck -- makes it resolve
correctly now.
"""
import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import run_figure_refine as m
from run_figure_refine import run_pipeline

TABLE1_COLUMNS = [
    "Metal Symbol", "Sample ID", "Synthesis Method", "Treatment",
    "Initial Temperature (K)", "Quasi-static Yield Stress (MPa)",
    "Free Surface Velocity at HEL (m/s)", "Shear Stress at HEL (GPa)",
    "Hardness", "Bulk Modulus (GPa)", "Shear Modulus (GPa)",
    "Young's Modulus (GPa)", "Poisson's Ratio", "Melting Point (K)",
    "Sample Thickness (mm)", "Sample Diameter (mm)", "Grain Size (µm)",
    "Initial Density (g/cm³)", "Longitudinal Sound Speed (m/s)",
    "Shear Sound Speed (m/s)", "Bulk Sound Speed (m/s)",
    "Flyer Material Name", "Flyer Material Code", "Flyer Thickness (mm)",
    "Flyer Diameter (mm)", "Impact Velocity (m/s)",
    "Longitudinal Stress at HEL (GPa)", "Peak Stress / Hugoniot Stress (GPa)",
    "Strain Rate (s⁻¹)", "Pulse Duration (µs)", "Experiment Type",
    "Gas Gun Diameter (mm)", "Spall Strength (GPa)",
    "Spall Pullback Velocity (m/s)", "Reference Title", "DOI", "Verification",
]


def _pass1_markdown() -> str:
    header = "| " + " | ".join(TABLE1_COLUMNS) + " |"
    separator = "|" + "---|" * len(TABLE1_COLUMNS)
    values = {col: "-" for col in TABLE1_COLUMNS}
    values["Sample ID"] = "S1"
    values["Initial Density (g/cm³)"] = "6.11"
    values["Longitudinal Sound Speed (m/s)"] = "6090"
    values["Free Surface Velocity at HEL (m/s)"] = "180"
    values["Poisson's Ratio"] = "0.3"
    values["Longitudinal Stress at HEL (GPa)"] = "3.0"  # Pass-1's raw guess, to be overwritten
    values["Shear Stress at HEL (GPa)"] = "0.9"  # Pass-1's raw guess, to be overwritten
    row = "| " + " | ".join(values[col] for col in TABLE1_COLUMNS) + " |"
    # Table 2 row order is deliberately "downstream field first" -- the
    # dependency this test exists to catch.
    return f"""## Table 1: Extracted Data

{header}
{separator}
{row}

## Table 2: Evidence Source

| Column Name | Source Location | Notes |
|---|---|---|
| Shear Stress at HEL (GPa) | CALCULATED | CALCULATED (P2): tau_HEL = sigma_HEL*(1-2*nu)/(2*(1-nu)) |
| Longitudinal Stress at HEL (GPa) | CALCULATED | CALCULATED (P2): sigma_HEL = 0.5*rho0*c_l*u_HEL |
| Initial Density (g/cm³) | Table I | [Priority 1]: stated in Table I |
| Longitudinal Sound Speed (m/s) | Table I | DIRECT (P1): stated in Table I |
| Free Surface Velocity at HEL (m/s) | Page 4 | [Priority 1]: explicit precursor plateau value |
| Poisson's Ratio | Table I | [Priority 1]: stated in Table I |
"""


class _FakeUploadedFile:
    name = "files/fake-pdf"


class _FakeClient:
    class files:
        @staticmethod
        def delete(name):
            pass


def _fake_upload_file(api_key, path):
    return _FakeClient(), _FakeUploadedFile()


def test_chained_t2_gate_calculation_resolves_regardless_of_evidence_table_order(tmp_path):
    pass1_text = _pass1_markdown()

    def fake_run_gemini(api_key, model_id, contents, media_resolution="MEDIA_RESOLUTION_HIGH"):
        if media_resolution == "MEDIA_RESOLUTION_UNSPECIFIED":
            return pass1_text, {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0, "total_tokens": 150}
        # identify_calculation's call, for "Shear Stress at HEL" (no known
        # hardcoded spec exists for it, unlike Longitudinal Stress at HEL).
        return (
            json.dumps({
                "formula": "sigma_HEL*(1-2*nu)/(2*(1-nu))",
                "variables": {"sigma_HEL": "Longitudinal Stress at HEL (GPa)", "nu": "Poisson's Ratio"},
            }),
            {"input_tokens": 20, "output_tokens": 10, "thinking_tokens": 0, "total_tokens": 30},
        )

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("base prompt text", encoding="utf-8")
    output_dir = tmp_path / "out"

    with mock.patch.object(m, "upload_file", _fake_upload_file), mock.patch.object(m, "run_gemini", fake_run_gemini):
        run_pipeline(
            pdf_path=Path("/nonexistent/fake.pdf"),
            prompt_path=prompt_path,
            paper_id="TestPaper",
            split="test",
            output_dir=output_dir,
            model_id="fake-model",
            api_key="fake-key",
            pass1_draws=1,
        )

    paper_dir = output_dir / "test" / "TestPaper"
    extracted = pd.read_csv(paper_dir / "extracted_data.csv", dtype=str)

    sigma_hel = float(extracted.loc[0, "Longitudinal Stress at HEL (GPa)"])
    assert sigma_hel == pytest.approx(3.348891, rel=1e-4)

    # The real bug: without dependency ordering, this came back "-" even
    # though its only input was resolved just fine moments later.
    tau_hel = extracted.loc[0, "Shear Stress at HEL (GPa)"]
    assert tau_hel != "-"
    assert float(tau_hel) == pytest.approx(0.956826, rel=1e-4)
