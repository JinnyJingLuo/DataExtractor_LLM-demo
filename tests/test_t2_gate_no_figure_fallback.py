"""End-to-end coverage for the "invents a number" gap: a calculated-only
field (tier vote says a formula was used, but no draw ever mentioned a
figure) whose formula input is itself unreliable, with no figure to fall
back to. Before this fix, such a field went straight from
reconcile_pass1_table's raw numeric clustering into extracted_data.csv with
no scrutiny of whether the calculation's inputs were trustworthy -- observed
concretely on Paper12's "Spall Pullback Velocity", which ground truth has no
value for at all. Now it must be blanked ("-") and flagged for review
instead of reporting a number nobody actually verified.
"""
import json
from pathlib import Path
from unittest import mock

import pandas as pd

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


def _pass1_markdown(spall_strength_note: str, pullback_note: str) -> str:
    header = "| " + " | ".join(TABLE1_COLUMNS) + " |"
    separator = "|" + "---|" * len(TABLE1_COLUMNS)
    values = {col: "-" for col in TABLE1_COLUMNS}
    values["Sample ID"] = "S1"
    values["Spall Strength (GPa)"] = "1.5"
    values["Spall Pullback Velocity (m/s)"] = "3.1"
    row = "| " + " | ".join(values[col] for col in TABLE1_COLUMNS) + " |"
    return f"""## Table 1: Extracted Data

{header}
{separator}
{row}

## Table 2: Evidence Source

| Column Name | Source Location | Notes |
|---|---|---|
| Spall Strength (GPa) | see notes | {spall_strength_note} |
| Spall Pullback Velocity (m/s) | CALCULATED | {pullback_note} |
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


def _run(tmp_path: Path, spall_strength_note: str, pullback_note: str) -> Path:
    pass1_text = _pass1_markdown(spall_strength_note, pullback_note)

    def fake_run_gemini(api_key, model_id, contents, media_resolution="MEDIA_RESOLUTION_HIGH"):
        if media_resolution == "MEDIA_RESOLUTION_UNSPECIFIED":
            # Pass-1 whole-document call.
            return pass1_text, {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0, "total_tokens": 150}
        # identify_calculation's call.
        return (
            json.dumps({"formula": "2*sigma_sp", "variables": {"sigma_sp": "Spall Strength (GPa)"}}),
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

    return output_dir / "test" / "TestPaper"


def test_calculated_only_field_with_unreliable_input_and_no_figure_is_blanked(tmp_path):
    # Spall Strength was visually extracted from a figure (not confirmed
    # direct), so a formula built on top of it can't be trusted. Spall
    # Pullback Velocity's own evidence never mentions a figure at all, so
    # there's nothing to fall through to -- it must be blanked, not left at
    # the invented "120" from Pass-1's raw numeric clustering.
    paper_dir = _run(
        tmp_path,
        spall_strength_note="[Priority 3]: visually extracted from the pullback curve; ⚠ visual extraction",
        pullback_note="[Priority 2]: calculated from Spall Strength using v_pb = 2*sigma_sp",
    )

    extracted = pd.read_csv(paper_dir / "extracted_data.csv", dtype=str)
    assert extracted.loc[0, "Spall Pullback Velocity (m/s)"] == "-"

    needs_review = pd.read_csv(paper_dir / "needs_review.csv", dtype=str)
    assert needs_review.loc[0, "Sample ID"] == "S1"
    assert "Spall Pullback Velocity (m/s)" in needs_review.loc[0, "Needs Review"]


def test_calculated_only_field_with_direct_input_and_no_figure_is_resolved(tmp_path):
    # Contrast case: same "calculated-only, no figure" shape, but the input
    # this time IS confirmed direct (Priority 1). The T2 gate should trust
    # this one and compute a value deterministically, not blank it.
    paper_dir = _run(
        tmp_path,
        spall_strength_note="[Priority 1]: stated directly in Table II as 1.5 GPa",
        pullback_note="[Priority 2]: calculated from Spall Strength using v_pb = 2*sigma_sp",
    )

    extracted = pd.read_csv(paper_dir / "extracted_data.csv", dtype=str)
    assert extracted.loc[0, "Spall Pullback Velocity (m/s)"] == "3.0"

    needs_review_path = paper_dir / "needs_review.csv"
    if needs_review_path.exists():
        needs_review = pd.read_csv(needs_review_path, dtype=str)
        assert "Spall Pullback Velocity (m/s)" not in "".join(needs_review["Needs Review"].astype(str))
