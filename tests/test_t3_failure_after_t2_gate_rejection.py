"""End-to-end coverage for T3-failure-reverts-to-rejected-value: a field
that reaches T3 because the T2 gate specifically rejected its calculation
as unreliable must NOT fall back to "keep Pass-1's value unchanged" if T3
also fails -- that value is exactly what the T2 gate just rejected. An
ordinary figure-only field that never touched the T2 gate should still get
the old, safer "keep Pass-1's value" fallback, since nothing ever flagged
its value as specifically unreliable.
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


class _FakeUploadedFile:
    name = "files/fake"


class _FakeClient:
    class files:
        @staticmethod
        def delete(name):
            pass


def _fake_upload_file(api_key, path):
    return _FakeClient(), _FakeUploadedFile()


def _pass1_markdown() -> str:
    header = "| " + " | ".join(TABLE1_COLUMNS) + " |"
    separator = "|" + "---|" * len(TABLE1_COLUMNS)
    values = {col: "-" for col in TABLE1_COLUMNS}
    values["Sample ID"] = "S1"
    values["Spall Strength (GPa)"] = "1.5"
    values["Spall Pullback Velocity (m/s)"] = "170"  # Pass-1's raw guess: must not survive
    row = "| " + " | ".join(values[col] for col in TABLE1_COLUMNS) + " |"
    return f"""## Table 1: Extracted Data

{header}
{separator}
{row}

## Table 2: Evidence Source

| Column Name | Source Location | Notes |
|---|---|---|
| Spall Strength (GPa) | see notes | [Priority 3]: visually extracted from the pullback curve; ⚠ visual extraction |
| Spall Pullback Velocity (m/s) | Fig. 3 | [Priority 2/3]: calculated from Spall Strength using v_pb = 2*sigma_sp; for S1, read from Fig. 3 |
"""


def _fake_run_gemini(pass1_text, bbox_response):
    def fake_run_gemini(api_key, model_id, contents, media_resolution="MEDIA_RESOLUTION_HIGH"):
        if media_resolution == "MEDIA_RESOLUTION_UNSPECIFIED":
            return pass1_text, {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0, "total_tokens": 150}
        if len(contents) == 1:
            # identify_calculation's call.
            return (
                json.dumps({"formula": "2*sigma_sp", "variables": {"sigma_sp": "Spall Strength (GPa)"}}),
                {"input_tokens": 20, "output_tokens": 10, "thinking_tokens": 0, "total_tokens": 30},
            )
        # locate_figure_bbox's call.
        return bbox_response, {"input_tokens": 15, "output_tokens": 5, "thinking_tokens": 0, "total_tokens": 20}

    return fake_run_gemini


def test_field_rejected_by_t2_gate_is_blanked_when_t3_also_fails(tmp_path):
    pass1_text = _pass1_markdown()
    fake_run_gemini = _fake_run_gemini(pass1_text, bbox_response="could not find that figure on this page")

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("base prompt text", encoding="utf-8")
    output_dir = tmp_path / "out"

    with (
        mock.patch.object(m, "upload_file", _fake_upload_file),
        mock.patch.object(m, "run_gemini", fake_run_gemini),
        mock.patch.object(m, "find_figure_page", lambda pdf_path, fig_num: 1),
        mock.patch.object(m, "render_page", lambda pdf_path, page, figures_dir: Path("/tmp/fake-page.png")),
    ):
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
    # Not "170" (the rejected Pass-1 value) -- the T2 gate rejected the
    # calculation, T3 also failed to localize Fig. 3, so there's nowhere
    # left to verify it.
    assert extracted.loc[0, "Spall Pullback Velocity (m/s)"] == "-"

    needs_review = pd.read_csv(paper_dir / "needs_review.csv", dtype=str)
    note = needs_review.loc[0, "Needs Review"]
    assert "Spall Pullback Velocity (m/s)" in note
    assert "figure fallback also failed" in note


def test_ordinary_figure_only_field_still_keeps_pass1_value_when_t3_fails(tmp_path):
    # Contrast case: a field that was ALWAYS figure-only (never touched the
    # T2 gate at all, since it has no calculation mentioned anywhere) must
    # keep its old, safer fallback when T3 fails -- its Pass-1 value was
    # never specifically flagged as unreliable.
    header = "| " + " | ".join(TABLE1_COLUMNS) + " |"
    separator = "|" + "---|" * len(TABLE1_COLUMNS)
    values = {col: "-" for col in TABLE1_COLUMNS}
    values["Sample ID"] = "S1"
    values["Strain Rate (s⁻¹)"] = "500000"
    row = "| " + " | ".join(values[col] for col in TABLE1_COLUMNS) + " |"
    pass1_text = f"""## Table 1: Extracted Data

{header}
{separator}
{row}

## Table 2: Evidence Source

| Column Name | Source Location | Notes |
|---|---|---|
| Strain Rate (s⁻¹) | Fig. 4 | [Priority 3]: read from strain rate vs stress plot |
"""
    fake_run_gemini = _fake_run_gemini(pass1_text, bbox_response="could not find that figure on this page")

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("base prompt text", encoding="utf-8")
    output_dir = tmp_path / "out"

    with (
        mock.patch.object(m, "upload_file", _fake_upload_file),
        mock.patch.object(m, "run_gemini", fake_run_gemini),
        mock.patch.object(m, "find_figure_page", lambda pdf_path, fig_num: 1),
        mock.patch.object(m, "render_page", lambda pdf_path, page, figures_dir: Path("/tmp/fake-page.png")),
    ):
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
    # Kept unchanged -- this field never touched the T2 gate, so there was
    # never a specific rejection to defer to. (reconcile_pass1_table
    # reformats the numeric string, hence 500000.0 not 500000.)
    assert float(extracted.loc[0, "Strain Rate (s⁻¹)"]) == 500000
