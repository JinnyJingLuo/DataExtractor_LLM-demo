"""End-to-end coverage for the unclear-tier gap found on Paper1: a field
whose evidence-tier vote is "unclear" (most draws couldn't characterize
its source at all -- not direct, calculated, or figure) gets no scrutiny
from any of the named-tier gates, since none of them check for "unclear"
by name. reconcile_pass1_table's raw numeric clustering can still ship a
confident-looking number from a single outvoted draw in that situation,
since "-" draws are filtered out before clustering and there's nothing
else numeric to compete against a lone value.

Real Paper1 case: "Peak Stress / Hugoniot Stress" had 1 of 3 draws
calculate a number via an approximation while the other 2 explicitly said
the calculation couldn't be done reliably from the paper's data -- ground
truth has no value for it at all.
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


def _row(values: dict) -> str:
    return "| " + " | ".join(values[col] for col in TABLE1_COLUMNS) + " |"


def _draw_markdown(peak_stress_value: str, peak_stress_note: str, source_location: str) -> str:
    header = "| " + " | ".join(TABLE1_COLUMNS) + " |"
    separator = "|" + "---|" * len(TABLE1_COLUMNS)
    row1 = {col: "-" for col in TABLE1_COLUMNS}
    row1["Sample ID"] = "S1"
    row1["Peak Stress / Hugoniot Stress (GPa)"] = peak_stress_value
    return f"""## Table 1: Extracted Data

{header}
{separator}
{_row(row1)}

## Table 2: Evidence Source

| Column Name | Source Location | Notes |
|---|---|---|
| Peak Stress / Hugoniot Stress (GPa) | {source_location} | {peak_stress_note} |
"""


def test_lone_confident_draw_outvoted_by_declines_is_blanked_not_reported(tmp_path):
    # 1 of 3 draws calculates a number via an approximation; the other 2
    # explicitly decline, matching the real Paper1 shape exactly.
    draws = [
        _draw_markdown("1.047", "[Priority 2]: CALCULATED via acoustic approximation", "CALCULATED"),
        _draw_markdown("-", "Not found in text (Not explicitly reported per shot)", "-"),
        _draw_markdown("-", "Not found in text (Hugoniot equation parameters not fully detailed)", "-"),
    ]
    call_count = {"n": 0}

    def fake_run_gemini(api_key, model_id, contents, media_resolution="MEDIA_RESOLUTION_HIGH"):
        if media_resolution == "MEDIA_RESOLUTION_UNSPECIFIED":
            call_count["n"] += 1
            text = draws[call_count["n"] - 1]
            return text, {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0, "total_tokens": 150}
        raise AssertionError("no other Gemini calls expected -- field must be blanked before T2/T3")

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
            pass1_draws=3,
        )

    paper_dir = output_dir / "test" / "TestPaper"
    extracted = pd.read_csv(paper_dir / "extracted_data.csv", dtype=str)
    # Not "1.047" -- the lone outvoted draw's guess -- blanked instead.
    assert extracted.loc[0, "Peak Stress / Hugoniot Stress (GPa)"] == "-"

    needs_review = pd.read_csv(paper_dir / "needs_review.csv", dtype=str)
    note = needs_review.loc[0, "Needs Review"]
    assert "Peak Stress / Hugoniot Stress (GPa)" in note
    assert "unclear" in note


def test_unanimous_value_with_atypically_worded_evidence_is_not_blanked(tmp_path):
    # All 3 draws agree on the exact same value, even though their evidence
    # text happens to be worded outside the DIRECT_RE/CALC_RE/FIG_RE
    # patterns (so the tier vote still lands on "unclear"). Must NOT be
    # blanked -- the value itself has strong cross-draw agreement, unlike
    # the real bug case above.
    draws = [
        _draw_markdown("2.5", "Explicitly given in the text as a round number", "-"),
        _draw_markdown("2.5", "Explicitly given in the text as a round number", "-"),
        _draw_markdown("2.5", "Explicitly given in the text as a round number", "-"),
    ]
    call_count = {"n": 0}

    def fake_run_gemini(api_key, model_id, contents, media_resolution="MEDIA_RESOLUTION_HIGH"):
        if media_resolution == "MEDIA_RESOLUTION_UNSPECIFIED":
            call_count["n"] += 1
            text = draws[call_count["n"] - 1]
            return text, {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0, "total_tokens": 150}
        raise AssertionError("no other Gemini calls expected")

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
            pass1_draws=3,
        )

    paper_dir = output_dir / "test" / "TestPaper"
    extracted = pd.read_csv(paper_dir / "extracted_data.csv", dtype=str)
    assert float(extracted.loc[0, "Peak Stress / Hugoniot Stress (GPa)"]) == 2.5
