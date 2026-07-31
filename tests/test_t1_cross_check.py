"""End-to-end coverage for cross-checking a "direct"-tier field against a
known physical relationship to other direct fields. "direct" fields skip
the T2/T3 gate entirely and are trusted as reported -- but the paper's own
table can be ambiguous about which quantity it's actually reporting.

Real Paper12 case: "Young's Modulus" was read straight from Table I as 226
GPa by the majority of Pass-1 draws, but the table actually reports
Longitudinal (P-wave) Modulus, not Young's Modulus -- ground truth is
~131.9 GPa, matching E = 9BG/(3B+G) computed from the paper's own (also
direct) Bulk and Shear Modulus values. Bare disagreement alone only flags
for review without changing the value, since either number could be right.
But 226.0 also matches M = B+4G/3 = 227.4 (0.6%) here -- a different,
nameable quantity, positively identifying a mislabeled column rather than
mere ambiguity, so this case is strong enough to auto-correct (with full
reasoning kept in needs_review.csv, not silent).
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


class _FakeUploadedFile:
    name = "files/fake-pdf"


class _FakeClient:
    class files:
        @staticmethod
        def delete(name):
            pass


def _fake_upload_file(api_key, path):
    return _FakeClient(), _FakeUploadedFile()


def _pass1_markdown(youngs_modulus_value: str) -> str:
    header = "| " + " | ".join(TABLE1_COLUMNS) + " |"
    separator = "|" + "---|" * len(TABLE1_COLUMNS)
    values = {col: "-" for col in TABLE1_COLUMNS}
    values["Sample ID"] = "S1"
    values["Bulk Modulus (GPa)"] = "163.0"
    values["Shear Modulus (GPa)"] = "48.3"
    values["Young's Modulus (GPa)"] = youngs_modulus_value
    row = "| " + " | ".join(values[col] for col in TABLE1_COLUMNS) + " |"
    return f"""## Table 1: Extracted Data

{header}
{separator}
{row}

## Table 2: Evidence Source

| Column Name | Source Location | Notes |
|---|---|---|
| Bulk Modulus (GPa) | Table I | DIRECT (P1): stated in Table I |
| Shear Modulus (GPa) | Table I | DIRECT (P1): stated in Table I |
| Young's Modulus (GPa) | Table I | DIRECT (P1): stated in Table I |
"""


def _run(tmp_path: Path, youngs_modulus_value: str) -> Path:
    pass1_text = _pass1_markdown(youngs_modulus_value)

    def fake_run_gemini(api_key, model_id, contents, media_resolution="MEDIA_RESOLUTION_HIGH"):
        return pass1_text, {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0, "total_tokens": 150}

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


def test_direct_field_matching_a_confusable_quantity_is_auto_corrected(tmp_path):
    paper_dir = _run(tmp_path, youngs_modulus_value="226.0")

    extracted = pd.read_csv(paper_dir / "extracted_data.csv", dtype=str)
    # 226.0 matches Longitudinal Modulus (B+4G/3=227.4) within 0.6% -- a
    # positive identification of a mislabeled column, strong enough to
    # replace the value with the target's own correctly-computed formula.
    assert float(extracted.loc[0, "Young's Modulus (GPa)"]) == pytest.approx(131.87, rel=1e-3)

    needs_review = pd.read_csv(paper_dir / "needs_review.csv", dtype=str)
    note = needs_review.loc[0, "Needs Review"]
    assert "AUTO-CORRECTED" in note
    assert "Young's Modulus (GPa)" in note
    assert "226" in note
    assert "Longitudinal" in note


def test_direct_field_disagreeing_with_no_confusable_match_is_flagged_not_overwritten(tmp_path):
    # 300.0 doesn't fit Young's Modulus (131.9) OR the Longitudinal Modulus
    # candidate (227.4) -- a genuine, unexplained disagreement, not a
    # positively-identified mislabeling. Must fall back to flag-only.
    paper_dir = _run(tmp_path, youngs_modulus_value="300.0")

    extracted = pd.read_csv(paper_dir / "extracted_data.csv", dtype=str)
    assert extracted.loc[0, "Young's Modulus (GPa)"] == "300.0"

    needs_review = pd.read_csv(paper_dir / "needs_review.csv", dtype=str)
    note = needs_review.loc[0, "Needs Review"]
    assert "AUTO-CORRECTED" not in note
    assert "Young's Modulus (GPa)" in note
    assert "300" in note


def test_direct_field_agreeing_with_known_formula_is_not_flagged(tmp_path):
    paper_dir = _run(tmp_path, youngs_modulus_value="131.9")

    extracted = pd.read_csv(paper_dir / "extracted_data.csv", dtype=str)
    assert extracted.loc[0, "Young's Modulus (GPa)"] == "131.9"

    needs_review_path = paper_dir / "needs_review.csv"
    if needs_review_path.exists():
        needs_review = pd.read_csv(needs_review_path, dtype=str)
        assert "Young's Modulus (GPa)" not in "".join(needs_review["Needs Review"].astype(str))
