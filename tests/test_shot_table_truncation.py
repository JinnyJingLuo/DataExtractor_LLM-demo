"""End-to-end coverage for shot-table truncation: when the T2 gate resolves
some but not all samples for a field, the remaining T3 figure-read call
used to be scoped to ONLY the unresolved samples' shot table -- the model
wasn't told the other samples/shots exist at all, which can push it toward
mis-assigning a plotted point to the wrong (listed) shot.

Fix: always show the model the FULL sample list as shot-matching context,
but still only ever write back values for the samples that actually still
need this field -- verified here by having the mocked model "helpfully"
report a value for an already-T2-gate-resolved sample too, and confirming
it's discarded rather than overwriting the T2 gate's value.
"""
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import repeated_sampling
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


def _pass1_markdown() -> str:
    header = "| " + " | ".join(TABLE1_COLUMNS) + " |"
    separator = "|" + "---|" * len(TABLE1_COLUMNS)

    s1 = {col: "-" for col in TABLE1_COLUMNS}
    s1["Sample ID"] = "S1"
    s1["Initial Density (g/cm³)"] = "6.11"
    s1["Longitudinal Sound Speed (m/s)"] = "6090"
    s1["Free Surface Velocity at HEL (m/s)"] = "180"
    s1["Longitudinal Stress at HEL (GPa)"] = "3.0"

    s2 = {col: "-" for col in TABLE1_COLUMNS}
    s2["Sample ID"] = "S2"
    s2["Initial Density (g/cm³)"] = "6.11"
    s2["Longitudinal Sound Speed (m/s)"] = "6090"
    s2["Free Surface Velocity at HEL (m/s)"] = "-"  # missing -- S2 can't resolve via formula
    s2["Longitudinal Stress at HEL (GPa)"] = "3.0"

    return f"""## Table 1: Extracted Data

{header}
{separator}
{_row(s1)}
{_row(s2)}

## Table 2: Evidence Source

| Column Name | Source Location | Notes |
|---|---|---|
| Initial Density (g/cm³) | Table I | DIRECT (P1): stated in Table I |
| Longitudinal Sound Speed (m/s) | Table I | DIRECT (P1): stated in Table I |
| Free Surface Velocity at HEL (m/s) | Page 4 | [Priority 1]: explicit precursor plateau value |
| Longitudinal Stress at HEL (GPa) | Page 6, Fig. 5(a) & CALCULATED | [Priority 2/3]: calculated where possible from u_HEL; for other shots, read from Fig. 5(a) |
"""


def test_partially_resolved_field_shows_full_shot_table_but_only_patches_unresolved_samples(tmp_path):
    pass1_text = _pass1_markdown()

    def fake_run_gemini(api_key, model_id, contents, media_resolution="MEDIA_RESOLUTION_HIGH"):
        return pass1_text, {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0, "total_tokens": 150}

    captured_pass1_extracted = {}

    def fake_collect_llm_draws(api_key, model_id, base_prompt, field_name, figure_number, figure_image, pass1_extracted, n_draws, call_refine=None):
        captured_pass1_extracted["value"] = pass1_extracted
        # The mocked model "helpfully" reports a value for S1 too, even
        # though S1 was already resolved by the T2 gate and wasn't asked
        # about -- this must NOT overwrite S1's T2-gate value.
        per_sample = {
            "S1": [{"value": 999.0, "evidence": "misbehaving model reported this anyway"} for _ in range(3)],
            "S2": [{"value": 1.234, "evidence": "read from Fig. 5(a)"} for _ in range(3)],
        }
        usage = {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0, "total_tokens": 15}
        return per_sample, usage

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("base prompt text", encoding="utf-8")
    output_dir = tmp_path / "out"

    with (
        mock.patch.object(m, "upload_file", _fake_upload_file),
        mock.patch.object(m, "run_gemini", fake_run_gemini),
        mock.patch.object(m, "find_figure_page", lambda pdf_path, fig_num: 1),
        mock.patch.object(m, "render_page", lambda pdf_path, page, figures_dir: Path("/tmp/fake-page.png")),
        mock.patch.object(
            m, "locate_figure_bbox",
            lambda api_key, model_id, page_image, fig_num: (
                (0.1, 0.1, 0.9, 0.9),
                {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0, "total_tokens": 2},
            ),
        ),
        mock.patch.object(m, "render_figure_crop", lambda pdf_path, page, bbox, figures_dir: Path("/tmp/fake-crop.png")),
        mock.patch("figure_classify.classify_figure_type", lambda api_key, model_id, image_path: (
            "continuous-curve", {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0, "total_tokens": 2},
        )),
        mock.patch.object(repeated_sampling, "collect_llm_draws", fake_collect_llm_draws),
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

    # The full shot table was shown -- both S1 and S2 -- not just S2.
    shown_sample_ids = set(captured_pass1_extracted["value"]["Sample ID"].astype(str).str.strip())
    assert shown_sample_ids == {"S1", "S2"}

    paper_dir = output_dir / "test" / "TestPaper"
    extracted = pd.read_csv(paper_dir / "extracted_data.csv", dtype=str)
    extracted = extracted.set_index(extracted["Sample ID"].astype(str).str.strip())

    # S1 keeps its T2-gate-computed value -- NOT overwritten by the "999.0"
    # the mocked model reported for it despite not being asked.
    assert float(extracted.loc["S1", "Longitudinal Stress at HEL (GPa)"]) != 999.0
    assert float(extracted.loc["S1", "Longitudinal Stress at HEL (GPa)"]) == pytest.approx(3.348891, rel=1e-3)

    # S2 gets patched with the T3-read value, since it genuinely needed it.
    assert float(extracted.loc["S2", "Longitudinal Stress at HEL (GPa)"]) == pytest.approx(1.234, rel=1e-3)
