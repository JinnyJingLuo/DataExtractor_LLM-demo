import pandas as pd

from heldout_pipeline.consensus import compare_artifact_roots, write_consensus_outputs
from heldout_pipeline.manifest import PaperRecord


def _write_prediction(root, split, paper_id, rows):
    path = root / split / paper_id
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "extracted_data.csv", index=False)


def test_consensus_accepts_numeric_tolerance_and_text_mechanical_matches(tmp_path):
    gemini = tmp_path / "gemini"
    claude = tmp_path / "claude"
    _write_prediction(
        gemini,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "S-1",
                "Metal Symbol": "Cu",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Spall Strength (GPa)": 5.000,
                "Treatment": "Lapping (0.3 mrad)",
            }
        ],
    )
    _write_prediction(
        claude,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "s_1",
                "Metal Symbol": "cu",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Spall Strength (GPa)": 5.024,
                "Treatment": "lapping 0.3 mrad",
            }
        ],
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    records = [PaperRecord("H1", pdf, "heldout", True, "new")]

    result = compare_artifact_roots(gemini, claude, records)

    agreed = result.consensus_candidates
    assert set(agreed["field_name"]) >= {
        "Metal Symbol",
        "Sample Thickness (mm)",
        "Impact Velocity (m/s)",
        "Spall Strength (GPa)",
        "Treatment",
    }
    spall = agreed[agreed["field_name"].eq("Spall Strength (GPa)")].iloc[0]
    assert spall["value"] == 5.0
    assert spall["agreement_type"] == "numeric_within_tolerance"
    assert spall["sample_id"] == "S-1"
    assert spall["claude_sample_id"] == "s_1"
    assert spall["normalized_sample_id"] == "s_1"
    assert result.disagreement_review.empty
    assert result.record_review.empty


def test_consensus_flags_semantic_text_disagreement_and_unmatched_records(tmp_path):
    gemini = tmp_path / "gemini"
    claude = tmp_path / "claude"
    _write_prediction(
        gemini,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "A",
                "Metal Symbol": "Ag",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Synthesis Method": "Annealing",
            },
            {
                "Sample ID": "B",
                "Metal Symbol": "Cu",
                "Sample Thickness (mm)": 2.0,
                "Impact Velocity (m/s)": 400.0,
                "Synthesis Method": "Casting",
            },
        ],
    )
    _write_prediction(
        claude,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "A",
                "Metal Symbol": "Ag",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Synthesis Method": "Annealed",
            }
        ],
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    records = [PaperRecord("H1", pdf, "heldout", True, "new")]

    result = compare_artifact_roots(gemini, claude, records)

    disagreement = result.disagreement_review
    assert disagreement.loc[0, "field_name"] == "Synthesis Method"
    assert disagreement.loc[0, "reason"] == "text_mismatch"
    record_review = result.record_review
    assert record_review.loc[0, "status"] == "RECORD_REVIEW_REQUIRED"
    assert record_review.loc[0, "gemini_sample_id"] == "B"
    assert record_review.loc[0, "claude_sample_id"] == ""


def test_consensus_matches_unique_records_by_experimental_anchors(tmp_path):
    gemini = tmp_path / "gemini"
    claude = tmp_path / "claude"
    _write_prediction(
        gemini,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "ModelA",
                "Metal Symbol": "Ta",
                "Initial Temperature (K)": 296,
                "Sample Thickness (mm)": 1.5,
                "Impact Velocity (m/s)": 500,
                "Spall Strength (GPa)": 7.0,
            }
        ],
    )
    _write_prediction(
        claude,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "Table row 1",
                "Metal Symbol": "Ta",
                "Initial Temperature (K)": 296,
                "Sample Thickness (mm)": 1.5,
                "Impact Velocity (m/s)": 500,
                "Spall Strength (GPa)": 7.0,
            }
        ],
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    records = [PaperRecord("H1", pdf, "heldout", True, "new")]

    result = compare_artifact_roots(gemini, claude, records)

    spall = result.consensus_candidates[
        result.consensus_candidates["field_name"].eq("Spall Strength (GPa)")
    ].iloc[0]
    assert spall["sample_id"] == "ModelA"
    assert spall["claude_sample_id"] == "Table row 1"
    assert spall["match_method"] == "anchor_fields"
    assert result.record_review.empty


def test_consensus_requires_strict_numeric_cells(tmp_path):
    gemini = tmp_path / "gemini"
    claude = tmp_path / "claude"
    _write_prediction(
        gemini,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "S1",
                "Metal Symbol": "Cu",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Spall Strength (GPa)": "5 GPa",
            }
        ],
    )
    _write_prediction(
        claude,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "S1",
                "Metal Symbol": "Cu",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Spall Strength (GPa)": "5.0",
            }
        ],
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    records = [PaperRecord("H1", pdf, "heldout", True, "new")]

    result = compare_artifact_roots(gemini, claude, records)

    assert result.consensus_candidates[
        result.consensus_candidates["field_name"].eq("Spall Strength (GPa)")
    ].empty
    assert result.disagreement_review.iloc[0]["reason"] == "text_mismatch"


def test_consensus_uses_gemini_value_as_relative_tolerance_baseline(tmp_path):
    gemini = tmp_path / "gemini"
    claude = tmp_path / "claude"
    _write_prediction(
        gemini,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "S1",
                "Metal Symbol": "Cu",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Spall Strength (GPa)": 100.0,
            }
        ],
    )
    _write_prediction(
        claude,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "S1",
                "Metal Symbol": "Cu",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Spall Strength (GPa)": 100.5001,
            }
        ],
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    records = [PaperRecord("H1", pdf, "heldout", True, "new")]

    result = compare_artifact_roots(gemini, claude, records)

    assert result.consensus_candidates[
        result.consensus_candidates["field_name"].eq("Spall Strength (GPa)")
    ].empty
    assert result.disagreement_review.iloc[0]["reason"] == "numeric_mismatch"


def test_consensus_keeps_hardness_as_text_when_config_omits_it(tmp_path):
    gemini = tmp_path / "gemini"
    claude = tmp_path / "claude"
    _write_prediction(
        gemini,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "S1",
                "Metal Symbol": "Cu",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Hardness": "HV 90",
            }
        ],
    )
    _write_prediction(
        claude,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "S1",
                "Metal Symbol": "Cu",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Hardness": "HRB 90",
            }
        ],
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    records = [PaperRecord("H1", pdf, "heldout", True, "new")]

    result = compare_artifact_roots(gemini, claude, records, categorical_fields=[])

    hardness = result.disagreement_review[
        result.disagreement_review["field_name"].eq("Hardness")
    ].iloc[0]
    assert hardness["reason"] == "text_mismatch"


def test_consensus_does_not_anchor_match_on_static_material_only(tmp_path):
    gemini = tmp_path / "gemini"
    claude = tmp_path / "claude"
    _write_prediction(
        gemini,
        "heldout",
        "H1",
        [{"Sample ID": "G1", "Metal Symbol": "Cu"}],
    )
    _write_prediction(
        claude,
        "heldout",
        "H1",
        [{"Sample ID": "C1", "Metal Symbol": "Cu"}],
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    records = [PaperRecord("H1", pdf, "heldout", True, "new")]

    result = compare_artifact_roots(gemini, claude, records)

    assert result.consensus_candidates.empty
    assert set(result.record_review["reason"]) == {
        "gemini_record_unmatched",
        "claude_record_unmatched",
    }


def test_consensus_manual_workbook_includes_evidence_and_original_ids(tmp_path):
    gemini = tmp_path / "gemini"
    claude = tmp_path / "claude"
    _write_prediction(
        gemini,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "880_Al_1",
                "Metal Symbol": "Al",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Spall Strength (GPa)": 5.0,
            }
        ],
    )
    _write_prediction(
        claude,
        "heldout",
        "H1",
        [
            {
                "Sample ID": "880-Al-1",
                "Metal Symbol": "Al",
                "Sample Thickness (mm)": 1.0,
                "Impact Velocity (m/s)": 300.0,
                "Spall Strength (GPa)": 6.0,
            }
        ],
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    records = [PaperRecord("H1", pdf, "heldout", True, "new")]
    result = compare_artifact_roots(gemini, claude, records)
    evidence = pd.DataFrame(
        [
            {
                "paper_id": "H1",
                "field_name": "Spall Strength (GPa)",
                "source_location": "Table 1",
                "notes": "direct value",
            }
        ]
    )

    output = tmp_path / "consensus"
    write_consensus_outputs(
        result,
        output,
        gemini_evidence=evidence,
        claude_evidence=evidence,
    )

    review = pd.read_excel(output / "manual_review_with_evidence.xlsx", sheet_name="manual_field_review")
    spall = review[review["field_name"].eq("Spall Strength (GPa)")].iloc[0]
    assert spall["gemini_sample_id"] == "880_Al_1"
    assert spall["claude_sample_id"] == "880-Al-1"
    assert spall["normalized_sample_id"] == "880_al_1"
    assert spall["Gemini generated result"] == 5.0
    assert spall["Claude Opus generated result"] == 6.0
    assert spall["Gemini evidence location"] == "Table 1"
    assert spall["Claude Opus evidence notes"] == "direct value"
