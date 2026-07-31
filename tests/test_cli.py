import hashlib
import json

import heldout_pipeline.cli as cli
from heldout_pipeline.api_runner import RunResult
from heldout_pipeline.cli import main


def test_validate_manifest_command(tmp_path, capsys):
    pdf = tmp_path / "new.pdf"
    pdf.write_bytes(b"%PDF")
    manifest = tmp_path / "papers.csv"
    manifest.write_text(
        "paper_id,pdf_path,split,include,selection_note\n"
        f"H1,{pdf},heldout,true,Selected before freeze\n"
    )
    assert main(["validate-manifest", "--manifest", str(manifest)]) == 0
    assert "1 included papers" in capsys.readouterr().out


def test_freeze_prompt_check_command(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("frozen")
    manifest = tmp_path / "prompt.json"
    manifest.write_text(json.dumps({"sha256": hashlib.sha256(b"frozen").hexdigest()}))
    assert (
        main(
            [
                "freeze-prompt",
                "--check",
                "--prompt",
                str(prompt),
                "--prompt-manifest",
                str(manifest),
            ]
        )
        == 0
    )


def test_evaluate_command_writes_separate_heldout_metrics(tmp_path):
    import json

    import pandas as pd

    pdf = tmp_path / "heldout.pdf"
    pdf.write_bytes(b"%PDF")
    manifest = tmp_path / "papers.csv"
    manifest.write_text(
        "paper_id,pdf_path,split,include,selection_note\n"
        f"H1,{pdf},heldout,true,Selected before freeze\n"
    )
    ground_truth = tmp_path / "ground_truth.xlsx"
    with pd.ExcelWriter(ground_truth, engine="openpyxl") as writer:
        pd.DataFrame(
            {"sheet": ["H1"], "Sample ID": ["S1"], "Value": [10.0]}
        ).to_excel(writer, index=False, startrow=1)
    artifact_dir = tmp_path / "artifacts" / "heldout" / "H1"
    artifact_dir.mkdir(parents=True)
    pd.DataFrame({"Sample ID": ["S1"], "Value": [10.0]}).to_csv(
        artifact_dir / "extracted_data.csv", index=False
    )
    provenance = tmp_path / "provenance.csv"
    pd.DataFrame(
        {
            "paper_id": ["H1"],
            "sample_id": ["s1"],
            "field_name": ["Value"],
            "tier": ["T1"],
            "provenance": ["paper_table"],
            "review_status": ["approved"],
        }
    ).to_csv(provenance, index=False)
    config = tmp_path / "evaluation.json"
    config.write_text(
        json.dumps(
            {
                "relative_tolerance": 0.005,
                "absolute_tolerance": 0,
                "numeric_floor": 1e-12,
                "null_values": ["", "-"],
                "categorical_fields": [],
                "external_reference_fields": [],
                "allowed_tiers": ["T1", "NA"],
                "allowed_provenance": ["paper_table", "unknown"],
            }
        )
    )
    output = tmp_path / "results"

    code = main(
        [
            "evaluate",
            "--manifest",
            str(manifest),
            "--split",
            "heldout",
            "--ground-truth",
            str(ground_truth),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--provenance",
            str(provenance),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    metrics = pd.read_csv(output / "overall_metrics.csv")
    assert metrics.loc[0, "accuracy_non_null"] == 1.0
    assert set(pd.read_csv(output / "field_outcomes.csv")["split"]) == {"heldout"}
    assert (output / "field_type_metrics.csv").exists()
    assert (output / "field_comparisons.csv").exists()
    assert (output / "per_field_results.csv").exists()
    assert (output / "ground_truth_error_audit.xlsx").exists()
    assert (output / "sample_alignment.csv").exists()
    assert pd.read_csv(output / "tier_metrics.csv").columns.tolist() == [
        "tier",
        "correct_value",
        "wrong_value",
        "missing_extraction",
        "false_extraction",
        "correct_null",
        "support",
        "precision",
        "recall",
        "f1",
        "accuracy_all",
        "accuracy_non_null",
        "strict_accuracy_all",
        "strict_accuracy_non_null",
    ]


def test_evaluate_command_applies_sample_crosswalk_to_predictions_and_provenance(tmp_path):
    import json

    import pandas as pd

    pdf = tmp_path / "development.pdf"
    pdf.write_bytes(b"%PDF")
    manifest = tmp_path / "papers.csv"
    manifest.write_text(
        "paper_id,pdf_path,split,include,selection_note\n"
        f"D1,{pdf},development,true,Original benchmark\n"
    )
    ground_truth = tmp_path / "ground_truth.xlsx"
    with pd.ExcelWriter(ground_truth, engine="openpyxl") as writer:
        pd.DataFrame(
            {"sheet": ["D1"], "Sample ID": ["shot_1"], "Value": [10.0]}
        ).to_excel(writer, index=False, startrow=1)
    artifact_dir = tmp_path / "artifacts" / "development" / "D1"
    artifact_dir.mkdir(parents=True)
    pd.DataFrame({"Sample ID": ["1"], "Value": [10.0]}).to_csv(
        artifact_dir / "extracted_data.csv", index=False
    )
    provenance = tmp_path / "provenance.csv"
    pd.DataFrame(
        {
            "paper_id": ["D1"],
            "sample_id": ["1"],
            "field_name": ["Value"],
            "tier": ["T1"],
            "provenance": ["paper_table"],
            "review_status": ["approved"],
        }
    ).to_csv(provenance, index=False)
    crosswalk = tmp_path / "crosswalk.csv"
    crosswalk.write_text(
        "paper_id,prediction_sample_id,ground_truth_sample_id,note\n"
        "D1,1,shot_1,shot prefix omitted\n"
    )
    config = tmp_path / "evaluation.json"
    config.write_text(
        json.dumps(
            {
                "relative_tolerance": 0.005,
                "absolute_tolerance": 0,
                "numeric_floor": 1e-12,
                "null_values": ["", "-"],
                "categorical_fields": [],
                "external_reference_fields": [],
                "allowed_tiers": ["T1", "NA"],
                "allowed_provenance": ["paper_table", "unknown"],
            }
        )
    )
    output = tmp_path / "results"

    code = main(
        [
            "evaluate",
            "--manifest",
            str(manifest),
            "--split",
            "development",
            "--ground-truth",
            str(ground_truth),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--provenance",
            str(provenance),
            "--sample-crosswalk",
            str(crosswalk),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    comparisons = pd.read_csv(output / "field_comparisons.csv")
    assert comparisons.loc[0, "sample_id"] == "shot_1"
    assert comparisons.loc[0, "outcome"] == "correct_value"
    assert comparisons.loc[0, "review_status"] == "approved"
    alignment = pd.read_csv(output / "sample_alignment_summary.csv")
    assert alignment.to_dict("records") == [
        {"paper_id": "D1", "sample_status": "matched", "count": 1}
    ]


def test_evaluate_command_can_record_match_without_exact_sample_ids(tmp_path):
    import json

    import pandas as pd

    pdf = tmp_path / "development.pdf"
    pdf.write_bytes(b"%PDF")
    manifest = tmp_path / "papers.csv"
    manifest.write_text(
        "paper_id,pdf_path,split,include,selection_note\n"
        f"D1,{pdf},development,true,Original benchmark\n"
    )
    ground_truth = tmp_path / "ground_truth.xlsx"
    with pd.ExcelWriter(ground_truth, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "sheet": ["D1"],
                "Sample ID": ["shot_1"],
                "Metal Symbol": ["Cu"],
                "Impact Velocity (m/s)": [300],
                "Spall Strength (GPa)": [1.2],
            }
        ).to_excel(writer, index=False, startrow=1)
    artifact_dir = tmp_path / "artifacts" / "development" / "D1"
    artifact_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Sample ID": ["1"],
            "Metal Symbol": ["Cu"],
            "Impact Velocity (m/s)": [301],
            "Spall Strength (GPa)": [1.2],
        }
    ).to_csv(artifact_dir / "extracted_data.csv", index=False)
    provenance = tmp_path / "provenance.csv"
    pd.DataFrame(
        {
            "paper_id": ["D1"],
            "sample_id": ["1"],
            "field_name": ["Spall Strength (GPa)"],
            "tier": ["T1"],
            "provenance": ["paper_table"],
            "review_status": ["approved"],
        }
    ).to_csv(provenance, index=False)
    config = tmp_path / "evaluation.json"
    config.write_text(
        json.dumps(
            {
                "relative_tolerance": 0.005,
                "absolute_tolerance": 0,
                "numeric_floor": 1e-12,
                "null_values": ["", "-"],
                "categorical_fields": ["Metal Symbol"],
                "external_reference_fields": [],
                "allowed_tiers": ["T1", "NA"],
                "allowed_provenance": ["paper_table", "unknown"],
            }
        )
    )
    output = tmp_path / "results"

    code = main(
        [
            "evaluate",
            "--manifest",
            str(manifest),
            "--split",
            "development",
            "--ground-truth",
            str(ground_truth),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--provenance",
            str(provenance),
            "--record-match",
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    review = pd.read_csv(output / "record_match_review.csv")
    assert review.loc[0, "record_status"] == "record_matched"
    assert review.loc[0, "ground_truth_sample_id"] == "shot_1"
    assert str(review.loc[0, "prediction_sample_id"]) == "1"
    comparisons = pd.read_csv(output / "field_comparisons.csv")
    assert set(comparisons["sample_id"]) == {"shot_1"}


def test_evaluate_command_can_score_numeric_fields_only(tmp_path):
    import json

    import pandas as pd

    pdf = tmp_path / "development.pdf"
    pdf.write_bytes(b"%PDF")
    manifest = tmp_path / "papers.csv"
    manifest.write_text(
        "paper_id,pdf_path,split,include,selection_note\n"
        f"D1,{pdf},development,true,Original benchmark\n"
    )
    ground_truth = tmp_path / "ground_truth.xlsx"
    with pd.ExcelWriter(ground_truth, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "sheet": ["D1"],
                "Sample ID": ["S1"],
                "Metal Symbol": ["Cu"],
                "Impact Velocity (m/s)": [100.0],
                "Spall Strength (GPa)": ["-"],
            }
        ).to_excel(writer, index=False, startrow=1)
    artifact_dir = tmp_path / "artifacts" / "development" / "D1"
    artifact_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Sample ID": ["S1"],
            "Metal Symbol": ["Al"],
            "Impact Velocity (m/s)": [104.0],
            "Spall Strength (GPa)": ["-"],
        }
    ).to_csv(artifact_dir / "extracted_data.csv", index=False)
    provenance = tmp_path / "provenance.csv"
    pd.DataFrame(
        {
            "paper_id": ["D1", "D1", "D1"],
            "sample_id": ["S1", "S1", "S1"],
            "field_name": [
                "Metal Symbol",
                "Impact Velocity (m/s)",
                "Spall Strength (GPa)",
            ],
            "tier": ["T1", "T1", "T1"],
            "provenance": ["paper_table", "paper_table", "paper_table"],
            "review_status": ["approved", "approved", "approved"],
        }
    ).to_csv(provenance, index=False)
    config = tmp_path / "evaluation.json"
    config.write_text(
        json.dumps(
            {
                "relative_tolerance": 0.005,
                "absolute_tolerance": 0,
                "numeric_floor": 1e-12,
                "null_values": ["", "-"],
                "categorical_fields": ["Metal Symbol"],
                "external_reference_fields": [],
                "allowed_tiers": ["T1", "NA"],
                "allowed_provenance": ["paper_table", "unknown"],
            }
        )
    )
    output = tmp_path / "results"

    code = main(
        [
            "evaluate",
            "--manifest",
            str(manifest),
            "--split",
            "development",
            "--ground-truth",
            str(ground_truth),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--provenance",
            str(provenance),
            "--numeric-only",
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    comparisons = pd.read_csv(output / "field_comparisons.csv")
    assert set(comparisons["field_name"]) == {
        "Impact Velocity (m/s)",
        "Spall Strength (GPa)",
    }
    assert set(comparisons["field_type"]) == {"numeric"}
    metrics = pd.read_csv(output / "overall_metrics.csv").iloc[0]
    assert metrics["correct_null"] == 1
    assert metrics["wrong_value"] == 1
    assert metrics["accuracy_all"] == 0.9


def test_ai_evaluate_command_writes_gemini_numeric_scores(tmp_path, monkeypatch):
    import json

    import pandas as pd

    comparisons = tmp_path / "field_comparisons.csv"
    pd.DataFrame(
        [
            {
                "split": "development",
                "paper_id": "P1",
                "sample_id": "S1",
                "field_name": "Impact Velocity (m/s)",
                "field_type": "numeric",
                "ground_truth": "100",
                "prediction": "~101",
                "outcome": "wrong_value",
                "score": 0.0,
                "score_reason": "mismatch",
                "tier": "T1",
                "provenance": "paper_table",
                "review_status": "approved",
                "include_primary": True,
            }
        ]
    ).to_csv(comparisons, index=False)
    model_config = tmp_path / "model.json"
    model_config.write_text(
        json.dumps(
            {
                "provider": "vertex",
                "project": "project",
                "location": "global",
                "model_id": "gemini-3.1-pro-preview",
                "generation_config": {},
            }
        )
    )

    def fake_generator(model):
        return lambda prompt: json.dumps(
            [
                {
                    "row_id": 0,
                    "score": 0.95,
                    "reason": "rounding",
                    "normalized_ground_truth": "100",
                    "normalized_prediction": "101",
                }
            ]
        )

    monkeypatch.setattr(cli, "_build_gemini_text_generator", fake_generator)
    output = tmp_path / "ai_eval"

    code = main(
        [
            "ai-evaluate",
            "--input",
            str(comparisons),
            "--model-config",
            str(model_config),
            "--ai-mode",
            "all",
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    scores = pd.read_csv(output / "ai_numeric_field_scores.csv")
    assert scores.loc[0, "ai_final_score"] == 0.95
    assert scores.loc[0, "ai_evaluation_source"] == "llm"
    assert (output / "overall_metrics.csv").exists()
    assert (output / "ai_numeric_evaluation.xlsx").exists()


def test_run_command_passes_vertex_identity_to_runner(tmp_path, monkeypatch):
    pdf = tmp_path / "development.pdf"
    pdf.write_bytes(b"%PDF")
    manifest = tmp_path / "papers.csv"
    manifest.write_text(
        "paper_id,pdf_path,split,include,selection_note\n"
        f"D1,{pdf},development,true,Original benchmark\n"
    )
    prompt = tmp_path / "prompt.md"
    prompt.write_text("frozen")
    prompt_manifest = tmp_path / "prompt.json"
    prompt_manifest.write_text(
        json.dumps({"sha256": hashlib.sha256(b"frozen").hexdigest()})
    )
    model_config = tmp_path / "model.json"
    model_config.write_text(
        json.dumps(
            {
                "provider": "vertex",
                "project": "dataextractionllm-503420",
                "location": "global",
                "model_id": "gemini-3.1-pro-preview",
                "generation_config": {},
                "inter_paper_delay_seconds": 0,
            }
        )
    )
    artifact_root = tmp_path / "artifacts"
    captured = {}

    def fake_build_client(model):
        captured["client_config"] = model
        return object()

    def fake_run_paper(record, settings, client):
        captured["settings"] = settings
        artifact_dir = artifact_root / record.split / record.paper_id
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "raw_response.md").write_text("fake")
        return RunResult(
            record.paper_id,
            record.split,
            True,
            False,
            artifact_dir,
        )

    monkeypatch.setattr(cli, "build_client", fake_build_client, raising=False)
    monkeypatch.setattr(cli, "run_paper", fake_run_paper)
    monkeypatch.setattr(cli, "parse_response", lambda raw, columns: object())
    monkeypatch.setattr(cli, "write_parsed_artifacts", lambda parsed, path: None)

    code = main(
        [
            "run",
            "--manifest",
            str(manifest),
            "--split",
            "development",
            "--prompt",
            str(prompt),
            "--prompt-manifest",
            str(prompt_manifest),
            "--model-config",
            str(model_config),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    assert code == 0
    assert captured["client_config"]["provider"] == "vertex"
    assert captured["settings"].provider == "vertex"
    assert captured["settings"].project == "dataextractionllm-503420"
    assert captured["settings"].location == "global"


def test_compare_consensus_command_writes_review_tables(tmp_path):
    import pandas as pd

    pdf = tmp_path / "heldout.pdf"
    pdf.write_bytes(b"%PDF")
    manifest = tmp_path / "papers.csv"
    manifest.write_text(
        "paper_id,pdf_path,split,include,selection_note\n"
        f"H1,{pdf},heldout,true,Selected before freeze\n"
    )
    gemini_dir = tmp_path / "gemini" / "heldout" / "H1"
    claude_dir = tmp_path / "claude" / "heldout" / "H1"
    gemini_dir.mkdir(parents=True)
    claude_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Sample ID": ["S1"],
            "Metal Symbol": ["Cu"],
            "Spall Strength (GPa)": [5.0],
        }
    ).to_csv(gemini_dir / "extracted_data.csv", index=False)
    pd.DataFrame(
        {
            "Sample ID": ["S1"],
            "Metal Symbol": ["Cu"],
            "Spall Strength (GPa)": [5.01],
        }
    ).to_csv(claude_dir / "extracted_data.csv", index=False)
    output = tmp_path / "consensus"

    code = main(
        [
            "compare-consensus",
            "--manifest",
            str(manifest),
            "--split",
            "heldout",
            "--gemini-artifact-root",
            str(tmp_path / "gemini"),
            "--claude-artifact-root",
            str(tmp_path / "claude"),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    assert (output / "consensus_candidates.csv").exists()
    assert (output / "disagreement_review.csv").exists()
    assert (output / "record_review.csv").exists()
    assert (output / "candidate_ground_truth.xlsx").exists()
