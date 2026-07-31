from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from .api_runner import RunSettings, run_paper
from .ai_evaluator import hybrid_numeric_ai_evaluate, write_ai_evaluation_outputs
from .clients import build_client
from .consensus import compare_artifact_roots, write_consensus_outputs
from .config import load_evaluation_config, verify_prompt
from .evaluator import evaluate_fields
from .manifest import load_manifest, select_papers
from .metrics import summarize_metrics
from .normalize import load_ground_truth, load_predictions, normalize_sample_id
from .provenance import prepare_provenance
from .record_matching import apply_record_matching
from .reporting import build_experiment_overview, summarize_run_metadata
from .response_parser import ParseError, parse_response, write_parsed_artifacts
from .schema import TABLE1_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heldout-pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-prompt")
    freeze.add_argument("--check", action="store_true")
    freeze.add_argument("--prompt", type=Path, required=True)
    freeze.add_argument("--prompt-manifest", type=Path, required=True)

    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--split", choices=["development", "heldout"], required=True)
    run.add_argument("--prompt", type=Path, default=Path("prompts/prompty_frozen.md"))
    run.add_argument(
        "--prompt-manifest",
        type=Path,
        default=Path("prompts/prompt_manifest.json"),
    )
    run.add_argument("--model-config", type=Path, default=Path("configs/models.json"))
    run.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    run.add_argument("--force", action="store_true")

    provenance = commands.add_parser("prepare-provenance")
    provenance.add_argument("--manifest", type=Path, required=True)
    provenance.add_argument("--split", choices=["development", "heldout"], required=True)
    provenance.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    provenance.add_argument("--config", type=Path, default=Path("configs/evaluation.json"))
    provenance.add_argument("--output", type=Path)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--split", choices=["development", "heldout"], required=True)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    evaluate.add_argument("--provenance", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, default=Path("configs/evaluation.json"))
    evaluate.add_argument("--sample-crosswalk", type=Path)
    evaluate.add_argument("--record-match", action="store_true")
    evaluate.add_argument("--record-match-min-score", type=float, default=0.55)
    evaluate.add_argument("--record-match-min-anchors", type=int, default=2)
    evaluate.add_argument("--numeric-only", action="store_true")
    evaluate.add_argument("--output-dir", type=Path, required=True)

    report = commands.add_parser("report")
    report.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    report.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    report.add_argument("--original-chatbox-papers", type=int, default=30)

    consensus = commands.add_parser("compare-consensus")
    consensus.add_argument("--manifest", type=Path, required=True)
    consensus.add_argument("--split", choices=["development", "heldout"], required=True)
    consensus.add_argument("--gemini-artifact-root", type=Path, required=True)
    consensus.add_argument("--claude-artifact-root", type=Path, required=True)
    consensus.add_argument("--config", type=Path, default=Path("configs/evaluation.json"))
    consensus.add_argument("--output-dir", type=Path, required=True)

    ai_evaluate = commands.add_parser("ai-evaluate")
    ai_evaluate.add_argument("--input", type=Path, required=True)
    ai_evaluate.add_argument("--model-config", type=Path, default=Path("configs/models.json"))
    ai_evaluate.add_argument("--ai-mode", choices=["hybrid", "all"], default="hybrid")
    ai_evaluate.add_argument("--batch-size", type=int, default=25)
    ai_evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_model_settings(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_evidence(artifact_root: Path, records) -> pd.DataFrame:
    frames = []
    for record in records:
        path = artifact_root / record.split / record.paper_id / "evidence_source.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path).rename(
            columns={
                "Column Name": "field_name",
                "Source Location": "source_location",
                "Notes": "notes",
            }
        )
        frame.insert(0, "paper_id", record.paper_id)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["paper_id", "field_name", "source_location", "notes"]
        )
    return pd.concat(frames, ignore_index=True)


def _run_command(args) -> int:
    model = _load_model_settings(args.model_config)
    settings = RunSettings(
        artifact_root=args.artifact_root,
        prompt_path=args.prompt,
        prompt_manifest_path=args.prompt_manifest,
        model_id=model["model_id"],
        generation_config=model.get("generation_config", {}),
        provider=model["provider"],
        project=model.get("project"),
        location=model.get("location"),
        max_attempts=int(model.get("max_attempts", 2)),
        retry_delay_seconds=float(model.get("retry_delay_seconds", 2)),
        force=args.force,
        input_price_per_million_tokens_usd=model.get(
            "input_price_per_million_tokens_usd"
        ),
        output_price_per_million_tokens_usd=model.get(
            "output_price_per_million_tokens_usd"
        ),
    )
    records = select_papers(load_manifest(args.manifest), args.split)
    client = build_client(model)
    failures = 0
    for index, record in enumerate(records):
        result = run_paper(record, settings, client)
        if result.success:
            raw = (result.artifact_dir / "raw_response.md").read_text(encoding="utf-8")
            try:
                parsed = parse_response(raw, TABLE1_COLUMNS)
                write_parsed_artifacts(parsed, result.artifact_dir)
            except ParseError as exc:
                (result.artifact_dir / "parse_report.json").write_text(
                    json.dumps({"valid": False, "error": str(exc)}, indent=2),
                    encoding="utf-8",
                )
                failures += 1
        else:
            failures += 1
        if index + 1 < len(records):
            time.sleep(float(model.get("inter_paper_delay_seconds", 0)))
    print(f"Processed {len(records)} {args.split} papers; failures={failures}")
    return 1 if failures else 0


def _prepare_provenance_command(args) -> int:
    records = select_papers(load_manifest(args.manifest), args.split)
    predictions = load_predictions(args.artifact_root, records)
    evidence = _load_evidence(args.artifact_root, records)
    config = load_evaluation_config(args.config)
    result = prepare_provenance(
        predictions,
        evidence,
        set(config.external_reference_fields),
    )
    output = args.output or args.artifact_root / args.split / "provenance_review.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} provenance rows to {output}")
    return 0


def _field_comparisons(outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "split",
        "paper_id",
        "sample_id",
        "field_name",
        "field_type",
        "value_ground_truth",
        "value_prediction",
        "outcome",
        "score",
        "score_reason",
        "tier",
        "provenance",
        "review_status",
        "include_primary",
    ]
    result = outcomes[[column for column in columns if column in outcomes.columns]].copy()
    return result.rename(
        columns={
            "value_ground_truth": "ground_truth",
            "value_prediction": "prediction",
        }
    )


def _sample_alignment(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    gt_samples = ground_truth[["paper_id", "sample_id"]].drop_duplicates()
    pred_samples = predictions[["paper_id", "sample_id"]].drop_duplicates()
    merged = gt_samples.merge(
        pred_samples,
        on=["paper_id", "sample_id"],
        how="outer",
        indicator=True,
    )
    status = {
        "both": "matched",
        "left_only": "ground_truth_only",
        "right_only": "prediction_only",
    }
    merged["sample_status"] = merged["_merge"].map(status).astype(str)
    return merged.drop(columns=["_merge"]).sort_values(
        ["paper_id", "sample_status", "sample_id"]
    )


def _load_sample_crosswalk(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"paper_id", "prediction_sample_id", "ground_truth_sample_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"sample crosswalk missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["paper_id"] = frame["paper_id"].astype(str).str.strip()
    frame["prediction_sample_id"] = frame["prediction_sample_id"].map(normalize_sample_id)
    frame["ground_truth_sample_id"] = frame["ground_truth_sample_id"].map(
        normalize_sample_id
    )
    duplicates = frame.duplicated(["paper_id", "prediction_sample_id"], keep=False)
    if duplicates.any():
        keys = (
            frame.loc[duplicates, ["paper_id", "prediction_sample_id"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"duplicate sample crosswalk keys: {keys[:5]}")
    return frame


def _apply_sample_crosswalk(
    frame: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    if frame.empty or crosswalk.empty:
        return frame
    merged = frame.merge(
        crosswalk[["paper_id", "prediction_sample_id", "ground_truth_sample_id"]],
        left_on=["paper_id", "sample_id"],
        right_on=["paper_id", "prediction_sample_id"],
        how="left",
    )
    remapped = merged["ground_truth_sample_id"].notna()
    merged.loc[remapped, "sample_id"] = merged.loc[remapped, "ground_truth_sample_id"]
    result = merged.drop(columns=["prediction_sample_id", "ground_truth_sample_id"])
    duplicates = result.duplicated(["paper_id", "sample_id", "field_name"], keep=False)
    if duplicates.any():
        keys = (
            result.loc[duplicates, ["paper_id", "sample_id", "field_name"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"sample crosswalk creates duplicate {label} keys: {keys[:5]}")
    return result


def _summarize_errors(errors: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame(columns=[*group_by, "outcome", "count"])
    return (
        errors.groupby([*group_by, "outcome"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", *group_by], ascending=[False, *([True] * len(group_by))])
    )


def _write_development_audit(
    output_dir: Path,
    metrics: dict[str, pd.DataFrame],
    comparisons: pd.DataFrame,
    errors: pd.DataFrame,
    sample_alignment: pd.DataFrame,
) -> None:
    errors_by_field = _summarize_errors(errors, ["field_name", "field_type"])
    errors_by_paper = _summarize_errors(errors, ["paper_id"])
    if errors.empty:
        score_reasons = pd.DataFrame(columns=["score_reason", "count"])
    else:
        score_reasons = (
            errors["score_reason"]
            .value_counts(dropna=False)
            .rename_axis("score_reason")
            .reset_index(name="count")
        )
    if sample_alignment.empty:
        sample_summary = pd.DataFrame(columns=["paper_id", "sample_status", "count"])
    else:
        sample_summary = (
            sample_alignment.groupby(["paper_id", "sample_status"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["paper_id", "sample_status"])
        )

    errors_by_field.to_csv(output_dir / "error_summary_by_field.csv", index=False)
    errors_by_paper.to_csv(output_dir / "error_summary_by_paper.csv", index=False)
    score_reasons.to_csv(output_dir / "score_reason_summary.csv", index=False)
    sample_alignment.to_csv(output_dir / "sample_alignment.csv", index=False)
    sample_summary.to_csv(output_dir / "sample_alignment_summary.csv", index=False)

    audit = output_dir / "ground_truth_error_audit.xlsx"
    with pd.ExcelWriter(audit, engine="openpyxl") as writer:
        for sheet_name, frame in metrics.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
        errors_by_field.to_excel(writer, sheet_name="errors_by_field", index=False)
        errors_by_paper.to_excel(writer, sheet_name="errors_by_paper", index=False)
        score_reasons.to_excel(writer, sheet_name="score_reasons", index=False)
        sample_summary.to_excel(writer, sheet_name="sample_alignment_summary", index=False)
        sample_alignment.to_excel(writer, sheet_name="sample_alignment", index=False)
        errors.head(5000).to_excel(writer, sheet_name="error_examples", index=False)
        errors.to_excel(writer, sheet_name="all_errors", index=False)
        comparisons.head(5000).to_excel(writer, sheet_name="all_comparisons_sample", index=False)


def _evaluate_command(args) -> int:
    records = select_papers(load_manifest(args.manifest), args.split)
    paper_ids = {record.paper_id for record in records}
    ground_truth = load_ground_truth(args.ground_truth, paper_ids)
    predictions = load_predictions(args.artifact_root, records)
    provenance = pd.read_csv(args.provenance)
    provenance["paper_id"] = provenance["paper_id"].astype(str).str.strip()
    provenance["sample_id"] = provenance["sample_id"].map(normalize_sample_id)
    crosswalk = pd.DataFrame()
    if args.sample_crosswalk:
        crosswalk = _load_sample_crosswalk(args.sample_crosswalk)
        predictions = _apply_sample_crosswalk(
            predictions,
            crosswalk,
            label="prediction",
        )
        provenance = _apply_sample_crosswalk(
            provenance,
            crosswalk,
            label="provenance",
        )
    record_match_review = pd.DataFrame()
    if args.record_match:
        predictions, provenance, record_match_review = apply_record_matching(
            ground_truth,
            predictions,
            provenance,
            min_score=args.record_match_min_score,
            min_anchors=args.record_match_min_anchors,
        )
    config = load_evaluation_config(args.config)
    outcomes = evaluate_fields(ground_truth, predictions, provenance, config)
    if args.numeric_only:
        outcomes = outcomes[outcomes["field_type"].eq("numeric")].copy()
    outcomes.insert(0, "split", args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(args.output_dir / "field_outcomes.csv", index=False)
    comparisons = _field_comparisons(outcomes)
    comparisons.to_csv(args.output_dir / "field_comparisons.csv", index=False)
    comparisons.to_csv(args.output_dir / "per_field_results.csv", index=False)
    metrics = {
        "overall_metrics": summarize_metrics(outcomes, ["split"]),
        "paper_metrics": summarize_metrics(outcomes, ["paper_id"]),
        "field_metrics": summarize_metrics(outcomes, ["field_name"]),
        "field_type_metrics": summarize_metrics(outcomes, ["field_type"]),
    }
    metrics["overall_metrics"].to_csv(args.output_dir / "overall_metrics.csv", index=False)
    metrics["paper_metrics"].to_csv(args.output_dir / "paper_metrics.csv", index=False)
    metrics["field_metrics"].to_csv(args.output_dir / "field_metrics.csv", index=False)
    metrics["field_type_metrics"].to_csv(
        args.output_dir / "field_type_metrics.csv", index=False
    )
    approved = outcomes[
        (outcomes["review_status"] == "approved") & (outcomes["tier"] != "NA")
    ]
    metrics["tier_metrics"] = summarize_metrics(approved, ["tier"])
    metrics["provenance_metrics"] = summarize_metrics(outcomes, ["provenance"])
    metrics["tier_metrics"].to_csv(args.output_dir / "tier_metrics.csv", index=False)
    metrics["provenance_metrics"].to_csv(
        args.output_dir / "provenance_metrics.csv", index=False
    )
    errors = comparisons[
        ~comparisons["outcome"].isin(["correct_value", "correct_null"])
    ].copy()
    errors.to_csv(args.output_dir / "error_breakdown.csv", index=False)
    alignment = _sample_alignment(ground_truth, predictions)
    if args.record_match:
        record_match_review.to_csv(args.output_dir / "record_match_review.csv", index=False)
    _write_development_audit(args.output_dir, metrics, comparisons, errors, alignment)
    print(
        f"Wrote corrected {args.split} evaluation to {args.output_dir}; "
        f"{len(errors)} field errors and {len(alignment)} sample-alignment rows"
    )
    return 0


def _report_command(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_run_metadata(args.artifact_root)
    summary.to_csv(args.output_dir / "cost_runtime.csv", index=False)
    build_experiment_overview(summary, args.original_chatbox_papers).to_csv(
        args.output_dir / "experiment_overview.csv", index=False
    )
    print(f"Wrote comparison reports to {args.output_dir}")
    return 0


def _compare_consensus_command(args) -> int:
    records = select_papers(load_manifest(args.manifest), args.split)
    config = load_evaluation_config(args.config)
    gemini_evidence = _load_evidence(args.gemini_artifact_root, records)
    claude_evidence = _load_evidence(args.claude_artifact_root, records)
    result = compare_artifact_roots(
        args.gemini_artifact_root,
        args.claude_artifact_root,
        records,
        relative_tolerance=config.relative_tolerance,
        absolute_tolerance=config.absolute_tolerance,
        numeric_floor=config.numeric_floor,
        null_values=config.null_values,
        categorical_fields=config.categorical_fields,
    )
    write_consensus_outputs(
        result,
        args.output_dir,
        gemini_evidence=gemini_evidence,
        claude_evidence=claude_evidence,
    )
    print(
        "Wrote consensus review tables to "
        f"{args.output_dir}: "
        f"{len(result.consensus_candidates)} agreed fields, "
        f"{len(result.disagreement_review)} field disagreements, "
        f"{len(result.record_review)} record review rows"
    )
    return 0


def _build_gemini_text_generator(model: dict):
    provider = model.get("provider")
    model_id = model["model_id"]
    generation_config = dict(model.get("generation_config", {}))
    if provider == "vertex":
        project = model.get("project")
        location = model.get("location")
        if not project or not location:
            raise ValueError("Vertex Gemini text evaluator requires project and location")
        from google import genai
        from google.genai import types

        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(api_version="v1"),
        )
        config = (
            types.GenerateContentConfig(**generation_config)
            if generation_config
            else None
        )

        def generate(prompt: str) -> str:
            response = client.models.generate_content(
                model=model_id,
                contents=[types.Part.from_text(text=prompt)],
                config=config,
            )
            return getattr(response, "text", "") or ""

        return generate
    if provider in {"gemini_api", "gemini_file_api"}:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(f"GEMINI_API_KEY is required for {provider} evaluator")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = (
            types.GenerateContentConfig(**generation_config)
            if generation_config
            else None
        )

        def generate(prompt: str) -> str:
            response = client.models.generate_content(
                model=model_id,
                contents=[types.Part.from_text(text=prompt)],
                config=config,
            )
            return getattr(response, "text", "") or ""

        return generate
    raise ValueError("ai-evaluate currently supports Gemini providers only")


def _ai_evaluate_command(args) -> int:
    comparisons = pd.read_csv(args.input)
    model = _load_model_settings(args.model_config)
    generate = _build_gemini_text_generator(model)
    result = hybrid_numeric_ai_evaluate(
        comparisons,
        generate,
        batch_size=args.batch_size,
        mode=args.ai_mode,
    )
    write_ai_evaluation_outputs(result, args.output_dir)
    print(
        f"Wrote {args.ai_mode} Gemini numeric AI evaluation for "
        f"{len(result)} numeric rows to {args.output_dir}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-prompt":
        digest = verify_prompt(args.prompt, args.prompt_manifest)
        print(f"Prompt hash verified: {digest}")
        return 0
    if args.command == "validate-manifest":
        records = load_manifest(args.manifest)
        included = sum(record.include for record in records)
        print(f"Manifest valid: {included} included papers")
        return 0
    if args.command == "run":
        return _run_command(args)
    if args.command == "prepare-provenance":
        return _prepare_provenance_command(args)
    if args.command == "evaluate":
        return _evaluate_command(args)
    if args.command == "report":
        return _report_command(args)
    if args.command == "compare-consensus":
        return _compare_consensus_command(args)
    if args.command == "ai-evaluate":
        return _ai_evaluate_command(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
