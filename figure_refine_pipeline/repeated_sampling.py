"""Collect N repeated draws of the existing refine_field() crop-read call,
per sample, for downstream reconciliation. Does not decide how many draws
to take (see reconcile.draw_count) or how to reconcile them (see
reconcile.cluster_and_reconcile) -- this module only collects.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def collect_llm_draws(
    api_key: str,
    model_id: str,
    base_prompt: str,
    field_name: str,
    figure_number: str,
    figure_image: Path,
    pass1_extracted: pd.DataFrame,
    n_draws: int,
    call_refine=None,
) -> tuple[dict[str, list[dict]], dict]:
    if call_refine is None:
        from run_figure_refine import refine_field as call_refine  # noqa: N813
    from run_figure_refine import _sum_usage

    per_sample: dict[str, list[dict]] = {}
    usages = []
    for draw_idx in range(1, n_draws + 1):
        try:
            patch, usage = call_refine(
                api_key, model_id, base_prompt, field_name, figure_number, figure_image, pass1_extracted
            )
        except Exception as exc:  # noqa: BLE001 -- isolate one flaky draw (parse
            # failure, transient network error -- google-genai already retries
            # transient errors internally via tenacity, so anything reaching
            # here has survived that and is worth logging, not retrying again)
            # from the rest. Losing 1 of N draws should cost that one draw,
            # not every draw already collected.
            print(f"    [draw {draw_idx}/{n_draws}] failed for '{field_name}', skipping this draw: {exc}")
            continue
        usages.append(usage)
        for _, row in patch.iterrows():
            sample_id = str(row["Sample ID"]).strip()
            raw_value = row[field_name]
            evidence = str(row.get("Evidence Note", ""))
            try:
                value = float(str(raw_value).split("±")[0].strip())
            except (ValueError, TypeError):
                continue  # "-" or unparseable: this draw has nothing for this sample
            per_sample.setdefault(sample_id, []).append({"value": value, "evidence": evidence})
    return per_sample, _sum_usage(*usages)


def collect_cv_anchor(image_path: Path, y_min: float, y_max: float) -> list[dict]:
    """Run the validated CV marker detector once. Deterministic -- pixel
    measurement doesn't vary between calls, so unlike the LLM-only path
    this is never repeated for the value itself (see reconcile.draw_count:
    3 draws with a CV anchor are spent on shot-matching, not measurement)."""
    from digitize_figure import classify_markers, find_axis_box, load_gray, pixel_to_value

    gray = load_gray(image_path)
    box = find_axis_box(gray)
    markers = classify_markers(gray, box)
    filled = sorted([m for m in markers if m["kind"] == "filled"], key=lambda m: m["cx"])
    for i, marker in enumerate(filled, start=1):
        marker["marker_number"] = i
        marker["value"] = pixel_to_value(marker["cy"], box, y_min, y_max)
    return filled


def collect_match_draws(
    api_key: str,
    model_id: str,
    image_path: Path,
    filled_markers: list[dict],
    field_name: str,
    figure_number: str,
    pass1_extracted: pd.DataFrame,
    n_draws: int,
    call_match=None,
) -> tuple[dict[str, list[str]], dict]:
    from hybrid_match import MATCH_PROMPT_TEMPLATE, annotate_markers, parse_match_table
    from run_figure_refine import _sum_usage, build_shot_table

    if call_match is None:

        def call_match(api_key, model_id, image_path, prompt):  # noqa: ANN001
            from run_figure_refine import run_gemini, upload_file

            client, uploaded = upload_file(api_key, image_path)
            text, usage = run_gemini(api_key, model_id, [uploaded, prompt])
            client.files.delete(name=uploaded.name)
            return text, usage

    annotated_path = image_path.with_name(image_path.stem + "_annotated.png")
    if image_path.exists():
        from digitize_figure import load_gray

        annotate_markers(load_gray(image_path), filled_markers, annotated_path)

    shot_table = build_shot_table(pass1_extracted)
    prompt = MATCH_PROMPT_TEMPLATE.format(
        figure_number=figure_number, field_name=field_name, shot_table=shot_table
    )

    per_sample: dict[str, list[str]] = {}
    usages = []
    for draw_idx in range(1, n_draws + 1):
        try:
            text, usage = call_match(api_key, model_id, annotated_path, prompt)
            usages.append(usage)
            mapping = parse_match_table(text)
        except Exception as exc:  # noqa: BLE001 -- same rationale as collect_llm_draws:
            # isolate one flaky draw from the rest rather than losing all of them.
            print(f"    [draw {draw_idx}/{n_draws}] failed for '{field_name}', skipping this draw: {exc}")
            continue
        for sample_id, marker_label in mapping.items():
            per_sample.setdefault(str(sample_id).strip(), []).append(marker_label)
    return per_sample, _sum_usage(*usages)
