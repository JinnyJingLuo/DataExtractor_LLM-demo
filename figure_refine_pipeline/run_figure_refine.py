#!/usr/bin/env python3
"""
Experimental two-pass extraction pipeline with targeted figure refinement.

NOT part of the frozen-prompt reproducible pipeline in src/heldout_pipeline.
This is a standalone exploratory script: it may modify the prompt, is not
hash-verified, and is not wired into the `heldout-pipeline` CLI.

Design (multi-resolution, T3-only patching):
  1. Pass 1: run the base extraction prompt against the whole PDF, unmodified.
     This is the sole source of truth for T1 (direct text) and T2 (derived)
     fields -- they are never touched again.
  2. Parse Table 2 (evidence) for fields whose source is a figure
     (Priority 3 / "Fig. N" reference) -- these are the only candidates for
     refinement.
  3. For each such field, locate the PDF page containing the referenced
     figure's caption and render that whole page as a standalone
     high-resolution PNG (image media has a higher resolution ceiling in the
     Gemini API than PDF-page media).
  4. Pass 2 (one call per flagged field): give the model ONLY the high-res
     figure image plus the shot list (Sample ID + distinguishing conditions
     from Pass 1), and ask it to re-read *just that one field* per shot. This
     keeps Pass 2 narrow so it cannot regress anything Pass 1 already got
     right.
  5. Merge: overwrite only the flagged field's column in Pass 1's table with
     Pass 2's values. T1/T2 fields are copied through unchanged.

Output artifacts are written in the layout heldout_pipeline's
prepare-provenance/evaluate commands expect, so the existing evaluator can be
reused unmodified.

Usage:
  export GEMINI_API_KEY=...
  python run_figure_refine.py \
      --pdf "../Papers/Sample Papers/Paper1.pdf" \
      --prompt ../prompts/prompty_frozen.md \
      --paper-id Paper1 \
      --split development \
      --output-dir ./artifacts/paper1_figrefine
"""
from __future__ import annotations

import argparse
import math
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from heldout_pipeline.response_parser import (
    ParseError,
    _table_blocks,
    _parse_block,
    parse_response,
    write_parsed_artifacts,
)
from heldout_pipeline.schema import TABLE1_COLUMNS
from reconcile import ReconciliationResult

FIG_RE = re.compile(r"\b(?:Fig\.?|Figure)\s*(\d+)", re.IGNORECASE)
DIRECT_RE = re.compile(r"\bDIRECT\b|\bP(?:RIORITY)?\s*1\b|\bT1\b", re.IGNORECASE)
PRIORITY3_RE = re.compile(r"\bP(?:RIORITY)?\s*3\b|\bT3\b|FIGURE", re.IGNORECASE)
CALC_RE = re.compile(r"CALCULATED|\bP(?:RIORITY)?\s*2\b", re.IGNORECASE)

# Columns to show the model as shot-matching context in the targeted re-read
# call, in addition to Sample ID. Kept small on purpose -- just enough to
# disambiguate which plotted point belongs to which shot.
MATCH_CONTEXT_COLUMNS = [
    "Initial Temperature (K)",
    "Sample Thickness (mm)",
    "Impact Velocity (m/s)",
]

# poppler-utils (pdfinfo/pdftotext/pdftoppm) are linked against the system
# glibc; this venv sets LD_LIBRARY_PATH to nix-store libs so numpy/pandas can
# import, which is incompatible with the system poppler binaries. Strip it
# just for these subprocess calls.
_POPPLER_ENV = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}

# IMPORTANT: this addendum is appended to the *full, unmodified* frozen
# prompt (all of it -- symbol disambiguation, figure-extraction protocol,
# unit rules, etc.), not used on its own. The base prompt is what taught the
# model how to correctly read this exact figure (e.g. distinguishing
# overlapping dual-axis series) in the working manual test; a short custom
# instruction on its own was tried and produced a systematic misread because
# it dropped that guidance. Do not strip the base prompt out here again.
TARGETED_ADDENDUM_TEMPLATE = """

---

## TARGETED RE-READ TASK (OVERRIDES THE OUTPUT FORMAT ABOVE FOR THIS CALL ONLY)

You already extracted the full Table 1 / Table 2 for this paper in a previous
pass; those results are correct and are not being redone here. Your only job
in this call is to re-read ONE column, using a high-resolution image of the
figure it comes from (attached), applying all the rules above that are
relevant to reading figures, disambiguating symbols/axes, and preserving
uncertainty notation.

Column to re-read: "{field_name}"
This value comes from Figure {figure_number} (high-resolution image attached above).

Below is the list of experimental shots already identified in this paper,
with distinguishing conditions to help you match each plotted data point to
the correct shot:

{shot_table}

Instructions specific to this task:
- Use ONLY the attached high-resolution figure image to read "{field_name}" for each shot listed above.
- Apply Section C (symbol disambiguation) and Section G (figure extraction protocol) from the rules above -- in particular, if the figure has multiple series/axes, verify which series and which axis corresponds to "{field_name}" before reading values, exactly as those sections instruct.
- Match each plotted point to a shot using the distinguishing conditions given (temperature, thickness, impact velocity, etc.). Multiple shots with the same conditions may share the same plotted value.
- If a shot's value cannot be determined from this figure, write "-". Never invent a value.
- Preserve uncertainty/error-bar notation exactly as shown (e.g. "1.28±0.05").

Ignore the "Table 1: Extracted Data" / "Table 2: Evidence Source" output format from the rules above for this call only. Instead, output ONLY a single Markdown table with exactly these columns, no other text:

| Sample ID | {field_name} | Evidence Note |
| --- | --- | --- |
"""


def _fmt_usage(usage: dict) -> str:
    return (
        f"input={usage.get('input_tokens')} output={usage.get('output_tokens')} "
        f"thinking={usage.get('thinking_tokens')} total={usage.get('total_tokens')}"
    )


def _sum_usage(*usages: dict) -> dict:
    total = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
    for usage in usages:
        for key in total:
            total[key] += usage.get(key) or 0
    return total


def run_gemini(api_key: str, model_id: str, contents: list, media_resolution: str = "MEDIA_RESOLUTION_HIGH"):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        media_resolution=getattr(types.MediaResolution, media_resolution)
    )
    response = client.models.generate_content(model=model_id, contents=contents, config=config)
    usage = getattr(response, "usage_metadata", None)
    return response.text or "", {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "thinking_tokens": getattr(usage, "thoughts_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }


def upload_file(api_key: str, path: Path):
    from google import genai

    client = genai.Client(api_key=api_key)
    return client, client.files.upload(file=str(path))


def find_figure_fields(evidence_rows: list[dict]) -> dict[str, str]:
    """Return {field_name: figure_number} for Priority-3 (figure-sourced) fields."""
    fields: dict[str, str] = {}
    for row in evidence_rows:
        field_name = str(row.get("Column Name", "")).strip()
        notes = str(row.get("Notes", ""))
        location = str(row.get("Source Location", ""))
        text = f"{notes} {location}"
        if not field_name or not PRIORITY3_RE.search(text):
            continue
        match = FIG_RE.search(text)
        if match:
            fields[field_name] = match.group(1)
    return fields


def find_direct_fields(evidence_rows: list[dict]) -> set[str]:
    """Return fields whose evidence indicates a direct/Priority-1 source."""
    fields: set[str] = set()
    for row in evidence_rows:
        field_name = str(row.get("Column Name", "")).strip()
        notes = str(row.get("Notes", ""))
        location = str(row.get("Source Location", ""))
        text = f"{notes} {location}"
        if field_name and DIRECT_RE.search(text):
            fields.add(field_name)
    return fields


def _has_nonmissing_values(frame: pd.DataFrame, field_name: str) -> bool:
    if field_name not in frame.columns:
        return False
    values = frame[field_name].astype(str).str.strip().str.casefold()
    return not values.isin(["", "-", "nan", "none"]).any()


def filter_t1_resolved_fields(
    figure_fields: dict[str, str], evidence_rows: list[dict], extracted: pd.DataFrame
) -> tuple[dict[str, str], set[str]]:
    """Remove figure candidates already fully resolved by direct evidence.

    This is deliberately field-level and conservative: it only removes a field
    if the selected Pass-1 table has non-missing values for every sample. Mixed
    cases stay eligible for T2/T3 handling.
    """
    direct_fields = find_direct_fields(evidence_rows)
    resolved = {
        field_name
        for field_name in figure_fields
        if field_name in direct_fields and _has_nonmissing_values(extracted, field_name)
    }
    return (
        {field_name: fig_num for field_name, fig_num in figure_fields.items() if field_name not in resolved},
        resolved,
    )


def find_figure_page(pdf_path: Path, figure_number: str) -> int | None:
    """Find the page containing Figure N's actual caption.

    A plain "Fig. N" mention also matches in-text citations like "As shown in
    Fig. 25(a), ..." which commonly appear on earlier pages than the figure
    itself. Captions are reliably distinguished by the number being followed
    immediately by a period (e.g. "Fig. 25. Comparison of ..."), which
    citations are not -- they're followed by "(a)", a comma, "shows", etc.
    Prefer a caption match; only fall back to a loose mention if no page has
    an actual caption for this figure.
    """
    info = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True, env=_POPPLER_ENV
    ).stdout
    pages_match = re.search(r"Pages:\s*(\d+)", info)
    total_pages = int(pages_match.group(1)) if pages_match else 0
    caption_pattern = re.compile(
        rf"\bFIG\.?\s*{figure_number}\s*[.:]|\bFigure\s*{figure_number}\s*[.:]", re.IGNORECASE
    )
    mention_pattern = re.compile(rf"\bFIG\.?\s*{figure_number}\b|\bFigure\s*{figure_number}\b", re.IGNORECASE)
    first_mention_page = None
    for page in range(1, total_pages + 1):
        text = subprocess.run(
            ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf_path), "-"],
            capture_output=True, text=True, check=True, env=_POPPLER_ENV,
        ).stdout
        if caption_pattern.search(text):
            return page
        if first_mention_page is None and mention_pattern.search(text):
            first_mention_page = page
    if first_mention_page is not None:
        return first_mention_page
    return None


def render_page(pdf_path: Path, page: int, output_dir: Path, dpi: int = 200) -> Path:
    """Render a whole page (default: modest DPI, used only for bbox localization)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"page{page}"
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-f", str(page), "-l", str(page), str(pdf_path), str(prefix)],
        check=True, env=_POPPLER_ENV,
    )
    candidates = sorted(output_dir.glob(f"page{page}-*.png"))
    if not candidates:
        raise RuntimeError(f"pdftoppm did not produce an image for page {page}")
    return candidates[0]


def get_page_size_pts(pdf_path: Path) -> tuple[float, float]:
    info = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True, env=_POPPLER_ENV
    ).stdout
    match = re.search(r"Page size:\s*([\d.]+)\s*x\s*([\d.]+)\s*pts", info)
    if not match:
        raise RuntimeError("could not determine page size from pdfinfo output")
    return float(match.group(1)), float(match.group(2))


BBOX_PROMPT_TEMPLATE = """This image shows one page of a scientific paper, which may contain multiple figures.

Locate Figure {figure_number} specifically (identified by its caption starting "FIG. {figure_number}" or "Figure {figure_number}").

Return ONLY a JSON object (no other text, no markdown fences) giving a bounding box tightly around Figure {figure_number}'s plot area -- including its axes, data points/curves, tick labels, axis titles, and legend -- but NOT the caption paragraph below it and NOT any other figure or body text. Use fractional coordinates where (0,0) is the top-left corner of the image and (1,1) is the bottom-right corner:

{{"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}}
"""


class LocalizationError(RuntimeError):
    """Like RuntimeError, but carries the token usage of the call that led
    to the failure -- localization can fail after a real, billed API call
    (bad JSON, out-of-range bbox, degenerate area), and that usage would
    otherwise be silently lost since the function never returns normally."""

    def __init__(self, message: str, usage: dict):
        super().__init__(message)
        self.usage = usage


def locate_figure_bbox(
    api_key: str, model_id: str, page_image: Path, figure_number: str
) -> tuple[tuple[float, float, float, float], dict]:
    client, uploaded_img = upload_file(api_key, page_image)
    prompt = BBOX_PROMPT_TEMPLATE.format(figure_number=figure_number)
    text, usage = run_gemini(api_key, model_id, [uploaded_img, prompt])
    client.files.delete(name=uploaded_img.name)
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        raise LocalizationError(
            f"could not parse bounding-box JSON from localization response: {text!r}", usage
        )
    box = json.loads(match.group(0))
    raw_bbox = (
        float(box["x_min"]), float(box["y_min"]), float(box["x_max"]), float(box["y_max"]),
    )
    try:
        bbox = normalize_bbox_to_fraction(raw_bbox, png_size(page_image))
    except ValueError as exc:
        raise LocalizationError(f"localization returned out-of-range bounding box: {raw_bbox}: {exc}", usage) from exc
    x_min, y_min, x_max, y_max = bbox
    area = (x_max - x_min) * (y_max - y_min)
    # A degenerate/near-zero box means the model couldn't actually find the
    # figure on this page (e.g. the page-finder pointed at the wrong page).
    if area < 0.01:
        raise LocalizationError(
            f"localization returned a degenerate bounding box (area={area:.4f}) -- "
            f"figure {figure_number} probably isn't on this page",
            usage,
        )
    return bbox, usage


def png_size(path: Path) -> tuple[int, int]:
    """Return PNG image size without adding an imaging dependency."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG image: {path}")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return width, height


def normalize_bbox_to_fraction(
    bbox: tuple[float, float, float, float], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Normalize bbox coordinates to fractions.

    Gemini sometimes returns mixed coordinates such as fractional x values and
    pixel y values. Normalize each axis independently.
    """
    width, height = image_size
    x_min, y_min, x_max, y_max = bbox

    def normalize_axis(v_min: float, v_max: float, size: int) -> tuple[float, float]:
        if 0.0 <= v_min <= 1.0 and 0.0 <= v_max <= 1.0:
            return v_min, v_max
        if 0.0 <= v_min <= size and 0.0 <= v_max <= size:
            return v_min / size, v_max / size
        raise ValueError(f"out-of-range axis coordinates: {(v_min, v_max)} for size {size}")

    x_min, x_max = normalize_axis(x_min, x_max, width)
    y_min, y_max = normalize_axis(y_min, y_max, height)
    if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
        raise ValueError(f"out-of-range bounding box after normalization: {bbox}")
    return x_min, y_min, x_max, y_max


def render_figure_crop(
    pdf_path: Path,
    page: int,
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    dpi: int = 600,
    margin: float = 0.03,
) -> Path:
    """Render a tight, high-DPI crop of just the figure region, direct from the PDF."""
    page_w_pts, page_h_pts = get_page_size_pts(pdf_path)
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0.0, x_min - margin)
    y_min = max(0.0, y_min - margin)
    x_max = min(1.0, x_max + margin)
    y_max = min(1.0, y_max + margin)

    px_w = page_w_pts * dpi / 72
    px_h = page_h_pts * dpi / 72
    x, y = int(x_min * px_w), int(y_min * px_h)
    w, h = int((x_max - x_min) * px_w), int((y_max - y_min) * px_h)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"page{page}_bbox_crop"
    subprocess.run(
        [
            "pdftoppm", "-png", "-r", str(dpi),
            "-f", str(page), "-l", str(page),
            "-x", str(x), "-y", str(y), "-W", str(w), "-H", str(h),
            str(pdf_path), str(prefix),
        ],
        check=True, env=_POPPLER_ENV,
    )
    candidates = sorted(output_dir.glob(f"page{page}_bbox_crop-*.png"))
    if not candidates:
        raise RuntimeError(f"pdftoppm did not produce a cropped image for page {page}")
    return candidates[0]


def load_evidence_rows(csv_path: Path) -> list[dict]:
    import csv

    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_pass1_cell(value) -> str:
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none"}:
        return "-"
    return re.sub(r"\s+", " ", text)


def _normalize_sample_id_for_matching(value) -> str:
    """Collapse '-' and '_' separator variants for cross-draw lookup only.

    Pass-1 formats Sample IDs inconsistently between calls -- observed on
    the same paper, same prompt: draw 1 wrote "VA0_300", draws 2 and 3 wrote
    "VA0-300". Comparing raw strings across draws made every sample look
    "missing" from every other draw, producing false low-confidence results
    for the entire table. This is used only as a lookup key; the original,
    unnormalized Sample ID string is still what gets reported and what
    downstream code (the extracted table, the T2 gate) keys on.
    """
    text = str(value).strip()
    return re.sub(r"[-_]+", "_", text)


def _parse_float(value: str) -> float | None:
    if value in {"-", "<MISSING_SAMPLE>", "<MISSING_FIELD>"}:
        return None
    cleaned = value.replace(",", "")
    cleaned = cleaned.split("+/-")[0].split("±")[0].strip()
    cleaned = re.sub(r"^[~≈]", "", cleaned).strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _pass1_values_agree(selected_value: str, observed_value: str) -> bool:
    selected_numeric = _parse_float(selected_value)
    observed_numeric = _parse_float(observed_value)
    if selected_numeric is not None and observed_numeric is not None:
        return math.isclose(selected_numeric, observed_numeric, rel_tol=0.005, abs_tol=1e-12)
    return selected_value.casefold() == observed_value.casefold()


def summarize_pass1_consistency(
    selected: pd.DataFrame, draws: list[pd.DataFrame]
) -> pd.DataFrame:
    """Compare repeated Pass-1 outputs against the selected Pass-1 table.

    This is an audit artifact only. It does not change the selected extraction,
    so downstream evaluation remains comparable with the original one-pass run.
    """
    rows = []
    fields = [column for column in selected.columns if column != "Sample ID"]
    indexed_draws = []
    for draw in draws:
        if "Sample ID" not in draw.columns:
            continue
        normalized_draw = draw.assign(
            **{"Sample ID": draw["Sample ID"].map(_normalize_sample_id_for_matching)}
        )
        indexed_draws.append(
            normalized_draw.drop_duplicates(subset=["Sample ID"], keep="first").set_index("Sample ID")
        )
    for _, selected_row in selected.iterrows():
        sample_id = str(selected_row["Sample ID"]).strip()
        lookup_key = _normalize_sample_id_for_matching(sample_id)
        for field_name in fields:
            selected_value = _normalize_pass1_cell(selected_row[field_name])
            observed_values = []
            for draw in indexed_draws:
                if lookup_key not in draw.index:
                    observed_values.append("<MISSING_SAMPLE>")
                    continue
                if field_name not in draw.columns:
                    observed_values.append("<MISSING_FIELD>")
                    continue
                observed_values.append(_normalize_pass1_cell(draw.loc[lookup_key, field_name]))

            agreement_count = sum(
                _pass1_values_agree(selected_value, value) for value in observed_values
            )
            compared_draws = len(observed_values)
            agreement_fraction = agreement_count / compared_draws if compared_draws else 0.0
            if agreement_fraction == 1.0:
                confidence = "high"
            elif agreement_fraction >= 0.5:
                confidence = "medium"
            else:
                confidence = "low"
            rows.append(
                {
                    "sample_id": sample_id,
                    "field_name": field_name,
                    "selected_value": selected_value,
                    "agreement_count": agreement_count,
                    "compared_draws": compared_draws,
                    "agreement_fraction": agreement_fraction,
                    "confidence": confidence,
                    "observed_values_json": json.dumps(observed_values),
                }
            )
    return pd.DataFrame(rows)


def build_shot_table(extracted: pd.DataFrame) -> str:
    columns = ["Sample ID"] + [c for c in MATCH_CONTEXT_COLUMNS if c in extracted.columns]
    subset = extracted[columns]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in subset.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def parse_targeted_table(text: str, field_name: str) -> pd.DataFrame:
    blocks = _table_blocks(text)
    if not blocks:
        raise ParseError("targeted re-read response contained no Markdown table")
    frame = _parse_block(blocks[0])
    if "Sample ID" not in frame.columns or field_name not in frame.columns:
        raise ParseError(f"targeted re-read table missing required columns: {list(frame.columns)}")
    return frame


def refine_field(
    api_key: str,
    model_id: str,
    base_prompt: str,
    field_name: str,
    figure_number: str,
    figure_image: Path,
    pass1_extracted: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    client, uploaded_img = upload_file(api_key, figure_image)
    shot_table = build_shot_table(pass1_extracted)
    addendum = TARGETED_ADDENDUM_TEMPLATE.format(
        field_name=field_name, figure_number=figure_number, shot_table=shot_table
    )
    # Full original frozen prompt + narrow output-format override, not a
    # standalone custom instruction -- see note above TARGETED_ADDENDUM_TEMPLATE.
    prompt = base_prompt + addendum
    text, usage = run_gemini(api_key, model_id, [uploaded_img, prompt])
    client.files.delete(name=uploaded_img.name)
    frame = parse_targeted_table(text, field_name)
    return frame, {**usage, "raw_text": text}


def apply_patch(
    extracted: pd.DataFrame,
    evidence: pd.DataFrame,
    field_name: str,
    figure_number: str,
    sample_results: dict,
) -> None:
    """Apply per-sample reconciliation results to extracted[field_name].

    Only overwrites samples with confidence == "high"; low-confidence
    samples keep their Pass-1 value and get flagged in a "Needs Review"
    column instead of being silently blanked. Evidence text is appended
    to, not replaced, so the original Pass-1 reasoning survives.
    """
    if "Needs Review" not in extracted.columns:
        extracted["Needs Review"] = ""

    patched = 0
    flagged = 0
    for idx, sample_id in extracted["Sample ID"].astype(str).str.strip().items():
        result = sample_results.get(sample_id)
        if result is None:
            continue
        if result.confidence == "high":
            extracted.at[idx, field_name] = result.value
            patched += 1
        else:
            existing_flag = str(extracted.at[idx, "Needs Review"] or "")
            extracted.at[idx, "Needs Review"] = (
                f"{existing_flag}; {field_name}" if existing_flag else field_name
            )
            flagged += 1
    print(f"    patched {patched}/{len(extracted)} rows, flagged {flagged} for review, for '{field_name}'")

    note_addition = (
        f"[Priority 3 - HIGH-RES REVIEW]: re-read from cropped high-resolution "
        f"image of Figure {figure_number} ({patched} patched, {flagged} flagged for review)"
    )
    mask = evidence["Column Name"] == field_name
    if mask.any():
        existing_notes = evidence.loc[mask, "Notes"].iloc[0]
        evidence.loc[mask, "Notes"] = f"{existing_notes}; {note_addition}"
    else:
        evidence.loc[len(evidence)] = {c: "" for c in evidence.columns} | {
            "Column Name": field_name,
            "Source Location": f"Fig {figure_number} (high-res crop)",
            "Notes": note_addition,
        }


def run_pipeline(
    pdf_path: Path,
    prompt_path: Path,
    paper_id: str,
    split: str,
    output_dir: Path,
    model_id: str,
    api_key: str,
    pass1_draws: int = 1,
    pass1_max_draws: int | None = None,
    llm_draws: int = 5,
    cv_match_draws: int = 3,
) -> None:
    paper_dir = output_dir / split / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = paper_dir / "figure_crops"

    base_prompt = prompt_path.read_text(encoding="utf-8")

    # --- Pass 1: baseline extraction (source of truth for T1/T2) ---
    print(
        f"[pass 1] extracting {paper_id} with base prompt "
        f"(unmodified, draws={pass1_draws})..."
    )
    client, uploaded_pdf = upload_file(api_key, pdf_path)
    # Pass 1 reads the whole document; MEDIA_RESOLUTION_HIGH here roughly
    # doubles the visual-token budget per page across the entire paper, which
    # empirically hurt cross-page attribution (Paper1 T2, Paper12's VA0 shot
    # misassigned to the wrong curve) without a compensating benefit -- more
    # local pixel detail per page, at the cost of global coherence across
    # pages. Use the provider default here; HIGH is reserved for the
    # figure-crop refinement call below, where the task is purely local.
    effective_max_draws = pass1_max_draws if pass1_max_draws is not None else pass1_draws
    if effective_max_draws < pass1_draws:
        raise ValueError("pass1_max_draws must be >= pass1_draws")

    pass1_attempts = []
    call_usages = []

    def _take_pass1_draw(draw_idx: int) -> None:
        print(f"  [pass 1] draw {draw_idx}")
        text, usage = run_gemini(
            api_key, model_id, [uploaded_pdf, base_prompt], media_resolution="MEDIA_RESOLUTION_UNSPECIFIED"
        )
        call_usages.append(usage)
        (paper_dir / f"raw_response_pass1_draw{draw_idx}.md").write_text(text, encoding="utf-8")
        print(f"  [tokens] pass 1 draw {draw_idx}: {_fmt_usage(usage)}")
        try:
            parsed = parse_response(text, TABLE1_COLUMNS)
        except ParseError as exc:
            print(f"  [pass 1] draw {draw_idx} FAILED to parse: {exc}")
            return
        pass1_attempts.append({"draw_idx": draw_idx, "text": text, "usage": usage, "parsed": parsed})

    for draw_idx in range(1, pass1_draws + 1):
        _take_pass1_draw(draw_idx)

    if not pass1_attempts:
        client.files.delete(name=uploaded_pdf.name)
        print("[pass 1] FAILED: no successful parsed draws")
        return

    selected_pass1 = pass1_attempts[0]
    usage1 = selected_pass1["usage"]
    (paper_dir / "raw_response_pass1.md").write_text(selected_pass1["text"], encoding="utf-8")

    # --- Reconcile Table 1 (values) and Table 2 (source tiers) across every
    # draw collected so far, instead of trusting whichever draw ran first
    # for everything downstream. See pass1_reconcile.py for the mechanism --
    # numeric fields reconcile via tolerance-cluster + median, text fields
    # (including which source-tier a field's evidence indicates) via
    # majority/plurality vote. Degenerates cleanly to "just use draw #1" as
    # pass1_draws=1's default behavior.
    from pass1_reconcile import reconcile_evidence_tiers, reconcile_pass1_table

    def _reconcile_current_attempts():
        extracted_tables = [attempt["parsed"].extracted_data for attempt in pass1_attempts]
        evidence_tables = [attempt["parsed"].evidence_source for attempt in pass1_attempts]
        extracted_reconciled, table_audit = reconcile_pass1_table(
            selected_pass1["parsed"].extracted_data, extracted_tables
        )
        tier_votes = reconcile_evidence_tiers(evidence_tables)
        return extracted_reconciled, table_audit, tier_votes

    extracted, pass1_table_audit, tier_votes = _reconcile_current_attempts()

    # --- Adaptive re-draw: only for fields whose evidence-tier vote is a
    # genuine tie (no plurality leader -- see reconcile.majority_vote_categorical)
    # AND that mention a figure in at least one draw, since only those ties
    # could actually change T1/T2-gate/T3 routing. Off by default: with no
    # --pass1-max-draws override, effective_max_draws == pass1_draws, so
    # this loop never executes. Each extra draw is a full whole-document
    # Pass-1 call -- not cheap -- hence the narrow trigger and explicit ceiling.
    while len(pass1_attempts) < effective_max_draws:
        unresolved_routing_ties = [
            field_name
            for field_name, info in tier_votes.items()
            if info["tier"] is None and info["figure_number"] is not None
        ]
        if not unresolved_routing_ties:
            break
        print(
            f"  [pass 1] routing tie unresolved for {sorted(unresolved_routing_ties)}; "
            f"taking another draw ({len(pass1_attempts) + 1}/{effective_max_draws})"
        )
        _take_pass1_draw(len(pass1_attempts) + 1)
        extracted, pass1_table_audit, tier_votes = _reconcile_current_attempts()

    client.files.delete(name=uploaded_pdf.name)

    pass1_table_audit.to_csv(paper_dir / "pass1_reconciliation.csv", index=False)
    if len(pass1_attempts) > 1:
        pass1_consistency = summarize_pass1_consistency(
            extracted, [attempt["parsed"].extracted_data for attempt in pass1_attempts]
        )
        pass1_consistency.to_csv(paper_dir / "pass1_consistency.csv", index=False)
        unstable = pass1_consistency[pass1_consistency["confidence"].ne("high")]
        print(
            f"  [pass 1] consistency: {len(unstable)}/{len(pass1_consistency)} "
            "selected cells varied across parsed draws"
        )

    # Build evidence (Table 2) from the winning tier's own representative
    # Notes text per field, not draw #1's -- so a field whose majority tier
    # disagrees with draw #1's tag (e.g. the circular-derivation case)
    # actually reflects the majority's reasoning downstream, not the
    # outvoted minority draw's.
    evidence = selected_pass1["parsed"].evidence_source.copy()
    for field_name, info in tier_votes.items():
        if info["tier"] is None:
            continue
        mask = evidence["Column Name"] == field_name
        if mask.any():
            evidence.loc[mask, "Notes"] = info["representative_note"]

    # --- T2 gate: dual-path fields (both a calc and a figure source,
    # decided by majority vote across draws -- see pass1_reconcile.py) get
    # a chance to resolve deterministically, per-sample, before ever
    # touching a figure. Narrow scope: only fields the vote flagged
    # dual-path -- see t2_gate.py docstring.
    from t2_gate import (
        CalculationSpec,
        identify_calculation,
        known_calculation_specs,
        known_confusable_specs,
        order_fields_by_dependency,
        resolve_via_t2_gate,
        validate_input_provenance,
    )
    from audit_log import write_audit_log

    evidence_rows = load_evidence_rows_from_frame(evidence)
    known_specs = known_calculation_specs(list(extracted.columns))
    known_confusables = known_confusable_specs(list(extracted.columns))

    t1_resolved_fields = {
        field_name
        for field_name, info in tier_votes.items()
        if info["tier"] == "direct" and _has_nonmissing_values(extracted, field_name)
    }
    if t1_resolved_fields:
        print(
            "[t1-gate] direct fields already resolved; skipping T2/T3: "
            f"{sorted(t1_resolved_fields)}"
        )

    # Cross-check "direct"-tier fields against a known physical relationship
    # to OTHER direct fields, when one exists. "direct" skips the T2/T3 gate
    # entirely and is trusted as reported -- but "the paper stated it
    # plainly" and "it's the specific quantity the schema wants" aren't the
    # same claim. Observed on Paper12: "Young's Modulus" was read straight
    # from Table I as 226 GPa by 2 of 3 draws, but the table actually
    # reports Longitudinal Modulus (E'), not Young's Modulus (E) -- a
    # labeling ambiguity only one minority draw noticed, computing the real
    # E = 9BG/(3B+G) instead. Majority vote picked the confidently-wrong
    # table read, and nothing downstream ever cross-checked it. Only
    # cross-checks against OTHER direct fields, so the alternative isn't
    # itself built on something uncertain.
    #
    # A bare disagreement only proves the reported value doesn't match the
    # target's own formula -- it doesn't say which one is wrong. But if the
    # reported value instead fits a DIFFERENT, easily-confused quantity
    # (known_confusable_specs) far better, that's not ambiguity anymore --
    # it's a positive, checkable identification of a mislabeled column, and
    # worth correcting (with the full reasoning kept, so it's still
    # verifiable, not silent). Absent that, fall back to flagging the bare
    # disagreement for a human to judge.
    for field_name in sorted(t1_resolved_fields):
        calc_spec = known_specs.get(field_name)
        if calc_spec is None or not all(
            column in t1_resolved_fields for column in calc_spec.variables.values()
        ):
            continue
        computed_by_sample = resolve_via_t2_gate(calc_spec, extracted)
        candidates = known_confusables.get(field_name, [])
        for idx, row in extracted.iterrows():
            sample_id = str(row["Sample ID"]).strip()
            computed_value = computed_by_sample.get(sample_id)
            if computed_value is None:
                continue
            try:
                reported_value = float(str(row[field_name]).split("±")[0].strip())
            except (ValueError, TypeError):
                continue
            if reported_value == 0:
                continue
            rel_diff = abs(computed_value - reported_value) / abs(reported_value)
            if rel_diff <= 0.15:
                continue

            best_match = None
            for candidate in candidates:
                candidate_value = resolve_via_t2_gate(candidate.calc_spec, extracted).get(sample_id)
                if candidate_value is None:
                    continue
                candidate_rel_diff = abs(candidate_value - reported_value) / abs(reported_value)
                if candidate_rel_diff <= 0.05 and (best_match is None or candidate_rel_diff < best_match[2]):
                    best_match = (candidate, candidate_value, candidate_rel_diff)

            if "Needs Review" not in extracted.columns:
                extracted["Needs Review"] = ""
            existing_flag = str(extracted.at[idx, "Needs Review"] or "")

            if best_match is not None:
                candidate, candidate_value, candidate_rel_diff = best_match
                note = (
                    f"{field_name} (AUTO-CORRECTED from reported {reported_value:.4g}: that value "
                    f"matched {candidate.label} ({candidate.calc_spec.formula}={candidate_value:.4g}) "
                    f"within {candidate_rel_diff * 100:.1f}% but disagreed with {field_name}'s own "
                    f"formula {calc_spec.formula}={computed_value:.4g} by {rel_diff * 100:.0f}% -- "
                    f"source table likely reports {candidate.label}, not {field_name}; verify against source)"
                )
                extracted.at[idx, field_name] = computed_value
                print(
                    f"  [t1-gate] '{field_name}' for {sample_id}: reported {reported_value:.4g} matches "
                    f"{candidate.label} ({candidate_rel_diff * 100:.1f}%), not {field_name} -- "
                    f"auto-corrected to {computed_value:.4g}"
                )
            else:
                note = (
                    f"{field_name} (reported {reported_value:.4g} disagrees with "
                    f"{calc_spec.formula}={computed_value:.4g} by {rel_diff * 100:.0f}%"
                    " -- check which quantity the source table actually reports)"
                )
                print(
                    f"  [t1-gate] '{field_name}' for {sample_id}: reported {reported_value:.4g} "
                    f"vs computed {computed_value:.4g} ({rel_diff * 100:.0f}% off) -- flagged for review"
                )
            extracted.at[idx, "Needs Review"] = f"{existing_flag}; {note}" if existing_flag else note

    dual_path_fields = {
        field_name: info["figure_number"]
        for field_name, info in tier_votes.items()
        if info["tier"] == "dual-path" and info["figure_number"] and field_name not in t1_resolved_fields
    }
    figure_fields = {
        field_name: info["figure_number"]
        for field_name, info in tier_votes.items()
        if info["tier"] == "figure-only" and info["figure_number"] and field_name not in t1_resolved_fields
    }
    # Genuine ties (no plurality tier) still route to T3 if any draw
    # mentioned a figure -- the safest fallback when draws can't agree on
    # how a field was even sourced, matching how every other ambiguous case
    # in this pipeline defers to the figure-read + reconciliation path
    # rather than guessing.
    for field_name, info in tier_votes.items():
        if (
            info["tier"] is None
            and info["figure_number"]
            and field_name not in t1_resolved_fields
            and field_name not in dual_path_fields
        ):
            figure_fields.setdefault(field_name, info["figure_number"])

    # Fields whose tier vote is "calculated-only" (a formula, but no draw
    # ever mentioned a figure at all) never had their calculation's inputs
    # checked for reliability -- dual_path_fields gets that check via
    # validate_input_provenance below, but a calculated-only field went
    # straight from reconcile_pass1_table's raw numeric clustering into the
    # output with no scrutiny of whether the formula's inputs were
    # trustworthy. Observed concretely on Paper12: "Spall Pullback Velocity"
    # was calculated by 2 of 4 draws from "Spall Strength" -- itself a
    # visually-extracted, uncertain field -- while a 3rd draw explicitly
    # declined, citing a missing correction parameter and compounding
    # uncertainty. Ground truth has no value for this field at all. Routing
    # these fields through the same T2-gate validation closes that gap.
    calculated_only_fields = {
        field_name: info["figure_number"] or ""
        for field_name, info in tier_votes.items()
        if info["tier"] == "calculated-only" and field_name not in t1_resolved_fields
    }

    t2_gate_fields = dict(dual_path_fields)
    t2_gate_fields.update(calculated_only_fields)
    for field_name in known_specs:
        if field_name in figure_fields or field_name in dual_path_fields or field_name in calculated_only_fields:
            t2_gate_fields.setdefault(
                field_name,
                figure_fields.get(field_name, dual_path_fields.get(field_name, calculated_only_fields.get(field_name, ""))),
            )
    print(f"[t2-gate] fields: {t2_gate_fields or 'none'}")

    def _blank_unverified_field(field_name: str, reason: str, sample_ids=None) -> None:
        # There's nowhere left to get a reliable value from (no figure to
        # fall back to for a rejected calculation, or -- see the unclear-tier
        # pass below -- no draw could even characterize this value's source
        # in the first place). Report "-" rather than keep whatever
        # reconcile_pass1_table's raw clustering produced, since that
        # clustering never checked whether the underlying value was sound.
        # sample_ids=None means every sample; otherwise only those listed.
        if "Needs Review" not in extracted.columns:
            extracted["Needs Review"] = ""
        target_ids = set(sample_ids) if sample_ids is not None else None
        for idx, sample_id in extracted["Sample ID"].astype(str).str.strip().items():
            if target_ids is not None and sample_id not in target_ids:
                continue
            extracted.at[idx, field_name] = "-"
            existing_flag = str(extracted.at[idx, "Needs Review"] or "")
            note = f"{field_name} ({reason})"
            extracted.at[idx, "Needs Review"] = f"{existing_flag}; {note}" if existing_flag else note
        scope = f"{len(target_ids)} sample(s)" if target_ids is not None else "all samples"
        print(f"  [blank] '{field_name}': blanked for {scope} instead of reporting an unverified number ({reason})")

    # Fields whose evidence-tier vote is "unclear" (most draws couldn't even
    # characterize where this value came from -- not direct, calculated, or
    # figure-sourced) get no scrutiny from any of the gates below, since
    # none of them check for this tier by name; reconcile_pass1_table's raw
    # per-cell clustering just ships whatever it produced. That's dangerous
    # specifically when that clustering's OWN confidence for the same cell
    # is also not "high": since "-" draws are filtered out before numeric
    # clustering, a single lone draw's number can trivially "win" with
    # nothing else to compete against it. Observed concretely on Paper1:
    # "Peak Stress / Hugoniot Stress" had 1 of 3 draws calculate a number
    # via an approximation while the other 2 explicitly said the
    # calculation couldn't be done reliably from the paper's data -- ground
    # truth has no value for it at all. Requiring BOTH signals (unclear
    # tier AND low value-confidence) avoids blanking a field whose evidence
    # text just happened to be worded outside the DIRECT_RE/CALC_RE/FIG_RE
    # patterns while every draw still agreed on the same value.
    unclear_tier_fields = {
        field_name for field_name, info in tier_votes.items() if info["tier"] == "unclear"
    }
    if unclear_tier_fields and not pass1_table_audit.empty:
        low_confidence_cells = pass1_table_audit[
            pass1_table_audit["field_name"].isin(unclear_tier_fields)
            & (pass1_table_audit["confidence"] != "high")
        ]
        for field_name, group in low_confidence_cells.groupby("field_name"):
            _blank_unverified_field(
                field_name,
                "evidence tier is unclear (most draws could not characterize this value's "
                "source) and cross-draw value agreement is also low",
                sample_ids=group["sample_id"].astype(str).tolist(),
            )

    # Identify each field's calculation up front (rather than lazily inside
    # the resolution loop below) so fields can be processed in dependency
    # order -- a field whose formula uses another T2-gate field as an input
    # (e.g. "Shear Stress at HEL" needs "Longitudinal Stress at HEL", which
    # the T2 gate itself computes deterministically) needs that upstream
    # field resolved first, or validate_input_provenance has no
    # T2-gate-resolved value to trust it against yet -- see
    # order_fields_by_dependency in t2_gate.py. Iterating t2_gate_fields.items()
    # in plain dict-insertion order (the previous behavior) got this backwards
    # for exactly this pair on Paper12, incorrectly blanking "Shear Stress at
    # HEL" even though its input had, moments later in the same loop, been
    # resolved just fine.
    field_calc_specs: dict[str, CalculationSpec | None] = {}
    field_calc_errors: dict[str, Exception] = {}
    for field_name in t2_gate_fields:
        try:
            note_row = evidence[evidence["Column Name"] == field_name]
            evidence_note = note_row["Notes"].iloc[0] if not note_row.empty else ""
            if field_name in known_specs:
                field_calc_specs[field_name] = known_specs[field_name]
                print(f"  [t2-gate] using known formula for '{field_name}'")
            else:
                field_calc_specs[field_name] = identify_calculation(
                    field_name, evidence_note, list(extracted.columns), api_key, model_id
                )
        except (RuntimeError, ValueError, SyntaxError, ZeroDivisionError) as exc:
            field_calc_specs[field_name] = None
            field_calc_errors[field_name] = exc

    ordered_field_names = order_fields_by_dependency(field_calc_specs)

    unresolved_samples_by_field: dict[str, list[str]] = {}
    t2_gate_resolved_fields: dict[str, CalculationSpec] = {}
    for field_name in ordered_field_names:
        fig_num = t2_gate_fields[field_name]
        has_figure_fallback = bool(fig_num)
        try:
            if field_name in field_calc_errors:
                raise field_calc_errors[field_name]
            calc_spec = field_calc_specs[field_name]
            if calc_spec is None:
                if has_figure_fallback:
                    print(f"  [t2-gate] no closed-form formula identified for '{field_name}', falling through to T3")
                    unresolved_samples_by_field[field_name] = extracted["Sample ID"].astype(str).str.strip().tolist()
                else:
                    _blank_unverified_field(field_name, "no closed-form formula identified, no figure available")
                continue
            valid, reason = validate_input_provenance(
                calc_spec, evidence_rows, field_name=field_name, resolved_by_t2_gate=t2_gate_resolved_fields
            )
            if not valid:
                if has_figure_fallback:
                    print(f"  [t2-gate] rejecting formula for '{field_name}': {reason}; falling through to T3")
                    unresolved_samples_by_field[field_name] = extracted["Sample ID"].astype(str).str.strip().tolist()
                else:
                    _blank_unverified_field(field_name, reason)
                continue
            if field_name not in extracted.columns:
                raise ValueError(f"T2 gate computed '{field_name}' but that column does not exist in extracted data")

            # Input stability across draws is already accounted for here:
            # `extracted` was built by reconcile_pass1_table (tolerance-cluster
            # reconciliation), so a T2-gate input column's value is already
            # the cross-draw-reconciled one, not draw #1's raw value. A
            # separate post-hoc stability check on top of that (the old
            # filter_unstable_inputs, which compared against exact-string
            # consistency) was both redundant and used a less accurate
            # signal -- removed.
            resolved = resolve_via_t2_gate(calc_spec, extracted)
            # Only trust this field as an input for a later, dependent field
            # once it has actually been computed here -- not merely once its
            # provenance was judged valid -- so a downstream field never
            # trusts a formula that turned out to error out below.
            t2_gate_resolved_fields[field_name] = calc_spec
            still_unresolved = []
            t2_gate_results: dict[str, ReconciliationResult] = {}
            for sample_id, value in resolved.items():
                if value is None:
                    still_unresolved.append(sample_id)
                    continue
                row_mask = extracted["Sample ID"].astype(str).str.strip() == sample_id
                pass1_value = None
                existing_raw = extracted.loc[row_mask, field_name]
                if not existing_raw.empty:
                    try:
                        pass1_value = float(str(existing_raw.iloc[0]).split("±")[0].strip())
                    except (ValueError, TypeError):
                        pass1_value = None
                extracted.loc[row_mask, field_name] = value
                flagged_note = None
                if pass1_value is not None and pass1_value != 0:
                    ratio = abs(value / pass1_value)
                    if ratio > 10 or ratio < 0.1:
                        if "Needs Review" not in extracted.columns:
                            extracted["Needs Review"] = ""
                        flagged_note = f"{field_name} (T2-gate value {value:.4g} differs >10x from Pass-1 {pass1_value:.4g})"
                        existing_flag = str(extracted.loc[row_mask, "Needs Review"].iloc[0] or "")
                        extracted.loc[row_mask, "Needs Review"] = f"{existing_flag}; {flagged_note}" if existing_flag else flagged_note
                # Deterministic (not sampled), so this is always "high"
                # confidence in the reconciliation sense -- the >10x sanity
                # check above is a separate, coarser safety net, not a
                # confidence signal for this record.
                t2_gate_results[sample_id] = ReconciliationResult(
                    value=value, confidence="high", majority_fraction=1.0,
                    outliers=[{"value": None, "evidence": flagged_note}] if flagged_note else [],
                )
            if t2_gate_results:
                # Provenance: this field's evidence text still reflects
                # whatever Pass-1's tier vote produced (e.g. "read from Fig.
                # N"), which is no longer accurate for samples the T2 gate
                # just computed deterministically. Append (not replace) so
                # the original reasoning survives alongside this note, same
                # pattern apply_patch already uses for T3.
                gate_note = (
                    f"[T2-gate] resolved deterministically for {len(t2_gate_results)} sample(s) via "
                    f"formula: {calc_spec.formula} using {calc_spec.variables}"
                )
                mask = evidence["Column Name"] == field_name
                if mask.any():
                    existing_notes = evidence.loc[mask, "Notes"].iloc[0]
                    evidence.loc[mask, "Notes"] = f"{existing_notes}; {gate_note}"
                write_audit_log(paper_dir, field_name, t2_gate_results)
            print(
                f"  [t2-gate] '{field_name}': resolved {len(resolved) - len(still_unresolved)}/"
                f"{len(resolved)} samples deterministically; {len(still_unresolved)} fall through to T3"
            )
            if still_unresolved:
                if has_figure_fallback:
                    unresolved_samples_by_field[field_name] = still_unresolved
                else:
                    _blank_unverified_field(
                        field_name,
                        "T2-gate formula inputs missing for these samples, no figure available",
                        sample_ids=still_unresolved,
                    )
        except (RuntimeError, ValueError, SyntaxError, ZeroDivisionError) as exc:
            if has_figure_fallback:
                print(f"  [t2-gate] failed for '{field_name}' ({exc}), falling through to T3")
                unresolved_samples_by_field[field_name] = extracted["Sample ID"].astype(str).str.strip().tolist()
            else:
                _blank_unverified_field(field_name, f"T2-gate error: {exc}")

    # --- Identify remaining figure-sourced (T3) fields: merge in T2-gate
    # leftovers (still-unresolved dual-path fields), and make sure
    # fully-resolved dual-path fields are never re-processed as T3 even if
    # the tier vote separately flagged them figure-only for some draws.
    for field_name, fig_num in t2_gate_fields.items():
        if field_name in unresolved_samples_by_field:
            figure_fields.setdefault(field_name, fig_num)
        else:
            figure_fields.pop(field_name, None)
    print(f"[figures] Priority-3 figure-sourced fields: {figure_fields or 'none'}")

    from figure_classify import classify_figure_type, cv_available_for
    from reconcile import cluster_and_reconcile, draw_count, majority_vote_categorical
    from repeated_sampling import collect_cv_anchor, collect_llm_draws, collect_match_draws
    from audit_log import write_audit_log

    refine_metadata = {}
    for field_name, fig_num in figure_fields.items():
        page = find_figure_page(pdf_path, fig_num)
        if page is None:
            print(f"  [skip] could not locate page for Fig. {fig_num} (field '{field_name}')")
            continue
        page_image = render_page(pdf_path, page, figures_dir)
        try:
            bbox, bbox_usage = locate_figure_bbox(api_key, model_id, page_image, fig_num)
            call_usages.append(bbox_usage)
            print(f"  [tokens] bbox localization: {_fmt_usage(bbox_usage)}")
            image_path = render_figure_crop(pdf_path, page, bbox, figures_dir)
            print(
                f"  [refine] '{field_name}' <- Fig. {fig_num} (page {page}, "
                f"bbox={tuple(round(v, 3) for v in bbox)}, crop={image_path.name})"
            )
        except (RuntimeError, KeyError, ValueError) as exc:
            if isinstance(exc, LocalizationError):
                call_usages.append(exc.usage)
            print(f"  [skip] localization failed for '{field_name}' <- Fig. {fig_num} (page {page}): {exc}")
            # If this field only reached T3 because the T2 gate had already
            # rejected its calculation as unreliable (has a
            # unresolved_samples_by_field entry), "keep Pass-1's value" would
            # silently re-surface the exact value the T2 gate rejected --
            # there's nowhere left to verify it, so blank instead. An
            # ordinary figure-only field that never touched the T2 gate
            # (no entry here) keeps its previous, safer fallback: its Pass-1
            # value was never specifically flagged as unreliable, just
            # unverified this round.
            if field_name in unresolved_samples_by_field:
                _blank_unverified_field(
                    field_name,
                    f"T2-gate rejected this calculation and the figure fallback also failed to localize: {exc}",
                    sample_ids=unresolved_samples_by_field[field_name],
                )
            else:
                print(f"         keeping Pass 1 value for '{field_name}' unchanged")
            continue

        try:
            figure_type, classify_usage = classify_figure_type(api_key, model_id, image_path)
            call_usages.append(classify_usage)
            has_cv = cv_available_for(figure_type)
            if has_cv:
                print(
                    f"  [classify] '{field_name}' figure type = {figure_type} "
                    f"(CV anchor available, but axis-range detection is not yet implemented -- falling back to LLM-only)"
                )
                has_cv = False
            else:
                print(f"  [classify] '{field_name}' figure type = {figure_type} (CV anchor: {has_cv})")
            n_draws = draw_count(
                has_cv_anchor=has_cv,
                cv_match_draws=cv_match_draws,
                llm_draws=llm_draws,
            )
            print(f"  [sampling] '{field_name}' repeated draws = {n_draws}")

            # Always show the model the FULL sample list as shot-matching
            # context, even when only some samples still need this field --
            # otherwise the model isn't told the other samples/shots exist
            # at all, which can push it toward mis-assigning a plotted point
            # to the wrong (listed) shot. Only ever WRITE BACK values for
            # the samples that actually still need it, though (filtered
            # below, before apply_patch): the T2 gate's already-resolved
            # values for the rest must not be silently overwritten by a
            # less-reliable visual re-read, even if the model reports one.
            unresolved_ids = unresolved_samples_by_field.get(field_name)
            target_extracted = extracted

            sample_results = {}
            if has_cv:
                filled_markers = collect_cv_anchor(image_path, y_min=0.0, y_max=1.0)
                match_draws, match_usage = collect_match_draws(
                    api_key, model_id, image_path, filled_markers, field_name, fig_num, target_extracted, n_draws
                )
                call_usages.append(match_usage)
                marker_value_by_number = {str(m["marker_number"]): m["value"] for m in filled_markers}
                for sample_id, labels in match_draws.items():
                    winner, confidence, fraction = majority_vote_categorical(labels)
                    value = marker_value_by_number.get(winner) if winner else None
                    sample_results[sample_id] = ReconciliationResult(
                        value=value,
                        confidence=confidence if value is not None else "low",
                        majority_fraction=fraction,
                        outliers=[{"value": None, "evidence": f"matched marker {label}"} for label in labels if label != winner],
                    )
            else:
                per_sample_draws, refine_draws_usage = collect_llm_draws(
                    api_key, model_id, base_prompt, field_name, fig_num, image_path, target_extracted, n_draws
                )
                call_usages.append(refine_draws_usage)
                for sample_id, draws in per_sample_draws.items():
                    sample_results[sample_id] = cluster_and_reconcile(draws, n_attempted=n_draws)

            if unresolved_ids is not None:
                unresolved_id_set = set(unresolved_ids)
                sample_results = {
                    sample_id: result
                    for sample_id, result in sample_results.items()
                    if sample_id in unresolved_id_set
                }

            apply_patch(extracted, evidence, field_name, fig_num, sample_results)
            write_audit_log(paper_dir, field_name, sample_results)
            refine_metadata[field_name] = {
                "figure_number": fig_num,
                "page": page,
                "figure_type": figure_type,
                "cv_anchor": has_cv,
                "n_draws": n_draws,
            }
        except (RuntimeError, KeyError, ValueError, ParseError) as exc:
            print(f"  [skip] figure processing failed for '{field_name}' <- Fig. {fig_num}: {exc}")
            # Same rationale as the localization-failure branch above.
            if field_name in unresolved_samples_by_field:
                _blank_unverified_field(
                    field_name,
                    f"T2-gate rejected this calculation and the figure fallback also failed: {exc}",
                    sample_ids=unresolved_samples_by_field[field_name],
                )
            else:
                print(f"         keeping Pass 1 value for '{field_name}' unchanged")
            continue

    # --- Final output: Pass 1 table with only flagged T3 fields patched ---
    # "Needs Review" is an internal flag column (set by reconcile_pass1_table,
    # apply_patch, and the T2-gate sanity check) -- it must not reach
    # extracted_data.csv. heldout_pipeline's evaluator melts every
    # non-identifier column into a scored "field", so this column would be
    # scored against ground truth as a phantom field (empty rows counted as
    # correct_null, flagged rows counted as false_extraction), silently
    # perturbing accuracy comparisons. Split it into its own sidecar file
    # instead, and only for samples that actually have a flag.
    if "Needs Review" in extracted.columns:
        needs_review = extracted.loc[
            extracted["Needs Review"].astype(str).str.strip() != "", ["Sample ID", "Needs Review"]
        ]
        needs_review.to_csv(paper_dir / "needs_review.csv", index=False)
        extracted = extracted.drop(columns=["Needs Review"])
    extracted.to_csv(paper_dir / "extracted_data.csv", index=False)
    evidence.to_csv(paper_dir / "evidence_source.csv", index=False)
    (paper_dir / "parse_report.json").write_text(
        json.dumps(
            {
                "valid": True,
                "extracted_rows": len(extracted),
                "evidence_rows": len(evidence),
                "unexpected_columns": [],
                "fields_refined": list(refine_metadata.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paper_total_usage = _sum_usage(*call_usages)
    (paper_dir / "response_metadata.json").write_text(
        json.dumps(
            {
                "pass1_usage": usage1,
                "pass1_selected_draw": selected_pass1["draw_idx"],
                "pass1_draws_requested": pass1_draws,
                "pass1_max_draws": effective_max_draws,
                "pass1_draws_taken": len(pass1_attempts),
                "pass1_draws_parsed": len(pass1_attempts),
                "pass1_draw_usages": [
                    {"draw_idx": attempt["draw_idx"], "usage": attempt["usage"]}
                    for attempt in pass1_attempts
                ],
                "refine_usage": refine_metadata,
                "paper_total_usage": paper_total_usage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  [tokens] paper total ({len(call_usages)} calls): {_fmt_usage(paper_total_usage)}")
    print(f"[done] final output written to {paper_dir} (T1/T2 = pass 1 verbatim, T3 = {list(refine_metadata)})")


def load_evidence_rows_from_frame(evidence: pd.DataFrame) -> list[dict]:
    return evidence.to_dict("records")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument(
        "--pass1-draws",
        type=int,
        default=1,
        help="Repeated full-PDF Pass-1 extraction attempts (default: 1).",
    )
    parser.add_argument(
        "--pass1-max-draws",
        type=int,
        default=None,
        help=(
            "Ceiling for adaptive extra Pass-1 draws when a field's source-tier "
            "vote is a genuine tie that could change T1/T2-gate/T3 routing "
            "(default: same as --pass1-draws, i.e. no extra draws taken)."
        ),
    )
    parser.add_argument(
        "--llm-draws",
        type=int,
        default=5,
        help="Repeated figure-read attempts when no CV anchor is used (default: 5).",
    )
    parser.add_argument(
        "--cv-match-draws",
        type=int,
        default=3,
        help="Repeated shot-matching attempts when a CV anchor is used (default: 3).",
    )
    args = parser.parse_args()

    if args.pass1_draws < 1 or args.llm_draws < 1 or args.cv_match_draws < 1:
        raise SystemExit("--pass1-draws, --llm-draws, and --cv-match-draws must all be >= 1")
    if args.pass1_max_draws is not None and args.pass1_max_draws < args.pass1_draws:
        raise SystemExit("--pass1-max-draws must be >= --pass1-draws")

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key and args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set and --api-key-file not provided")

    run_pipeline(
        pdf_path=args.pdf,
        prompt_path=args.prompt,
        paper_id=args.paper_id,
        split=args.split,
        output_dir=args.output_dir,
        model_id=args.model,
        api_key=api_key,
        pass1_draws=args.pass1_draws,
        pass1_max_draws=args.pass1_max_draws,
        llm_draws=args.llm_draws,
        cv_match_draws=args.cv_match_draws,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
