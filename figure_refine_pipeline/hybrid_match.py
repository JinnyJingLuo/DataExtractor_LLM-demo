#!/usr/bin/env python3
"""
Hybrid CV+LLM prototype: CV detects markers and computes their exact values
deterministically; the LLM's only job is to match each numbered marker to the
correct experimental shot (a discrete matching task, not numeric estimation).

Draws numbered labels on the detected markers, sends the annotated image plus
the shot list to the model, and asks only "which marker number goes with
which shot" -- the reported value always comes from CV, never from the model.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
from digitize_figure import load_gray, find_axis_box, classify_markers, pixel_to_value
from run_figure_refine import run_gemini, upload_file, build_shot_table

MATCH_PROMPT_TEMPLATE = """This image shows Figure {figure_number} from a shock-physics paper, with numbered
labels placed at each detected data-point marker (both filled and open circles).

Filled circles = {field_name} (left axis). Open circles = a different series (right axis) -- ignore these.

Below is the list of experimental shots identified in this paper:

{shot_table}

Your ONLY task: match each shot to the marker number of its FILLED circle in the image,
using the plot's axis, legend/arrows, and the shots' known conditions (temperature,
thickness) to decide which point belongs to which shot.

CRITICAL RULE -- READ CAREFULLY: multiple shots in the list below can share the SAME
plotted point. This is common and expected: several shots at the identical temperature
and thickness are indistinguishable on this plot and were plotted as ONE marker. Before
answering, first GROUP the shots by their (Initial Temperature (K), Sample Thickness (mm))
values. Every shot within the same group MUST be assigned the SAME marker number -- do
not leave any group member as "none" just because only one representative point exists
for that group. Check your final answer: if two shots have identical temperature and
thickness, they must have identical marker numbers in your output.

Do NOT report a numeric value -- only report which marker number matches which shot.
If a shot is not plotted in this figure (e.g. wrong thickness), write "none".

Output ONLY a Markdown table with exactly these columns, no other text:

| Sample ID | Marker Number |
| --- | --- |
"""


def annotate_markers(gray: np.ndarray, markers: list[dict], output_path: Path) -> None:
    rgb = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for i, m in enumerate(markers, start=1):
        x, y = m["cx"], m["cy"]
        r = m["diameter_px"] / 2 + 4
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=2)
        label = str(i)
        draw.text((x + r + 2, y - r - 2), label, fill=(255, 0, 0), font=font)
    rgb.save(output_path)


def parse_match_table(text: str) -> dict[str, str]:
    from heldout_pipeline.response_parser import _table_blocks, _parse_block

    blocks = _table_blocks(text)
    if not blocks:
        raise RuntimeError(f"no table found in match response: {text!r}")
    frame = _parse_block(blocks[0])
    return dict(zip(frame["Sample ID"].astype(str).str.strip(), frame["Marker Number"].astype(str).str.strip()))


def main() -> None:
    import os

    api_key = os.environ["GEMINI_API_KEY"]
    model_id = "gemini-3.1-pro-preview"
    image_path = Path("artifacts/native_extract/img-000.png")
    out_dir = Path("artifacts/hybrid_match")
    out_dir.mkdir(parents=True, exist_ok=True)

    gray = load_gray(image_path)
    box = find_axis_box(gray)
    all_markers = classify_markers(gray, box)
    filled = sorted([m for m in all_markers if m["kind"] == "filled"], key=lambda m: m["cx"])
    open_m = sorted([m for m in all_markers if m["kind"] == "open"], key=lambda m: m["cx"])
    for m in filled:
        m["value"] = round(pixel_to_value(m["cy"], box, 0.0, 1.5), 3)

    print(f"CV detected {len(filled)} filled markers:")
    for i, m in enumerate(filled, start=1):
        print(f"  #{i}: value={m['value']} GPa (px cx={m['cx']:.0f})")

    # annotate filled+open together so numbering matches what the model sees;
    # filled get numbers 1..N, open get numbers continuing after
    annotated_path = out_dir / "annotated.png"
    annotate_markers(gray, filled + open_m, annotated_path)
    marker_number_of_filled = {i: filled[i - 1] for i in range(1, len(filled) + 1)}

    extracted = pd.read_csv("artifacts/paper1_figrefine/development/Paper1/extracted_data.csv")
    shot_table = build_shot_table(extracted)

    gt = {
        "AGA": 1.21, "AGA1": 1.21, "AGA2": 1.21, "AGA3": 1.21,
        "AGB": 1.15, "AGBI": 1.15, "AGC": 0.61, "AGD": 1.28, "AGE": 1.28, "AGE1": 1.28,
    }

    def score(gtv, pred):
        err = abs(pred - gtv) / gtv
        if err <= 0.005: return 1.0
        if err <= 0.02: return 0.95
        if err <= 0.05: return 0.80
        if err <= 0.10: return 0.60
        if err <= 0.20: return 0.30
        return 0.0

    prompt = MATCH_PROMPT_TEMPLATE.format(figure_number="4", field_name="Spall Strength (GPa)", shot_table=shot_table)

    all_runs = []
    for attempt in range(1, 6):
        client, uploaded = upload_file(api_key, annotated_path)
        text, _usage = run_gemini(api_key, model_id, [uploaded, prompt])
        client.files.delete(name=uploaded.name)
        (out_dir / f"match_response_{attempt}.md").write_text(text, encoding="utf-8")
        try:
            mapping = parse_match_table(text)
        except RuntimeError as exc:
            print(f"[attempt {attempt}] FAILED to parse: {exc}")
            continue

        run_scores = []
        run_detail = {}
        for sample, gtv in gt.items():
            marker_no = mapping.get(sample, "none")
            m = re.search(r"\d+", marker_no)
            if not m or int(m.group(0)) not in marker_number_of_filled:
                run_detail[sample] = ("no-match", None, 0.0)
                run_scores.append(0.0)
                continue
            marker = marker_number_of_filled[int(m.group(0))]
            pred = marker["value"]
            sc = score(gtv, pred)
            run_detail[sample] = (marker_no, pred, sc)
            run_scores.append(sc)

        graded = 100 * sum(run_scores) / len(run_scores)
        strict = 100 * sum(1 for s in run_scores if s == 1.0) / len(run_scores)
        print(f"\n[attempt {attempt}] graded={graded:.1f}% strict={strict:.1f}%")
        for sample, (marker_no, pred, sc) in run_detail.items():
            print(f"    {sample:6s} gt={gt[sample]:.2f} -> marker {marker_no} -> pred={pred} score={sc}")
        all_runs.append({"attempt": attempt, "graded": graded, "strict": strict, "detail": run_detail})

    print("\n=== summary across", len(all_runs), "attempts ===")
    for r in all_runs:
        print(f"  attempt {r['attempt']}: graded={r['graded']:.1f}% strict={r['strict']:.1f}%")
    if all_runs:
        mean_graded = sum(r["graded"] for r in all_runs) / len(all_runs)
        mean_strict = sum(r["strict"] for r in all_runs) / len(all_runs)
        print(f"  MEAN: graded={mean_graded:.1f}% strict={mean_strict:.1f}%")

    (out_dir / "summary.json").write_text(
        json.dumps(
            [{"attempt": r["attempt"], "graded": r["graded"], "strict": r["strict"]} for r in all_runs],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
