"""Classify a figure crop's chart type and decide whether the (v1) CV
detector applies. Only discrete-marker scatter plots have a real CV module
in v1 (validated on Paper1 Fig. 4) -- every other type is explicitly
routed to the LLM-only path rather than assumed to work, per
docs/superpowers/specs/2026-07-28-t3-figure-pipeline-redesign-design.md.
"""
from __future__ import annotations

from pathlib import Path

FIGURE_TYPES = [
    "discrete-marker",
    "line-with-markers",
    "continuous-curve",
    "bar",
    "box-plot",
    "non-chart",
]

CLASSIFY_PROMPT_TEMPLATE = """This image shows a figure (or the relevant portion of one) from a
scientific paper. Classify its chart type as exactly one of:

- discrete-marker: scatter plot with distinct point markers (circles, squares, triangles, etc.), no continuous line
- line-with-markers: a line connecting distinct marker points at each data value
- continuous-curve: a smooth line/curve with no distinct markers at individual data points
- bar: a bar chart
- box-plot: a box-and-whisker or error-bar-style plot
- non-chart: not an axis-based data chart (e.g. micrograph, schematic, diagram, photo)

Respond with ONLY the single matching label from the list above, nothing else.
"""


def classify_figure_type(
    api_key: str,
    model_id: str,
    figure_image: Path,
    call_llm=None,
    call_upload=None,
) -> tuple[str, dict]:
    if call_llm is None:
        from run_figure_refine import run_gemini as call_llm  # noqa: N813
    if call_upload is None:
        from run_figure_refine import upload_file as call_upload  # noqa: N813
    client, uploaded = call_upload(api_key, figure_image)
    text, usage = call_llm(api_key, model_id, [uploaded, CLASSIFY_PROMPT_TEMPLATE])
    client.files.delete(name=uploaded.name)
    label = text.strip().lower()
    for candidate in FIGURE_TYPES:
        if candidate in label:
            return candidate, usage
    return "non-chart", usage


def cv_available_for(figure_type: str) -> bool:
    return figure_type == "discrete-marker"
