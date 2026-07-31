# T3 Figure-Extraction Pipeline Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `figure_refine_pipeline/` with a narrow T2 gate (deterministic recompute for fields whose Pass-1 evidence text shows both a calculation and a figure source), figure-type classification with CV routing, and tolerance-cluster reconciliation of repeated samples, per `docs/superpowers/specs/2026-07-28-t3-figure-pipeline-redesign-design.md`.

**Architecture:** Four new focused modules (`t2_gate.py`, `figure_classify.py`, `reconcile.py`, `repeated_sampling.py`) each expose small, pure or dependency-injectable functions that `run_pipeline()` in `run_figure_refine.py` orchestrates. `apply_patch()` is modified to accept a reconciliation result instead of a raw patch and to preserve rather than overwrite original evidence text. Existing validated code (`find_figure_page`, `locate_figure_bbox`, `render_figure_crop`, `refine_field`, `digitize_figure.py`, `hybrid_match.py`) is reused unmodified.

**Tech Stack:** Python 3.10+, pandas, pytest, `google-genai` (Gemini Developer API), existing `heldout_pipeline.response_parser` table parsing.

## Global Constraints

- The frozen prompt (`prompts/prompty_frozen.md`) is never modified by any task in this plan.
- `figure_refine_pipeline/` has no `__init__.py` and is not an installed package; new modules import each other the same way existing ones do (`sys.path.insert(0, ".")` at script entry, or `sys.path.insert(0, str(Path(__file__).resolve().parent))` in tests).
- All new LLM-calling functions accept the underlying call as an injectable parameter (defaulting to the real implementation), matching this repo's existing `FakeClient`-style DI pattern (see `tests/test_api_runner.py`), so they're unit-testable without hitting the network.
- Tests live in the repo-root `tests/` directory (per `pyproject.toml`'s `testpaths = ["tests"]`), not inside `figure_refine_pipeline/`.
- No task in this plan runs the pipeline against a live paper/API — that's the manual pilot step at the end, run by the user per the spec's rollout plan (Paper12, single field, before wider use).

---

### Task 1: Test path setup for `figure_refine_pipeline` imports

**Files:**
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: importing `t2_gate`, `figure_classify`, `reconcile`, `repeated_sampling`, `run_figure_refine`, `digitize_figure`, `hybrid_match` directly by module name works from any test file under `tests/`.

- [ ] **Step 1: Write `conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figure_refine_pipeline"))
```

- [ ] **Step 2: Verify it works**

Run: `python -m pytest --collect-only tests/ -q`
Expected: collection succeeds with no import errors (no test files reference the new modules yet, so this just confirms conftest.py itself doesn't error).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add sys.path setup for figure_refine_pipeline imports"
```

---

### Task 2: Dual-path field detection

**Files:**
- Create: `figure_refine_pipeline/t2_gate.py`
- Test: `tests/test_t2_gate.py`

**Interfaces:**
- Consumes: `FIG_RE` from `run_figure_refine` (existing, compiled regex `r"Fig\.?\s*(\d+)"`).
- Produces: `detect_dual_path_fields(evidence_rows: list[dict]) -> dict[str, str]` mapping field name to figure number, for fields whose combined `Notes` + `Source Location` text matches both a calculation marker and a figure reference.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_t2_gate.py
from t2_gate import detect_dual_path_fields


def test_detects_paper12_longitudinal_stress_dual_path():
    evidence_rows = [
        {
            "Column Name": "Longitudinal Stress at HEL (GPa)",
            "Source Location": "Page 6, Fig. 5(a) & CALCULATED",
            "Notes": (
                "[Priority 3/2]: For VA0, [Priority 3] visually extracted "
                "from Fig. 5(a) (~1.0 GPa); ⚠ visual extraction; type: "
                "extracted_from_figure. For VA0.6/VA5.5, [Priority 2] "
                "calculated from text u_HEL values; type: calculated_from_u_HEL"
            ),
        }
    ]
    assert detect_dual_path_fields(evidence_rows) == {
        "Longitudinal Stress at HEL (GPa)": "5"
    }


def test_ignores_pure_calculation_field_with_no_figure():
    evidence_rows = [
        {
            "Column Name": "Young's Modulus (GPa)",
            "Source Location": "CALCULATED",
            "Notes": "[Priority 2]: E = 9BG / (3B+G)",
        }
    ]
    assert detect_dual_path_fields(evidence_rows) == {}


def test_ignores_pure_figure_field_with_no_calculation():
    evidence_rows = [
        {
            "Column Name": "Strain Rate (s⁻¹)",
            "Source Location": "Page 5, Fig. 4(c)",
            "Notes": "[Priority 3]: Extracted from compressive strain rate vs stress plot",
        }
    ]
    assert detect_dual_path_fields(evidence_rows) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_t2_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 't2_gate'`

- [ ] **Step 3: Write minimal implementation**

```python
# figure_refine_pipeline/t2_gate.py
"""Narrow T2 gate: detect fields with both a calculation and figure source
in their Pass-1 evidence text, and resolve them deterministically in code
rather than trusting either the LLM's own arithmetic or a figure re-read.

Scope is intentionally narrow (only dual-path fields) -- see
docs/superpowers/specs/2026-07-28-t3-figure-pipeline-redesign-design.md.
A "broad" gate covering every Priority-2 field is an explicit possible
future change, not built here.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass

import pandas as pd

from run_figure_refine import FIG_RE

CALC_RE = re.compile(r"CALCULATED|\bP(?:RIORITY)?\s*2\b", re.IGNORECASE)


def detect_dual_path_fields(evidence_rows: list[dict]) -> dict[str, str]:
    """Return {field_name: figure_number} for fields whose evidence text
    shows BOTH a calculation source and a figure reference."""
    fields: dict[str, str] = {}
    for row in evidence_rows:
        field_name = str(row.get("Column Name", "")).strip()
        notes = str(row.get("Notes", ""))
        location = str(row.get("Source Location", ""))
        text = f"{notes} {location}"
        if not field_name:
            continue
        if not (CALC_RE.search(text) and FIG_RE.search(text)):
            continue
        match = FIG_RE.search(text)
        fields[field_name] = match.group(1)
    return fields
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_t2_gate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add figure_refine_pipeline/t2_gate.py tests/test_t2_gate.py
git commit -m "feat: detect dual-path (calc + figure) fields for T2 gate"
```

---

### Task 3: Safe deterministic formula evaluation

**Files:**
- Modify: `figure_refine_pipeline/t2_gate.py`
- Test: `tests/test_t2_gate.py`

**Interfaces:**
- Produces: `safe_eval_formula(formula: str, values: dict[str, float]) -> float`. Supports `+ - * / **` unary minus, numeric literals, and named variables looked up in `values`. Raises `ValueError` on anything else (function calls, attribute access, etc.) -- this evaluates LLM-supplied formula strings, so it must never execute arbitrary code.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_t2_gate.py (append)
import pytest

from t2_gate import safe_eval_formula


def test_evaluates_paper12_sigma_hel_formula():
    value = safe_eval_formula(
        "0.5*rho0*c_l*u_HEL/1e9",
        {"rho0": 6110, "c_l": 6090, "u_HEL": 180},
    )
    assert value == pytest.approx(3.348891, rel=1e-4)


def test_rejects_function_calls():
    with pytest.raises(ValueError, match="unsupported"):
        safe_eval_formula("__import__('os').system('ls')", {})


def test_rejects_unknown_variable():
    with pytest.raises(ValueError, match="unknown variable"):
        safe_eval_formula("a + b", {"a": 1.0})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_t2_gate.py -v -k eval_formula or rejects`
Expected: FAIL with `ImportError: cannot import name 'safe_eval_formula'`

- [ ] **Step 3: Write minimal implementation**

```python
# figure_refine_pipeline/t2_gate.py (append)
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_formula(formula: str, values: dict[str, float]) -> float:
    """Evaluate a restricted arithmetic expression (+-*/** and named
    variables only) against a values dict. Rejects anything else -- this
    evaluates formula strings returned by an LLM call, never trust it to
    only contain arithmetic."""
    tree = ast.parse(formula, mode="eval")
    return _eval_node(tree.body, values)


def _eval_node(node: ast.AST, values: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](
            _eval_node(node.left, values), _eval_node(node.right, values)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand, values))
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ValueError(f"unknown variable in formula: {node.id}")
        return values[node.id]
    raise ValueError(f"unsupported expression in formula: {ast.dump(node)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_t2_gate.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add figure_refine_pipeline/t2_gate.py tests/test_t2_gate.py
git commit -m "feat: add safe deterministic formula evaluator for T2 gate"
```

---

### Task 4: LLM formula/input identification call

**Files:**
- Modify: `figure_refine_pipeline/t2_gate.py`
- Test: `tests/test_t2_gate.py`

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `CalculationSpec` dataclass (`formula: str`, `variables: dict[str, str]`); `identify_calculation(field_name: str, evidence_note: str, api_key: str, model_id: str, call_llm=None) -> CalculationSpec | None`. `call_llm` defaults to `run_figure_refine.run_gemini` and has signature `(api_key, model_id, contents: list) -> tuple[str, dict]` matching `run_gemini`'s existing signature (called here with `contents=[prompt]`, a text-only call, no image). Returns `None` when the model reports no closed-form arithmetic formula exists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_t2_gate.py (append)
import json

from t2_gate import CalculationSpec, identify_calculation


def test_identify_calculation_parses_formula_json():
    def fake_call_llm(api_key, model_id, contents):
        return (
            json.dumps(
                {
                    "formula": "0.5*rho0*c_l*u_HEL/1e9",
                    "variables": {
                        "rho0": "Initial Density (g/cm3)",
                        "c_l": "Longitudinal Sound Speed (m/s)",
                        "u_HEL": "Free Surface Velocity at HEL (m/s)",
                    },
                }
            ),
            {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0, "total_tokens": 15},
        )

    spec = identify_calculation(
        field_name="Longitudinal Stress at HEL (GPa)",
        evidence_note="CALCULATED (P2): sigma_HEL = 0.5 * rho0 * c_l * u_HEL",
        api_key="fake-key",
        model_id="fake-model",
        call_llm=fake_call_llm,
    )
    assert spec == CalculationSpec(
        formula="0.5*rho0*c_l*u_HEL/1e9",
        variables={
            "rho0": "Initial Density (g/cm3)",
            "c_l": "Longitudinal Sound Speed (m/s)",
            "u_HEL": "Free Surface Velocity at HEL (m/s)",
        },
    )


def test_identify_calculation_returns_none_when_no_formula():
    def fake_call_llm(api_key, model_id, contents):
        return json.dumps({"formula": None}), {}

    spec = identify_calculation(
        field_name="Some Field",
        evidence_note="no closed-form relation available",
        api_key="fake-key",
        model_id="fake-model",
        call_llm=fake_call_llm,
    )
    assert spec is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_t2_gate.py -v -k identify_calculation`
Expected: FAIL with `ImportError: cannot import name 'CalculationSpec'`

- [ ] **Step 3: Write minimal implementation**

```python
# figure_refine_pipeline/t2_gate.py (append)
IDENTIFY_CALC_PROMPT_TEMPLATE = """You previously extracted data from a scientific paper. For the column
"{field_name}", your evidence notes indicate it can sometimes be calculated
from other extracted values rather than read from a figure:

{evidence_note}

Identify the exact arithmetic formula used, expressed only with +, -, *, /,
**, and parentheses (no functions, no square roots -- if the formula
genuinely needs those, respond with {{"formula": null}}). Use short
variable names for each input, and map each variable name to the exact
column name it corresponds to in the paper's extracted data table.

Return ONLY a JSON object, no other text:
{{"formula": "0.5*rho0*c_l*u_HEL/1e9", "variables": {{"rho0": "Initial Density (g/cm3)", "c_l": "Longitudinal Sound Speed (m/s)", "u_HEL": "Free Surface Velocity at HEL (m/s)"}}}}
"""


@dataclass
class CalculationSpec:
    formula: str
    variables: dict[str, str]


def identify_calculation(
    field_name: str,
    evidence_note: str,
    api_key: str,
    model_id: str,
    call_llm=None,
) -> CalculationSpec | None:
    if call_llm is None:
        from run_figure_refine import run_gemini as call_llm  # noqa: N813
    prompt = IDENTIFY_CALC_PROMPT_TEMPLATE.format(
        field_name=field_name, evidence_note=evidence_note
    )
    text, _usage = call_llm(api_key, model_id, [prompt])
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"could not parse calculation-identification JSON: {text!r}")
    payload = json.loads(match.group(0))
    if not payload.get("formula"):
        return None
    return CalculationSpec(formula=payload["formula"], variables=payload["variables"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_t2_gate.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add figure_refine_pipeline/t2_gate.py tests/test_t2_gate.py
git commit -m "feat: add LLM formula/input identification for T2 gate"
```

---

### Task 5: Resolve dual-path field via T2 gate

**Files:**
- Modify: `figure_refine_pipeline/t2_gate.py`
- Test: `tests/test_t2_gate.py`

**Interfaces:**
- Consumes: `CalculationSpec`, `safe_eval_formula` (this file).
- Produces: `resolve_via_t2_gate(calc_spec: CalculationSpec, pass1_extracted: pd.DataFrame) -> dict[str, float | None]` mapping `Sample ID` to a deterministically computed value, or `None` if any required input column is missing or non-numeric for that sample (meaning it must fall through to the T3 figure path).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_t2_gate.py (append)
import pandas as pd

from t2_gate import resolve_via_t2_gate


def test_resolve_via_t2_gate_computes_and_flags_missing():
    calc_spec = CalculationSpec(
        formula="0.5*rho0*c_l*u_HEL/1e9",
        variables={
            "rho0": "Initial Density (g/cm3)",
            "c_l": "Longitudinal Sound Speed (m/s)",
            "u_HEL": "Free Surface Velocity at HEL (m/s)",
        },
    )
    extracted = pd.DataFrame(
        [
            {
                "Sample ID": "VA0.6-300",
                "Initial Density (g/cm3)": 6110,
                "Longitudinal Sound Speed (m/s)": 6090,
                "Free Surface Velocity at HEL (m/s)": 180,
            },
            {
                "Sample ID": "VA0-300",
                "Initial Density (g/cm3)": 6110,
                "Longitudinal Sound Speed (m/s)": 6090,
                "Free Surface Velocity at HEL (m/s)": "-",
            },
        ]
    )
    result = resolve_via_t2_gate(calc_spec, extracted)
    assert result["VA0.6-300"] == pytest.approx(3.348891, rel=1e-4)
    assert result["VA0-300"] is None
```

(`import pytest` already present at top of the test file from Task 3.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_t2_gate.py -v -k resolve_via_t2_gate`
Expected: FAIL with `ImportError: cannot import name 'resolve_via_t2_gate'`

- [ ] **Step 3: Write minimal implementation**

```python
# figure_refine_pipeline/t2_gate.py (append)
def resolve_via_t2_gate(
    calc_spec: CalculationSpec, pass1_extracted: pd.DataFrame
) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for _, row in pass1_extracted.iterrows():
        sample_id = str(row["Sample ID"]).strip()
        values: dict[str, float] = {}
        missing = False
        for var, column in calc_spec.variables.items():
            if column not in pass1_extracted.columns:
                missing = True
                break
            raw = row[column]
            try:
                values[var] = float(str(raw).split("±")[0].strip())
            except (ValueError, TypeError):
                missing = True
                break
        results[sample_id] = None if missing else safe_eval_formula(calc_spec.formula, values)
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_t2_gate.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add figure_refine_pipeline/t2_gate.py tests/test_t2_gate.py
git commit -m "feat: resolve dual-path fields per-sample via T2 gate"
```

---

### Task 6: Continuous-value reconciliation (tolerance-cluster + median)

**Files:**
- Create: `figure_refine_pipeline/reconcile.py`
- Test: `tests/test_reconcile.py`

**Interfaces:**
- Produces: `ReconciliationResult` dataclass (`value: float | None`, `confidence: str`, `majority_fraction: float`, `outliers: list[dict]`); `cluster_and_reconcile(draws: list[dict], tolerance: float = 0.02, min_majority: float = 0.6) -> ReconciliationResult`, where each draw is `{"value": float, "evidence": str}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reconcile.py
import pytest

from reconcile import cluster_and_reconcile


def _draws(values_and_evidence):
    return [{"value": v, "evidence": e} for v, e in values_and_evidence]


def test_clean_majority_with_one_outlier():
    draws = _draws(
        [
            (3.501, "CALCULATED via sigma_HEL formula"),
            (3.502, "CALCULATED via sigma_HEL formula"),
            (3.503, "CALCULATED via sigma_HEL formula"),
            (3.492, "CALCULATED via sigma_HEL formula"),
            (1.02, "visually extracted from Fig. 5(a), could not find u_HEL in text"),
        ]
    )
    result = cluster_and_reconcile(draws)
    assert result.value == pytest.approx(3.5015, rel=1e-3)
    assert result.confidence == "high"
    assert result.majority_fraction == pytest.approx(0.8)
    assert len(result.outliers) == 1
    assert result.outliers[0]["value"] == 1.02
    assert "could not find u_HEL" in result.outliers[0]["evidence"]


def test_near_even_split_is_low_confidence():
    draws = _draws([(3.50, "calc"), (3.50, "calc"), (1.02, "fig"), (1.03, "fig"), (1.01, "fig")])
    result = cluster_and_reconcile(draws)
    assert result.confidence == "low"
    assert result.majority_fraction == pytest.approx(0.6)


def test_empty_draws_returns_low_confidence_none():
    result = cluster_and_reconcile([])
    assert result.value is None
    assert result.confidence == "low"
    assert result.outliers == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reconcile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reconcile'`

- [ ] **Step 3: Write minimal implementation**

```python
# figure_refine_pipeline/reconcile.py
"""Reconcile repeated-sample draws into one value + confidence.

Uses tolerance-based clustering + median-of-majority-cluster rather than
majority-vote-on-exact-value: continuous physical quantities almost never
repeat exactly across draws (observed spread like 1.040/1.042/1.043 for
the same true value), so exact-match voting would essentially never find
a majority. See docs/superpowers/specs/2026-07-28-t3-figure-pipeline-redesign-design.md
for the reasoning and the worked example this module's tests are drawn
from.
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass


@dataclass
class ReconciliationResult:
    value: float | None
    confidence: str  # "high" or "low"
    majority_fraction: float
    outliers: list[dict]


def cluster_and_reconcile(
    draws: list[dict], tolerance: float = 0.02, min_majority: float = 0.6
) -> ReconciliationResult:
    if not draws:
        return ReconciliationResult(value=None, confidence="low", majority_fraction=0.0, outliers=[])

    clusters: list[list[dict]] = []
    for draw in sorted(draws, key=lambda d: d["value"]):
        placed = False
        for cluster in clusters:
            center = statistics.median(m["value"] for m in cluster)
            rel_diff = abs(draw["value"] - center) / abs(center) if center else abs(draw["value"] - center)
            if rel_diff <= tolerance:
                cluster.append(draw)
                placed = True
                break
        if not placed:
            clusters.append([draw])

    best = max(clusters, key=len)
    fraction = len(best) / len(draws)
    value = statistics.median(m["value"] for m in best)
    confidence = "high" if fraction >= min_majority else "low"
    outliers = [d for d in draws if d not in best]
    return ReconciliationResult(
        value=value, confidence=confidence, majority_fraction=fraction, outliers=outliers
    )


def majority_vote_categorical(
    labels: list[str], min_majority: float = 0.6
) -> tuple[str | None, str, float]:
    """For discrete/categorical repeated samples (e.g. which CV marker
    number a shot was matched to), where tolerance-clustering doesn't
    apply -- exact-match majority vote is correct here since labels are
    discrete, not continuous. Always returns the observed majority
    fraction (even at low confidence) so callers can log it for audit,
    the same way cluster_and_reconcile always reports majority_fraction."""
    if not labels:
        return None, "low", 0.0
    label, count = Counter(labels).most_common(1)[0]
    fraction = count / len(labels)
    if fraction >= min_majority:
        return label, "high", fraction
    return None, "low", fraction


def draw_count(has_cv_anchor: bool) -> int:
    """3 draws when a CV anchor exists (LLM only does shot-matching against
    a fixed measurement), 5 when it doesn't (LLM must measure and match)."""
    return 3 if has_cv_anchor else 5
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reconcile.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add figure_refine_pipeline/reconcile.py tests/test_reconcile.py
git commit -m "feat: add tolerance-cluster reconciliation for repeated samples"
```

---

### Task 7: Categorical majority vote + draw count tests

**Files:**
- Modify: `tests/test_reconcile.py`

**Interfaces:**
- Consumes: `majority_vote_categorical`, `draw_count` (already implemented in Task 6 — this task tests them, since Task 6's step 3 wrote both functions together for cohesion but only wrote tests for `cluster_and_reconcile`).

- [ ] **Step 1: Write the tests**

```python
# tests/test_reconcile.py (append)
from reconcile import draw_count, majority_vote_categorical


def test_majority_vote_categorical_clear_majority():
    label, confidence, fraction = majority_vote_categorical(["3", "3", "5"])
    assert label == "3"
    assert confidence == "high"
    assert fraction == pytest.approx(2 / 3)


def test_majority_vote_categorical_no_majority():
    label, confidence, fraction = majority_vote_categorical(["3", "5", "7"])
    assert label is None
    assert confidence == "low"
    assert fraction == pytest.approx(1 / 3)


def test_draw_count_uses_cv_anchor():
    assert draw_count(has_cv_anchor=True) == 3
    assert draw_count(has_cv_anchor=False) == 5
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_reconcile.py -v`
Expected: PASS (6 tests total) — no implementation changes needed, `reconcile.py` already has both functions from Task 6.

- [ ] **Step 3: Commit**

```bash
git add tests/test_reconcile.py
git commit -m "test: cover majority_vote_categorical and draw_count"
```

---

### Task 8: Figure-type classification + CV routing

**Files:**
- Create: `figure_refine_pipeline/figure_classify.py`
- Test: `tests/test_figure_classify.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `FIGURE_TYPES` list; `classify_figure_type(api_key, model_id, figure_image, call_llm=None, call_upload=None) -> str` (one of `FIGURE_TYPES`, defaults to `"non-chart"` if the response doesn't match); `cv_available_for(figure_type: str) -> bool` (only `True` for `"discrete-marker"` in v1 — every other type routes to LLM-only per the spec's explicit v1 scope).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figure_classify.py
from pathlib import Path

from figure_classify import classify_figure_type, cv_available_for


class _FakeUploadedFile:
    name = "files/fake"


def _fake_call_upload(api_key, path):
    class _FakeClient:
        class files:
            @staticmethod
            def delete(name):
                pass

    return _FakeClient(), _FakeUploadedFile()


def test_classify_figure_type_exact_label():
    def fake_call_llm(api_key, model_id, contents):
        return "discrete-marker", {}

    result = classify_figure_type(
        "fake-key", "fake-model", Path("/tmp/fake.png"),
        call_llm=fake_call_llm, call_upload=_fake_call_upload,
    )
    assert result == "discrete-marker"


def test_classify_figure_type_extracts_label_from_extra_text():
    def fake_call_llm(api_key, model_id, contents):
        return "I think this is a Bar Chart.", {}

    result = classify_figure_type(
        "fake-key", "fake-model", Path("/tmp/fake.png"),
        call_llm=fake_call_llm, call_upload=_fake_call_upload,
    )
    assert result == "bar"


def test_classify_figure_type_unparseable_defaults_to_non_chart():
    def fake_call_llm(api_key, model_id, contents):
        return "unrelated gibberish response", {}

    result = classify_figure_type(
        "fake-key", "fake-model", Path("/tmp/fake.png"),
        call_llm=fake_call_llm, call_upload=_fake_call_upload,
    )
    assert result == "non-chart"


def test_cv_available_only_for_discrete_marker():
    assert cv_available_for("discrete-marker") is True
    assert cv_available_for("line-with-markers") is False
    assert cv_available_for("continuous-curve") is False
    assert cv_available_for("bar") is False
    assert cv_available_for("box-plot") is False
    assert cv_available_for("non-chart") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figure_classify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'figure_classify'`

- [ ] **Step 3: Write minimal implementation**

```python
# figure_refine_pipeline/figure_classify.py
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
) -> str:
    if call_llm is None:
        from run_figure_refine import run_gemini as call_llm  # noqa: N813
    if call_upload is None:
        from run_figure_refine import upload_file as call_upload  # noqa: N813
    client, uploaded = call_upload(api_key, figure_image)
    text, _usage = call_llm(api_key, model_id, [uploaded, CLASSIFY_PROMPT_TEMPLATE])
    client.files.delete(name=uploaded.name)
    label = text.strip().lower()
    for candidate in FIGURE_TYPES:
        if candidate in label:
            return candidate
    return "non-chart"


def cv_available_for(figure_type: str) -> bool:
    return figure_type == "discrete-marker"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figure_classify.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add figure_refine_pipeline/figure_classify.py tests/test_figure_classify.py
git commit -m "feat: add figure-type classification and CV routing decision"
```

---

### Task 9: Repeated LLM-draw collection

**Files:**
- Create: `figure_refine_pipeline/repeated_sampling.py`
- Test: `tests/test_repeated_sampling.py`

**Interfaces:**
- Consumes: nothing new (defaults to `refine_field` from `run_figure_refine`, matching its existing signature `(api_key, model_id, base_prompt, field_name, figure_number, figure_image, pass1_extracted) -> tuple[pd.DataFrame, dict]` where the DataFrame has columns `Sample ID`, `{field_name}`, `Evidence Note`).
- Produces: `collect_llm_draws(api_key, model_id, base_prompt, field_name, figure_number, figure_image, pass1_extracted, n_draws, call_refine=None) -> dict[str, list[dict]]` mapping `Sample ID` to a list of `{"value": float, "evidence": str}` draws (one entry per successful, numeric draw for that sample; draws where the model wrote `"-"` for a sample are skipped for that sample, not an error).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repeated_sampling.py
from pathlib import Path

import pandas as pd

from repeated_sampling import collect_llm_draws


def test_collect_llm_draws_across_multiple_calls():
    call_count = {"n": 0}

    def fake_call_refine(api_key, model_id, base_prompt, field_name, figure_number, figure_image, pass1_extracted):
        call_count["n"] += 1
        value = 3.500 + 0.001 * call_count["n"]
        frame = pd.DataFrame(
            [
                {"Sample ID": "VA0.6-300", field_name: f"{value:.3f}", "Evidence Note": "calc-derived"},
                {"Sample ID": "VA0-300", field_name: "-", "Evidence Note": "not plotted for 0% pre-strain"},
            ]
        )
        return frame, {}

    per_sample = collect_llm_draws(
        api_key="fake-key",
        model_id="fake-model",
        base_prompt="base prompt text",
        field_name="Longitudinal Stress at HEL (GPa)",
        figure_number="5",
        figure_image=Path("/tmp/fake.png"),
        pass1_extracted=pd.DataFrame([{"Sample ID": "VA0.6-300"}, {"Sample ID": "VA0-300"}]),
        n_draws=3,
        call_refine=fake_call_refine,
    )

    assert len(per_sample["VA0.6-300"]) == 3
    assert per_sample["VA0.6-300"][0]["value"] == 3.501
    assert per_sample["VA0.6-300"][0]["evidence"] == "calc-derived"
    # "-" draws are skipped for that sample, not recorded as a bad float
    assert "VA0-300" not in per_sample or per_sample["VA0-300"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_repeated_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'repeated_sampling'`

- [ ] **Step 3: Write minimal implementation**

```python
# figure_refine_pipeline/repeated_sampling.py
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
) -> dict[str, list[dict]]:
    if call_refine is None:
        from run_figure_refine import refine_field as call_refine  # noqa: N813

    per_sample: dict[str, list[dict]] = {}
    for _ in range(n_draws):
        patch, _usage = call_refine(
            api_key, model_id, base_prompt, field_name, figure_number, figure_image, pass1_extracted
        )
        for _, row in patch.iterrows():
            sample_id = str(row["Sample ID"]).strip()
            raw_value = row[field_name]
            evidence = str(row.get("Evidence Note", ""))
            try:
                value = float(str(raw_value).split("±")[0].strip())
            except (ValueError, TypeError):
                continue  # "-" or unparseable: this draw has nothing for this sample
            per_sample.setdefault(sample_id, []).append({"value": value, "evidence": evidence})
    return per_sample
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_repeated_sampling.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add figure_refine_pipeline/repeated_sampling.py tests/test_repeated_sampling.py
git commit -m "feat: collect repeated refine_field draws per sample"
```

---

### Task 10: CV anchor measurement + repeated match-only draws

**Files:**
- Modify: `figure_refine_pipeline/repeated_sampling.py`
- Test: `tests/test_repeated_sampling.py`

**Interfaces:**
- Consumes: `load_gray`, `find_axis_box`, `classify_markers`, `pixel_to_value` from `digitize_figure` (existing, unmodified); `annotate_markers`, `parse_match_table`, `MATCH_PROMPT_TEMPLATE` from `hybrid_match` (existing, unmodified); `build_shot_table` from `run_figure_refine` (existing).
- Produces: `collect_cv_anchor(image_path: Path, y_min: float, y_max: float) -> list[dict]` (list of detected filled markers, each with `marker_number`, `cx`, `cy`, `value`, deterministic — called once, not repeated); `collect_match_draws(api_key, model_id, image_path, filled_markers, field_name, figure_number, pass1_extracted, n_draws, call_match=None) -> dict[str, list[str]]` mapping `Sample ID` to a list of marker-number label strings, one per draw (this is what `reconcile.majority_vote_categorical` consumes — matching is the stochastic part when a CV anchor exists, not the value itself).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repeated_sampling.py (append)
from repeated_sampling import collect_match_draws


def test_collect_match_draws_across_multiple_calls():
    call_count = {"n": 0}

    def fake_call_match(api_key, model_id, image_path, prompt):
        call_count["n"] += 1
        # first two calls agree, third disagrees -- exercises majority-vote input shape
        marker = "1" if call_count["n"] != 3 else "2"
        return f"| Sample ID | Marker Number |\n| --- | --- |\n| VA0.6-300 | {marker} |\n", {}

    filled_markers = [
        {"marker_number": 1, "cx": 10.0, "cy": 20.0, "value": 3.5},
        {"marker_number": 2, "cx": 30.0, "cy": 40.0, "value": 1.5},
    ]

    per_sample = collect_match_draws(
        api_key="fake-key",
        model_id="fake-model",
        image_path=Path("/tmp/fake.png"),
        filled_markers=filled_markers,
        field_name="Spall Strength (GPa)",
        figure_number="4",
        pass1_extracted=pd.DataFrame([{"Sample ID": "VA0.6-300"}]),
        n_draws=3,
        call_match=fake_call_match,
    )

    assert per_sample["VA0.6-300"] == ["1", "1", "2"]
```

(`import pandas as pd` and `from pathlib import Path` already present at top of the test file from Task 9.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_repeated_sampling.py -v -k collect_match_draws`
Expected: FAIL with `ImportError: cannot import name 'collect_match_draws'`

- [ ] **Step 3: Write minimal implementation**

```python
# figure_refine_pipeline/repeated_sampling.py (append)
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
) -> dict[str, list[str]]:
    from hybrid_match import MATCH_PROMPT_TEMPLATE, annotate_markers, parse_match_table
    from run_figure_refine import build_shot_table

    if call_match is None:

        def call_match(api_key, model_id, image_path, prompt):  # noqa: ANN001
            from run_figure_refine import run_gemini, upload_file

            client, uploaded = upload_file(api_key, image_path)
            text, usage = run_gemini(api_key, model_id, [uploaded, prompt])
            client.files.delete(name=uploaded.name)
            return text, usage

    annotated_path = image_path.with_name(image_path.stem + "_annotated.png")
    if not annotated_path.exists():
        from digitize_figure import load_gray

        annotate_markers(load_gray(image_path), filled_markers, annotated_path)

    shot_table = build_shot_table(pass1_extracted)
    prompt = MATCH_PROMPT_TEMPLATE.format(
        figure_number=figure_number, field_name=field_name, shot_table=shot_table
    )

    per_sample: dict[str, list[str]] = {}
    for _ in range(n_draws):
        text, _usage = call_match(api_key, model_id, annotated_path, prompt)
        mapping = parse_match_table(text)
        for sample_id, marker_label in mapping.items():
            per_sample.setdefault(str(sample_id).strip(), []).append(marker_label)
    return per_sample
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_repeated_sampling.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add figure_refine_pipeline/repeated_sampling.py tests/test_repeated_sampling.py
git commit -m "feat: add CV anchor measurement and repeated shot-matching draws"
```

---

### Task 11: Fix `apply_patch()` to respect confidence and preserve evidence

**Files:**
- Modify: `figure_refine_pipeline/run_figure_refine.py:374-399`
- Test: `tests/test_apply_patch.py`

**Interfaces:**
- Consumes: `ReconciliationResult` from `reconcile.py`.
- Produces: `apply_patch(extracted: pd.DataFrame, evidence: pd.DataFrame, field_name: str, figure_number: str, sample_results: dict[str, ReconciliationResult]) -> None`. This **changes the existing signature** — it now takes a `dict[Sample ID, ReconciliationResult]` instead of a flat patch `DataFrame`, since confidence is decided per sample. Behavior: only overwrites `extracted[field_name]` for samples where `confidence == "high"`; for `"low"` confidence, leaves the original Pass-1 value untouched and sets that sample's `"Needs Review"` column (created if absent) to the field name; evidence text is appended (`"; "`-joined) rather than replacing the row outright.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_apply_patch.py
import pandas as pd

from reconcile import ReconciliationResult
from run_figure_refine import apply_patch


def test_apply_patch_overwrites_only_high_confidence_and_preserves_evidence():
    extracted = pd.DataFrame(
        [
            {"Sample ID": "VA0.6-300", "Longitudinal Stress at HEL (GPa)": "1.0 (pass1 guess)"},
            {"Sample ID": "VA0-300", "Longitudinal Stress at HEL (GPa)": "1.0 (pass1 guess)"},
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "Column Name": "Longitudinal Stress at HEL (GPa)",
                "Source Location": "Page 6, Fig. 5(a) & CALCULATED",
                "Notes": "[Priority 3/2]: original pass-1 evidence text",
            }
        ]
    )
    sample_results = {
        "VA0.6-300": ReconciliationResult(
            value=3.5015, confidence="high", majority_fraction=0.8,
            outliers=[{"value": 1.02, "evidence": "visually extracted, low confidence"}],
        ),
        "VA0-300": ReconciliationResult(
            value=None, confidence="low", majority_fraction=0.4, outliers=[],
        ),
    }

    apply_patch(extracted, evidence, "Longitudinal Stress at HEL (GPa)", "5", sample_results)

    assert extracted.loc[extracted["Sample ID"] == "VA0.6-300", "Longitudinal Stress at HEL (GPa)"].iloc[0] == 3.5015
    # low-confidence sample keeps its original Pass-1 value, untouched
    assert extracted.loc[extracted["Sample ID"] == "VA0-300", "Longitudinal Stress at HEL (GPa)"].iloc[0] == "1.0 (pass1 guess)"
    assert extracted.loc[extracted["Sample ID"] == "VA0-300", "Needs Review"].iloc[0] == "Longitudinal Stress at HEL (GPa)"

    updated_note = evidence.loc[evidence["Column Name"] == "Longitudinal Stress at HEL (GPa)", "Notes"].iloc[0]
    assert "original pass-1 evidence text" in updated_note  # preserved, not replaced
    assert "HIGH-RES REVIEW" in updated_note  # new info appended
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_apply_patch.py -v`
Expected: FAIL — `TypeError` (old `apply_patch` expects a `patch: pd.DataFrame`, not `sample_results: dict`) or `AssertionError` on the low-confidence/append behavior, since neither exists yet.

- [ ] **Step 3: Modify implementation**

Replace `apply_patch()` in `figure_refine_pipeline/run_figure_refine.py` (currently lines 374-399):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_apply_patch.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add figure_refine_pipeline/run_figure_refine.py tests/test_apply_patch.py
git commit -m "fix: apply_patch respects per-sample confidence, preserves evidence"
```

---

### Task 12: Per-field audit log

**Files:**
- Create: `figure_refine_pipeline/audit_log.py`
- Test: `tests/test_audit_log.py`

**Interfaces:**
- Consumes: `ReconciliationResult` (`reconcile.py`).
- Produces: `write_audit_log(paper_dir: Path, field_name: str, sample_results: dict[str, ReconciliationResult]) -> None`. Appends to (creates if absent) `{paper_dir}/audit_log.json`, one entry per `(field_name, sample_id)` recording the final value, confidence, majority fraction, and every excluded outlier's value + evidence text -- so an excluded outlier's reasoning is never silently lost, per the spec (an outlier draw's evidence text is exactly what previously revealed the Paper12 P2/P3 strategy-switching bug).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audit_log.py
import json
from pathlib import Path

from audit_log import write_audit_log
from reconcile import ReconciliationResult


def test_write_audit_log_creates_and_appends(tmp_path):
    paper_dir = tmp_path / "Paper12"
    paper_dir.mkdir()

    sample_results = {
        "VA0.6-300": ReconciliationResult(
            value=3.5015, confidence="high", majority_fraction=0.8,
            outliers=[{"value": 1.02, "evidence": "visually extracted, could not find u_HEL"}],
        ),
        "VA0-300": ReconciliationResult(
            value=None, confidence="low", majority_fraction=0.4, outliers=[],
        ),
    }

    write_audit_log(paper_dir, "Longitudinal Stress at HEL (GPa)", sample_results)
    # a second field's results should append, not overwrite
    write_audit_log(
        paper_dir, "Spall Strength (GPa)",
        {"VA0.6-300": ReconciliationResult(value=1.21, confidence="high", majority_fraction=1.0, outliers=[])},
    )

    entries = json.loads((paper_dir / "audit_log.json").read_text())
    assert len(entries) == 3

    hel_entry = next(e for e in entries if e["field_name"] == "Longitudinal Stress at HEL (GPa)" and e["sample_id"] == "VA0.6-300")
    assert hel_entry["value"] == 3.5015
    assert hel_entry["confidence"] == "high"
    assert hel_entry["majority_fraction"] == 0.8
    assert hel_entry["outliers"][0]["evidence"] == "visually extracted, could not find u_HEL"

    flagged_entry = next(e for e in entries if e["sample_id"] == "VA0-300")
    assert flagged_entry["value"] is None
    assert flagged_entry["confidence"] == "low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audit_log.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'audit_log'`

- [ ] **Step 3: Write minimal implementation**

```python
# figure_refine_pipeline/audit_log.py
"""Per-paper audit trail: every field/sample's reconciled value, confidence,
and excluded outlier draws (with their raw evidence text). Never silently
discard an outlier draw's reasoning -- see
docs/superpowers/specs/2026-07-28-t3-figure-pipeline-redesign-design.md,
"Audit trail" -- an excluded outlier's evidence text is exactly what
previously revealed the Paper12 P2/P3 strategy-switching bug.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_audit_log(paper_dir: Path, field_name: str, sample_results: dict) -> None:
    log_path = paper_dir / "audit_log.json"
    entries = json.loads(log_path.read_text()) if log_path.exists() else []
    for sample_id, result in sample_results.items():
        entries.append(
            {
                "field_name": field_name,
                "sample_id": sample_id,
                "value": result.value,
                "confidence": result.confidence,
                "majority_fraction": result.majority_fraction,
                "outliers": result.outliers,
            }
        )
    log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audit_log.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add figure_refine_pipeline/audit_log.py tests/test_audit_log.py
git commit -m "feat: add per-field audit log preserving outlier draws"
```

---

### Task 13: Wire the T2 gate + T3 routing/reconciliation into `run_pipeline()`

**Files:**
- Modify: `figure_refine_pipeline/run_figure_refine.py` (`run_pipeline()`, currently lines 402-518)

**Interfaces:**
- Consumes: `detect_dual_path_fields`, `identify_calculation`, `resolve_via_t2_gate` (`t2_gate.py`); `classify_figure_type`, `cv_available_for` (`figure_classify.py`); `collect_llm_draws`, `collect_cv_anchor`, `collect_match_draws` (`repeated_sampling.py`); `cluster_and_reconcile`, `majority_vote_categorical`, `draw_count`, `ReconciliationResult` (`reconcile.py`); the fixed `apply_patch` (Task 11); `write_audit_log` (Task 12).
- Produces: `run_pipeline()` with the same external signature and CLI behavior as today, but internally routing dual-path fields through the T2 gate before any figure work, and remaining T3 fields through figure-type classification, repeated sampling, and reconciliation instead of a single `refine_field()` + unconditional `apply_patch()` call.

There is no automated test for this task — it's integration wiring of already-unit-tested pieces, and the pipeline makes real API calls end to end. It's verified manually in Task 13 (the Paper12 pilot), per the spec's rollout plan.

- [ ] **Step 1: Replace the figure-fields loop in `run_pipeline()`**

In `figure_refine_pipeline/run_figure_refine.py`, replace the block from `# --- Identify figure-sourced (T3) fields ---` through the end of the `for field_name, fig_num in figure_fields.items():` loop (currently lines 444-491) with:

```python
    # --- T2 gate: dual-path fields (both a calc and a figure source in
    # Pass-1's own evidence) get a chance to resolve deterministically,
    # per-sample, before ever touching a figure. Narrow scope: only fields
    # Pass-1 itself flagged as dual-path -- see t2_gate.py docstring.
    from t2_gate import detect_dual_path_fields, identify_calculation, resolve_via_t2_gate

    evidence_rows = load_evidence_rows_from_frame(evidence)
    dual_path_fields = detect_dual_path_fields(evidence_rows)
    print(f"[t2-gate] dual-path fields: {dual_path_fields or 'none'}")

    unresolved_samples_by_field: dict[str, list[str]] = {}
    for field_name, fig_num in dual_path_fields.items():
        note_row = evidence[evidence["Column Name"] == field_name]
        evidence_note = note_row["Notes"].iloc[0] if not note_row.empty else ""
        calc_spec = identify_calculation(field_name, evidence_note, api_key, model_id)
        if calc_spec is None:
            print(f"  [t2-gate] no closed-form formula identified for '{field_name}', falling through to T3")
            unresolved_samples_by_field[field_name] = extracted["Sample ID"].astype(str).str.strip().tolist()
            continue
        resolved = resolve_via_t2_gate(calc_spec, extracted)
        still_unresolved = []
        for sample_id, value in resolved.items():
            if value is None:
                still_unresolved.append(sample_id)
                continue
            row_mask = extracted["Sample ID"].astype(str).str.strip() == sample_id
            extracted.loc[row_mask, field_name] = value
        print(
            f"  [t2-gate] '{field_name}': resolved {len(resolved) - len(still_unresolved)}/"
            f"{len(resolved)} samples deterministically; {len(still_unresolved)} fall through to T3"
        )
        if still_unresolved:
            unresolved_samples_by_field[field_name] = still_unresolved

    # --- Identify remaining figure-sourced (T3) fields (unchanged from before,
    # covers fields Pass-1 tagged Priority-3 with no dual-path calc option) ---
    figure_fields = find_figure_fields(evidence_rows)
    # merge in the T2-gate leftovers, which weren't necessarily tagged P3
    for field_name, fig_num in dual_path_fields.items():
        if field_name in unresolved_samples_by_field:
            figure_fields.setdefault(field_name, fig_num)
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
            print(f"  [skip] localization failed for '{field_name}' <- Fig. {fig_num} (page {page}): {exc}")
            print(f"         keeping Pass 1 value for '{field_name}' unchanged")
            continue

        figure_type = classify_figure_type(api_key, model_id, image_path)
        has_cv = cv_available_for(figure_type)
        print(f"  [classify] '{field_name}' figure type = {figure_type} (CV anchor: {has_cv})")
        n_draws = draw_count(has_cv_anchor=has_cv)

        sample_results = {}
        if has_cv:
            filled_markers = collect_cv_anchor(image_path, y_min=0.0, y_max=1.0)
            match_draws = collect_match_draws(
                api_key, model_id, image_path, filled_markers, field_name, fig_num, extracted, n_draws
            )
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
            per_sample_draws = collect_llm_draws(
                api_key, model_id, base_prompt, field_name, fig_num, image_path, extracted, n_draws
            )
            for sample_id, draws in per_sample_draws.items():
                sample_results[sample_id] = cluster_and_reconcile(draws)

        apply_patch(extracted, evidence, field_name, fig_num, sample_results)
        write_audit_log(paper_dir, field_name, sample_results)
        refine_metadata[field_name] = {
            "figure_number": fig_num,
            "page": page,
            "figure_type": figure_type,
            "cv_anchor": has_cv,
            "n_draws": n_draws,
        }
```

Also add the import at the top of the file (near the other stdlib/pandas imports): `from reconcile import ReconciliationResult`.

- [ ] **Step 2: Run the existing full test suite to confirm nothing else broke**

Run: `python -m pytest tests/ -v`
Expected: PASS for all tests (the new tests from Tasks 1-11, plus every pre-existing test in `tests/` — this task only touches `run_pipeline()`, which has no direct unit test, so this step is a regression check, not new coverage).

- [ ] **Step 3: Commit**

```bash
git add figure_refine_pipeline/run_figure_refine.py
git commit -m "feat: wire T2 gate and T3 classification/reconciliation into run_pipeline"
```

---

### Task 14: Paper12 single-field pilot (manual verification)

**Files:** none (verification step, no code changes)

This is the spec's rollout step 1: validate the T2 gate mechanism end to end against the exact failure mode this whole redesign targets, before running it on more fields or papers. Not automated — requires a live `GEMINI_API_KEY` and real API spend, so it's run by the user, not as part of this plan's automated steps.

- [ ] **Step 1: Run the pipeline on Paper12**

```bash
export GEMINI_API_KEY=...
cd figure_refine_pipeline
python run_figure_refine.py \
    --pdf "../Papers/Sample Papers/Paper12.pdf" \
    --prompt ../prompts/prompty_frozen.md \
    --paper-id Paper12 \
    --split development \
    --output-dir ./artifacts/t2_gate_pilot
```

- [ ] **Step 2: Confirm the T2 gate fired on `Longitudinal Stress at HEL (GPa)`**

Check the console output for a `[t2-gate]` line naming this field, and confirm `artifacts/t2_gate_pilot/development/Paper12/extracted_data.csv` has `Longitudinal Stress at HEL (GPa)` values close to `3.35` for whichever samples the console output reports as deterministically resolved -- per the design investigation, VA0.6 and VA5.5 are the samples most likely to resolve this way (their `u_HEL` is typically stated directly in the paper's text), while VA0 is the sample most likely to fall through to the T3 figure path (its `u_HEL` was not reliably recoverable from text in the original investigation). Which specific samples resolve via the gate is run-dependent -- Pass 1's text recovery of `u_HEL` varies draw to draw -- so treat the `[t2-gate]` log line itself, not a fixed sample list, as the source of truth for what to check.

- [ ] **Step 3: Confirm evidence text was preserved, not replaced**

Check `artifacts/t2_gate_pilot/development/Paper12/evidence_source.csv` for this field's `Notes` — it should still contain the original Pass-1 reasoning text, not just a bare `"HIGH-RES REVIEW"` string.

This step has no `git commit` — it produces run artifacts under `figure_refine_pipeline/artifacts/`, which are not committed (matching how existing pilot runs like `artifacts/t3_batch/` are handled in this repo).
