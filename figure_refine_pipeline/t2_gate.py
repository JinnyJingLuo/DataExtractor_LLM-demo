"""Narrow T2 gate for deterministic calculation before figure rereading.

The first pass sometimes labels a field as figure-derived even when all inputs
needed for a known physics calculation are already present. This module detects
those cases and resolves them in code before the T3 figure-refinement pass.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass

import pandas as pd

from run_figure_refine import CALC_RE, DIRECT_RE, FIG_RE


def detect_dual_path_fields(evidence_rows: list[dict]) -> dict[str, str]:
    """Return {field_name: figure_number} for calc+figure evidence rows."""
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


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_formula(formula: str, values: dict[str, float]) -> float:
    """Evaluate a restricted arithmetic expression against numeric values."""
    tree = ast.parse(formula, mode="eval")
    return _eval_node(tree.body, values)


def _eval_node(node: ast.AST, values: dict[str, float]) -> float:
    # bool is a subclass of int in Python (isinstance(True, int) is True),
    # so this must be checked before the int/float check below, or a
    # literal True/False in a formula string would silently evaluate as 1/0.
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        raise ValueError(f"unsupported constant in formula: {node.value!r}")
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


IDENTIFY_CALC_PROMPT_TEMPLATE = """You previously extracted data from a scientific paper. For the column
"{field_name}", your evidence notes indicate it can sometimes be calculated
from other extracted values rather than read from a figure:

{evidence_note}

The paper's extracted data table has exactly these columns available as
inputs -- you MUST map each variable to one of these EXACT strings
(copy-paste exact spelling, spacing, and any unit symbols):

{available_columns}

Identify the exact arithmetic formula used, expressed only with +, -, *, /,
**, and parentheses. If the formula genuinely requires unsupported operations,
respond with {{"formula": null}}. Include any unit conversion factor directly
in the formula.

Return ONLY a JSON object, no other text:
{{"formula": "0.5*rho0*c_l*u_HEL/1e6", "variables": {{"rho0": "Initial Density (g/cm^3)", "c_l": "Longitudinal Sound Speed (m/s)", "u_HEL": "Free Surface Velocity at HEL (m/s)"}}}}
"""


@dataclass
class CalculationSpec:
    formula: str
    variables: dict[str, str]


def _find_column(available_columns: list[str], required_terms: tuple[str, ...]) -> str | None:
    """Find the actual column name containing all required terms.

    Unit strings can contain superscripts or mojibake depending on where the
    file is viewed, so known formulas bind to the real parsed column names.
    """
    for column in available_columns:
        normalized = re.sub(r"\s+", " ", column.casefold())
        if all(term.casefold() in normalized for term in required_terms):
            return column
    return None


def known_calculation_specs(available_columns: list[str]) -> dict[str, CalculationSpec]:
    """Return deterministic formulas for high-value physics relations.

    These specs are intentionally narrow and only fire when all required input
    columns are present in the Pass-1 table.
    """
    density_col = _find_column(available_columns, ("initial density",))
    longitudinal_sound_col = _find_column(
        available_columns, ("longitudinal", "sound speed")
    )
    hel_velocity_col = _find_column(
        available_columns, ("free surface velocity", "hel")
    )
    bulk_modulus_col = _find_column(available_columns, ("bulk modulus",))
    shear_modulus_col = _find_column(available_columns, ("shear modulus",))

    specs: dict[str, CalculationSpec] = {}
    if density_col and longitudinal_sound_col and hel_velocity_col:
        specs["Longitudinal Stress at HEL (GPa)"] = CalculationSpec(
            # rho0 is g/cm^3; multiply by 1000 kg/m^3 and divide Pa by 1e9.
            formula="0.5*rho0*c_l*u_HEL/1e6",
            variables={
                "rho0": density_col,
                "c_l": longitudinal_sound_col,
                "u_HEL": hel_velocity_col,
            },
        )
    if bulk_modulus_col and shear_modulus_col:
        specs["Young's Modulus (GPa)"] = CalculationSpec(
            # Standard isotropic-elasticity relation; B and G are already
            # in GPa, no conversion factor needed.
            formula="9*B*G/(3*B+G)",
            variables={"B": bulk_modulus_col, "G": shear_modulus_col},
        )
    return specs


@dataclass
class ConfusableCandidate:
    label: str
    calc_spec: CalculationSpec


def known_confusable_specs(available_columns: list[str]) -> dict[str, list[ConfusableCandidate]]:
    """Alternative physical quantities a "direct"-tier field's reported
    value might actually be, for fields where a paper's own table can be
    ambiguous about which quantity it's reporting.

    Not a formula for the target field itself (that's known_calculation_specs)
    -- each entry here is a formula for a DIFFERENT, easily-confused
    quantity, used only to positively identify a mislabeling: if a field's
    reported value fits one of these far better than it fits the target
    field's own formula, the source table is very likely reporting this
    alternative quantity under the target's name, not the target itself.

    Paper12 case: "Young's Modulus" reported 226.0 GPa is a 42% mismatch
    against E=9BG/(3B+G)=131.9, but within 0.6% of the Longitudinal
    (P-wave) Modulus M=B+4G/3=227.4 -- and within 0.3% of the fully
    independent M=rho*c_l^2=226.6, computed from a completely different
    pair of input columns. Two independent formulas agreeing that closely
    is strong, quantifiable evidence of a mislabeled column, not just
    ambiguity between two plausible guesses.
    """
    bulk_modulus_col = _find_column(available_columns, ("bulk modulus",))
    shear_modulus_col = _find_column(available_columns, ("shear modulus",))
    density_col = _find_column(available_columns, ("initial density",))
    longitudinal_sound_col = _find_column(available_columns, ("longitudinal", "sound speed"))

    longitudinal_modulus_candidates: list[ConfusableCandidate] = []
    if bulk_modulus_col and shear_modulus_col:
        longitudinal_modulus_candidates.append(
            ConfusableCandidate(
                label="Longitudinal (P-wave) Modulus",
                calc_spec=CalculationSpec(
                    formula="B+4*G/3",
                    variables={"B": bulk_modulus_col, "G": shear_modulus_col},
                ),
            )
        )
    if density_col and longitudinal_sound_col:
        longitudinal_modulus_candidates.append(
            ConfusableCandidate(
                label="Longitudinal (P-wave) Modulus",
                calc_spec=CalculationSpec(
                    # rho0 is g/cm^3; multiply by 1000 kg/m^3 and divide Pa by 1e9.
                    formula="rho0*c_l**2/1e6",
                    variables={"rho0": density_col, "c_l": longitudinal_sound_col},
                ),
            )
        )

    candidates: dict[str, list[ConfusableCandidate]] = {}
    if longitudinal_modulus_candidates:
        candidates["Young's Modulus (GPa)"] = longitudinal_modulus_candidates
    return candidates


def validate_input_provenance(
    calc_spec: CalculationSpec,
    evidence_rows: list[dict],
    field_name: str | None = None,
    resolved_by_t2_gate: dict[str, CalculationSpec] | None = None,
) -> tuple[bool, str | None]:
    """Return (True, None) only if every input column calc_spec relies on is
    itself a direct (Priority-1) value in Pass-1's evidence, OR was itself
    already resolved deterministically by the T2 gate earlier this run (see
    `resolved_by_t2_gate` -- callers should process fields in dependency
    order via order_fields_by_dependency() so a field's T2-gate-computed
    inputs are already present here by the time it's checked).

    This rejects chains that loop back on themselves, not just literal
    A-depends-on-B-depends-on-A cycles: observed on Paper12's VA0 samples,
    the evidence for "Free Surface Velocity at HEL (m/s)" read "[Priority 2]:
    calculated from sigma_HEL using u_HEL = sigma_HEL / (0.5*rho0*C_l)" --
    that "input" was fabricated by rearranging the very formula this module
    applies, not read from the paper, so using it here computes sigma_HEL
    from a value that was itself derived from a (likely wrong) guess at
    sigma_HEL. Requiring every input to be a confirmed direct value (or a
    genuinely resolved T2-gate value) closes that whole class of problem,
    not just the one instance of it -- it does not require detecting the
    specific symbol/field the input was derived from. Returns (False,
    reason) for the first offending input found.

    A T2-gate-resolved input is only trusted if it isn't itself derived from
    `field_name` -- otherwise a genuine two-step circular pair (A needs B,
    B needs A) would trust each other in turn instead of both correctly
    failing.
    """
    evidence_by_column = {
        str(row.get("Column Name", "")).strip(): row for row in evidence_rows
    }
    resolved_by_t2_gate = resolved_by_t2_gate or {}
    for column in calc_spec.variables.values():
        upstream_spec = resolved_by_t2_gate.get(column)
        if upstream_spec is not None:
            if field_name is not None and field_name in upstream_spec.variables.values():
                return False, (
                    f"input column '{column}' was itself computed from '{field_name}' "
                    "by the T2 gate this run -- circular"
                )
            continue
        row = evidence_by_column.get(column)
        if row is None:
            return False, f"no evidence row found for input column '{column}'"
        text = f"{row.get('Notes', '')} {row.get('Source Location', '')}"
        if CALC_RE.search(text):
            return False, f"input column '{column}' is itself calculated/derived, not a direct value"
        if not DIRECT_RE.search(text):
            return False, f"input column '{column}' is not confirmed as a direct (Priority-1) value"
    return True, None


def order_fields_by_dependency(field_calc_specs: dict[str, CalculationSpec | None]) -> list[str]:
    """Order T2-gate-eligible fields so that any field depending on another
    T2-gate-eligible field (via a shared formula input) comes after it.

    Only edges between fields both present in field_calc_specs matter here --
    a formula's other inputs (leaf/direct values) aren't part of this graph
    at all. Uses Kahn's algorithm; any fields left over after the main pass
    are part of a genuine dependency cycle (or depend on one) and are
    appended in their original order -- they simply won't benefit from
    trusting an upstream T2-gate resolution, so validate_input_provenance
    falls back to its ordinary evidence-based check for them, which is the
    correct behavior for a real cycle.
    """
    field_names = list(field_calc_specs.keys())
    depends_on: dict[str, set[str]] = {name: set() for name in field_names}
    for field_name, calc_spec in field_calc_specs.items():
        if calc_spec is None:
            continue
        for column in calc_spec.variables.values():
            if column in field_calc_specs and column != field_name:
                depends_on[field_name].add(column)

    in_degree = {name: len(deps) for name, deps in depends_on.items()}
    dependents: dict[str, list[str]] = {name: [] for name in field_names}
    for field_name, deps in depends_on.items():
        for dep in deps:
            dependents[dep].append(field_name)

    queue = [name for name in field_names if in_degree[name] == 0]
    ordered: list[str] = []
    while queue:
        node = queue.pop(0)
        ordered.append(node)
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    remaining = [name for name in field_names if name not in ordered]
    return ordered + remaining


def identify_calculation(
    field_name: str,
    evidence_note: str,
    available_columns: list[str],
    api_key: str,
    model_id: str,
    call_llm=None,
) -> CalculationSpec | None:
    if call_llm is None:
        from run_figure_refine import run_gemini as call_llm  # noqa: N813

    columns_list = "\n".join(f"- {col}" for col in available_columns)
    prompt = IDENTIFY_CALC_PROMPT_TEMPLATE.format(
        field_name=field_name, evidence_note=evidence_note, available_columns=columns_list
    )
    text, _usage = call_llm(api_key, model_id, [prompt])
    # One level of nesting (the "variables" sub-object) is expected, so the
    # non-nesting r"\{[^{}]*\}" pattern used elsewhere (e.g. locate_figure_bbox)
    # won't capture the whole object -- but the naive greedy r"\{.*\}" this
    # used to be is worse: DOTALL + greedy means it spans from the FIRST "{"
    # to the LAST "}" anywhere in the response, silently merging unrelated
    # JSON-like fragments if the model's prose happens to contain more than
    # one brace pair. Balance exactly one level of nesting instead.
    match = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"could not parse calculation-identification JSON: {text!r}")
    payload = json.loads(match.group(0))
    if not payload.get("formula"):
        return None
    return CalculationSpec(formula=payload["formula"], variables=payload["variables"])


def _to_float(raw) -> float:
    text = str(raw).strip()
    if not text or text in {"-", "nan", "None"}:
        raise ValueError("missing numeric value")
    return float(text.split("+/-")[0].split("±")[0].strip())


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
            try:
                values[var] = _to_float(row[column])
            except (ValueError, TypeError):
                missing = True
                break
        results[sample_id] = None if missing else safe_eval_formula(calc_spec.formula, values)
    return results


def filter_unstable_inputs(
    resolved: dict[str, float | None],
    calc_spec: CalculationSpec,
    pass1_consistency: pd.DataFrame | None,
) -> dict[str, float | None]:
    """Downgrade a sample's T2-gate resolution back to unresolved if any
    input column it depends on wasn't stable ("high" confidence) across
    repeated Pass-1 draws for that specific sample.

    This is a secondary, empirical check alongside validate_input_provenance:
    the evidence text might claim an input is a direct value even when Pass-1
    itself can't reliably reproduce that same value draw to draw (this is
    exactly how VA0's "Free Surface Velocity at HEL (m/s)" behaved across
    this project's repeated testing -- sometimes ~56, sometimes ~180). A
    no-op when pass1_consistency is None (a single Pass-1 draw has nothing
    to cross-check against).
    """
    if pass1_consistency is None or pass1_consistency.empty:
        return resolved
    input_columns = set(calc_spec.variables.values())
    unstable_samples = set(
        pass1_consistency.loc[
            pass1_consistency["field_name"].isin(input_columns)
            & (pass1_consistency["confidence"] != "high"),
            "sample_id",
        ]
    )
    if not unstable_samples:
        return resolved
    filtered = dict(resolved)
    for sample_id in unstable_samples:
        if sample_id in filtered and filtered[sample_id] is not None:
            filtered[sample_id] = None
    return filtered
