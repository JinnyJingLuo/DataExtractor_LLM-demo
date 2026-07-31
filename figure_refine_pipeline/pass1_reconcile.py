"""Reconcile repeated Pass-1 draws instead of trusting one arbitrarily
selected draw for everything downstream.

Two independent reconciliation steps:
  - reconcile_pass1_table(): per-cell, per-sample reconciliation of Table 1
    values across draws (numeric fields via tolerance-cluster + median,
    text fields via exact-match majority/plurality vote) -- the same
    machinery already used to reconcile repeated figure-read draws in T3.
  - reconcile_evidence_tiers(): per-field vote across draws' Table 2
    evidence text on which source-tier (direct / calculated / figure /
    dual-path / unclear) a field belongs to, so T1/T2-gate/T3 routing
    isn't decided by whichever single draw happened to run first.

Both reuse reconcile.cluster_and_reconcile / reconcile.majority_vote_categorical
directly rather than reimplementing voting logic -- per the principle that
the reconciliation algorithm shouldn't depend on where its inputs came from.
"""
from __future__ import annotations

import pandas as pd

from reconcile import cluster_and_reconcile, majority_vote_categorical
from run_figure_refine import (
    CALC_RE,
    DIRECT_RE,
    FIG_RE,
    PRIORITY3_RE,
    _normalize_pass1_cell,
    _normalize_sample_id_for_matching,
)


def _try_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _index_draws_by_sample_id(draws: list[pd.DataFrame]) -> list[pd.DataFrame]:
    indexed = []
    for draw in draws:
        if "Sample ID" not in draw.columns:
            continue
        normalized = draw.assign(
            **{"Sample ID": draw["Sample ID"].map(_normalize_sample_id_for_matching)}
        )
        indexed.append(
            normalized.drop_duplicates(subset=["Sample ID"], keep="first").set_index("Sample ID")
        )
    return indexed


def reconcile_pass1_table(
    selected: pd.DataFrame, draws: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile Table 1 across repeated Pass-1 draws, per sample/field.

    A field is treated as numeric if every non-missing observed value across
    draws parses as a float (reconciled via cluster_and_reconcile); otherwise
    it's treated as text (reconciled via majority_vote_categorical, where a
    "-"/missing value is itself a valid competing label, not discarded --
    if most draws couldn't find a value, "-" winning the vote is the honest
    answer). Non-"high" cells are flagged in "Needs Review", same mechanism
    already used by the T2/T3 patching path.

    Degenerates cleanly to "just use the one draw's value" when only one
    draw is supplied (single-sample cluster / single-label vote both return
    that value at "high" confidence) -- safe to call unconditionally.

    Returns (reconciled_table, audit_rows) where audit_rows has columns
    sample_id, field_name, value, confidence, majority_fraction.
    """
    fields = [c for c in selected.columns if c != "Sample ID"]
    indexed_draws = _index_draws_by_sample_id(draws)

    reconciled = selected.copy()
    audit_rows = []
    for idx, selected_row in selected.iterrows():
        sample_id = str(selected_row["Sample ID"]).strip()
        lookup_key = _normalize_sample_id_for_matching(sample_id)
        for field_name in fields:
            observed = []
            for draw in indexed_draws:
                if lookup_key not in draw.index or field_name not in draw.columns:
                    continue
                observed.append(_normalize_pass1_cell(draw.loc[lookup_key, field_name]))
            if not observed:
                continue

            non_missing = [v for v in observed if v != "-"]
            numeric_values = [_try_float(v) for v in non_missing]
            is_numeric_field = bool(non_missing) and all(v is not None for v in numeric_values)

            if is_numeric_field:
                cluster_draws = [{"value": v, "evidence": ""} for v in numeric_values]
                result = cluster_and_reconcile(cluster_draws, n_attempted=len(observed))
                final_value = result.value if result.value is not None else selected_row[field_name]
                confidence = result.confidence
                fraction = result.majority_fraction
            else:
                label, confidence, fraction = majority_vote_categorical(observed)
                final_value = label if label is not None else selected_row[field_name]

            reconciled.at[idx, field_name] = final_value
            if confidence != "high":
                if "Needs Review" not in reconciled.columns:
                    reconciled["Needs Review"] = ""
                existing = str(reconciled.at[idx, "Needs Review"] or "")
                reconciled.at[idx, "Needs Review"] = (
                    f"{existing}; {field_name}" if existing else field_name
                )

            audit_rows.append(
                {
                    "sample_id": sample_id,
                    "field_name": field_name,
                    "value": final_value,
                    "confidence": confidence,
                    "majority_fraction": fraction,
                }
            )
    return reconciled, pd.DataFrame(audit_rows)


def classify_evidence_tier(notes: str, location: str) -> str:
    """Classify one draw's evidence text for one field into a source tier.

    Priority order when a text matches multiple patterns: an explicit
    direct-source claim wins outright (trust the model's own framing);
    otherwise dual-path (calc + figure both mentioned) beats either alone;
    otherwise whichever single pattern matched; otherwise "unclear".
    """
    text = f"{notes} {location}"
    if DIRECT_RE.search(text):
        return "direct"
    is_calc = bool(CALC_RE.search(text))
    is_figure = bool(PRIORITY3_RE.search(text)) or bool(FIG_RE.search(text))
    if is_calc and is_figure:
        return "dual-path"
    if is_figure:
        return "figure-only"
    if is_calc:
        return "calculated-only"
    return "unclear"


def reconcile_evidence_tiers(evidence_draws: list[pd.DataFrame]) -> dict[str, dict]:
    """Vote across draws' Table 2 evidence on each field's source tier.

    Returns {field_name: {"tier", "confidence", "fraction", "figure_number",
    "representative_note"}}. "figure_number" is the majority-voted figure
    number among draws that mentioned one AND whose own tier agreed with the
    winning tier (None if none did) -- scoping it to the winner matters
    because classify_evidence_tier only ever returns "calculated-only" for a
    draw that mentioned no figure at all, so an unscoped pool lets a
    minority of figure-only/dual-path draws attach a figure number to a
    field the majority just voted "calculated-only" (no figure), silently
    contradicting the tier vote it's attached to. Observed on Paper12's
    "Spall Pullback Velocity": 3 of 5 draws classified it calculated-only
    (no figure), but 2 minority draws (figure-only, dual-path) each
    mentioned "Fig 3", which used to leak a figure_number="3" onto the
    calculated-only-tier result.
    "representative_note" is one draw's actual Notes text from among the
    draws that agreed with the winning tier, for display/downstream use --
    not synthesized.
    """
    field_tiers: dict[str, list[str]] = {}
    field_fig_nums_by_tier: dict[str, dict[str, list[str]]] = {}
    field_notes_by_tier: dict[str, dict[str, str]] = {}

    for draw_evidence in evidence_draws:
        for _, row in draw_evidence.iterrows():
            field_name = str(row.get("Column Name", "")).strip()
            if not field_name:
                continue
            notes = str(row.get("Notes", ""))
            location = str(row.get("Source Location", ""))
            tier = classify_evidence_tier(notes, location)
            field_tiers.setdefault(field_name, []).append(tier)
            match = FIG_RE.search(f"{notes} {location}")
            if match:
                field_fig_nums_by_tier.setdefault(field_name, {}).setdefault(tier, []).append(
                    match.group(1)
                )
            field_notes_by_tier.setdefault(field_name, {}).setdefault(tier, notes)

    results: dict[str, dict] = {}
    for field_name, tiers in field_tiers.items():
        winner, confidence, fraction = majority_vote_categorical(tiers)
        fig_num = None
        winning_fig_nums = field_fig_nums_by_tier.get(field_name, {}).get(winner or "", [])
        if winning_fig_nums:
            fig_num, _, _ = majority_vote_categorical(winning_fig_nums)
        results[field_name] = {
            "tier": winner,
            "confidence": confidence,
            "fraction": fraction,
            "figure_number": fig_num,
            "representative_note": field_notes_by_tier.get(field_name, {}).get(winner or "", ""),
        }
    return results
