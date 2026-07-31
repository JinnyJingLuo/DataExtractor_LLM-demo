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
)


def _try_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def align_draws_by_position(
    selected: pd.DataFrame, draws: list[pd.DataFrame]
) -> list[pd.DataFrame | None]:
    """Align each draw to `selected` by row position, not by Sample ID text.

    Pass-1's own Sample ID text is unstable across draws when a paper has
    no clear native sample-naming scheme for the model to extract --
    observed on Paper13: draws 1/2 used bare numbers "1", "2", "3"... while
    draw 3 used "Expt 1", "Expt 2"... for the exact same 29 experiments in
    the exact same order (verified against other columns, e.g. Impact
    Velocity, which lined up exactly row-by-row across all three draws).
    Matching by text made every one of that draw's rows look "missing" from
    the others -- reconcile_pass1_table silently discarded a third of the
    sampling effort with no visible signal, since its confidence math just
    shrank its own denominator to match rather than flagging the loss.

    Only trusts a draw for position-based alignment when its row count
    matches `selected`'s -- a draw with a different row count can't be
    safely assumed to list the same shots in the same order (it may have
    dropped or added a row), so it's excluded entirely (None) rather than
    risk silently pairing unrelated rows together, which would be worse
    than the text-matching bug this replaces.

    Returns one entry per draw, in the same order as `draws`; None for any
    draw that doesn't qualify.
    """
    n = len(selected)
    return [draw.reset_index(drop=True) if len(draw) == n else None for draw in draws]


def reconcile_pass1_table(
    selected: pd.DataFrame, draws: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile Table 1 across repeated Pass-1 draws, per row-position/field.

    Rows are aligned across draws by position (see align_draws_by_position),
    not by matching Sample ID text -- so "Sample ID" itself is now just
    another field to reconcile like any other (majority-voted across draws,
    "-" competing as a valid label same as any text field), rather than the
    fragile join key it used to be.

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
    sample_id, field_name, value, confidence, majority_fraction. "Sample ID"
    itself is reconciled into the table but excluded from audit_rows -- it's
    an alignment key, not a scientific measurement, so its own cross-draw
    agreement shouldn't count toward downstream confidence/accuracy stats
    (those numbers should reflect the actual extracted values only).
    """
    fields = ["Sample ID"] + [c for c in selected.columns if c != "Sample ID"]
    aligned_draws = align_draws_by_position(selected, draws)

    reconciled = selected.copy()
    audit_rows = []
    for pos, (idx, selected_row) in enumerate(selected.iterrows()):
        # Reconciling "Sample ID" first (fields list starts with it) means
        # this reflects the reconciled label, not just the selected draw's
        # raw guess, for every other field's audit row in this same row.
        resolved_sample_id = str(selected_row["Sample ID"]).strip()
        for field_name in fields:
            observed = []
            for draw in aligned_draws:
                if draw is None or field_name not in draw.columns:
                    continue
                observed.append(_normalize_pass1_cell(draw.iloc[pos][field_name]))
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
            if field_name == "Sample ID" and final_value:
                resolved_sample_id = str(final_value).strip()
            if confidence != "high":
                if "Needs Review" not in reconciled.columns:
                    reconciled["Needs Review"] = ""
                existing = str(reconciled.at[idx, "Needs Review"] or "")
                reconciled.at[idx, "Needs Review"] = (
                    f"{existing}; {field_name}" if existing else field_name
                )

            if field_name == "Sample ID":
                continue
            audit_rows.append(
                {
                    "sample_id": resolved_sample_id,
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
