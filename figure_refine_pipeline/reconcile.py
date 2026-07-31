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
    draws: list[dict], n_attempted: int | None = None, tolerance: float = 0.02, min_majority: float = 0.6
) -> ReconciliationResult:
    if not draws:
        return ReconciliationResult(value=None, confidence="low", majority_fraction=0.0, outliers=[])

    denominator = n_attempted if n_attempted is not None else len(draws)

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
    # denominator is normally >= len(draws) (n_attempted counts draws that
    # failed/were skipped too), but a duplicate Sample ID row within one
    # draw's parsed response table can make len(draws) exceed n_attempted --
    # a fraction is never meaningfully > 1.0 regardless of how that happens.
    fraction = min(len(best) / denominator, 1.0)
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
    number a shot was matched to, or which source-tier a field's evidence
    text indicates), where tolerance-clustering doesn't apply -- exact-match
    voting is correct here since labels are discrete, not continuous.

    Three confidence tiers, based on plurality rather than a flat
    threshold, so a real signal isn't discarded just because it falls short
    of an outright majority:
      - "high": one label has >= min_majority of the votes.
      - "medium": one label has strictly more votes than every other label
        (a genuine plurality leader), but doesn't reach min_majority --
        e.g. a 2-1-1 split. Returns that leader.
      - "low": no label has strictly more votes than all others -- a true
        tie for first place (e.g. 1-1-1, or 2-2-1). Returns None; nothing
        to lead with.

    Always returns the observed leading fraction (even at low confidence)
    so callers can log it for audit, the same way cluster_and_reconcile
    always reports majority_fraction.
    """
    if not labels:
        return None, "low", 0.0
    ranked = Counter(labels).most_common()
    top_label, top_count = ranked[0]
    fraction = top_count / len(labels)
    if fraction >= min_majority:
        return top_label, "high", fraction
    is_plurality = len(ranked) == 1 or ranked[1][1] < top_count
    if is_plurality:
        return top_label, "medium", fraction
    return None, "low", fraction


def draw_count(has_cv_anchor: bool, cv_match_draws: int = 3, llm_draws: int = 5) -> int:
    """Return repeated-read count for the selected figure-refinement path.

    Defaults preserve the original behavior: 3 draws when a CV anchor exists
    and 5 draws when the LLM must both measure and match.
    """
    return cv_match_draws if has_cv_anchor else llm_draws
