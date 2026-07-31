import pytest

from reconcile import cluster_and_reconcile, draw_count, majority_vote_categorical


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
    draws = _draws([(3.50, "calc"), (3.50, "calc"), (1.02, "fig"), (1.03, "fig"), (7.00, "fig")])
    result = cluster_and_reconcile(draws)
    assert result.confidence == "low"
    assert result.majority_fraction == pytest.approx(0.4)


def test_majority_fraction_never_exceeds_one_when_draws_outnumber_n_attempted():
    # A duplicate Sample ID row within one draw's parsed response table can
    # make len(draws) exceed n_attempted -- the fraction must still be
    # capped at 1.0 rather than reporting e.g. 1.5.
    draws = _draws([(3.5, "a"), (3.5, "b"), (3.5, "c")])
    result = cluster_and_reconcile(draws, n_attempted=2)
    assert result.majority_fraction == pytest.approx(1.0)
    assert result.majority_fraction <= 1.0


def test_empty_draws_returns_low_confidence_none():
    result = cluster_and_reconcile([])
    assert result.value is None
    assert result.confidence == "low"
    assert result.outliers == []


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


def test_majority_vote_categorical_plurality_without_majority_is_medium():
    # 2-1-1 split: "A" leads, but only 2/4 = 50% < 60% majority threshold.
    label, confidence, fraction = majority_vote_categorical(["A", "A", "B", "C"])
    assert label == "A"
    assert confidence == "medium"
    assert fraction == pytest.approx(0.5)


def test_majority_vote_categorical_tied_for_first_is_low():
    # 2-2-1 split: "A" and "B" are tied for the lead -- no plurality winner.
    label, confidence, fraction = majority_vote_categorical(["A", "A", "B", "B", "C"])
    assert label is None
    assert confidence == "low"
    assert fraction == pytest.approx(0.4)


def test_draw_count_uses_cv_anchor():
    assert draw_count(has_cv_anchor=True) == 3
    assert draw_count(has_cv_anchor=False) == 5


def test_draw_count_accepts_configured_draw_counts():
    assert draw_count(has_cv_anchor=True, cv_match_draws=4, llm_draws=7) == 4
    assert draw_count(has_cv_anchor=False, cv_match_draws=4, llm_draws=7) == 7


def test_single_surviving_draw_among_many_attempted_is_low_confidence():
    draws = _draws([(1.02, "visually extracted from Fig. 5(a), one of five draws")])
    result = cluster_and_reconcile(draws, n_attempted=5)
    assert result.confidence == "low"
    assert result.majority_fraction == pytest.approx(0.2)
    # without n_attempted, the old (now-buggy) behavior is preserved for backward compat:
    result_no_denominator = cluster_and_reconcile(draws)
    assert result_no_denominator.confidence == "high"
