import pandas as pd

from heldout_pipeline.record_matching import apply_record_matching, match_records


def long(rows):
    return pd.DataFrame(rows, columns=["paper_id", "sample_id", "field_name", "value"])


def test_match_records_pairs_unmatched_samples_by_anchor_values():
    gt = long(
        [
            ("P1", "shot_1", "Metal Symbol", "Cu"),
            ("P1", "shot_1", "Impact Velocity (m/s)", 300),
            ("P1", "shot_1", "Spall Strength (GPa)", 1.2),
            ("P1", "shot_2", "Metal Symbol", "Cu"),
            ("P1", "shot_2", "Impact Velocity (m/s)", 500),
            ("P1", "shot_2", "Spall Strength (GPa)", 1.5),
        ]
    )
    pred = long(
        [
            ("P1", "1", "Metal Symbol", "Cu"),
            ("P1", "1", "Impact Velocity (m/s)", 301),
            ("P1", "1", "Spall Strength (GPa)", 1.2),
            ("P1", "2", "Metal Symbol", "Cu"),
            ("P1", "2", "Impact Velocity (m/s)", 499),
            ("P1", "2", "Spall Strength (GPa)", 1.5),
        ]
    )

    review = match_records(gt, pred)

    matched = review[review["record_status"].eq("record_matched")].sort_values(
        "ground_truth_sample_id"
    )
    assert matched[["ground_truth_sample_id", "prediction_sample_id"]].to_dict(
        "records"
    ) == [
        {"ground_truth_sample_id": "shot_1", "prediction_sample_id": "1"},
        {"ground_truth_sample_id": "shot_2", "prediction_sample_id": "2"},
    ]
    assert matched["match_score"].min() >= 0.9


def test_record_matching_preserves_exact_matches_and_reports_unmatched_records():
    gt = long(
        [
            ("P1", "same", "Metal Symbol", "Cu"),
            ("P1", "gt_only", "Metal Symbol", "Al"),
        ]
    )
    pred = long(
        [
            ("P1", "same", "Metal Symbol", "Cu"),
            ("P1", "pred_only", "Metal Symbol", "Ta"),
        ]
    )

    review = match_records(gt, pred)

    assert set(review["record_status"]) == {
        "exact_match",
        "ground_truth_only",
        "prediction_only",
    }
    exact = review[review["record_status"].eq("exact_match")].iloc[0]
    assert exact["ground_truth_sample_id"] == "same"
    assert exact["prediction_sample_id"] == "same"


def test_apply_record_matching_remaps_predictions_and_provenance():
    gt = long(
        [
            ("P1", "shot_1", "Metal Symbol", "Cu"),
            ("P1", "shot_1", "Impact Velocity (m/s)", 300),
        ]
    )
    pred = long(
        [
            ("P1", "1", "Metal Symbol", "Cu"),
            ("P1", "1", "Impact Velocity (m/s)", 300),
        ]
    )
    provenance = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "sample_id": "1",
                "field_name": "Impact Velocity (m/s)",
                "tier": "T1",
                "provenance": "paper_table",
                "review_status": "approved",
            }
        ]
    )

    remapped_pred, remapped_prov, review = apply_record_matching(gt, pred, provenance)

    assert set(remapped_pred["sample_id"]) == {"shot_1"}
    assert set(remapped_prov["sample_id"]) == {"shot_1"}
    assert review["record_status"].tolist() == ["record_matched"]
