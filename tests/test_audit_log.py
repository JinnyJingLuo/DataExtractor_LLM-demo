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


def test_write_audit_log_replaces_stale_entry_on_rerun(tmp_path):
    # Simulates re-running the pipeline against the same output directory --
    # a second write for the same (field_name, sample_id) must replace the
    # first, not accumulate a duplicate, stale entry alongside it.
    paper_dir = tmp_path / "Paper12"
    paper_dir.mkdir()

    write_audit_log(
        paper_dir, "Spall Strength (GPa)",
        {"VA0-300": ReconciliationResult(value=1.0, confidence="low", majority_fraction=0.4, outliers=[])},
    )
    write_audit_log(
        paper_dir, "Spall Strength (GPa)",
        {"VA0-300": ReconciliationResult(value=1.21, confidence="high", majority_fraction=1.0, outliers=[])},
    )

    entries = json.loads((paper_dir / "audit_log.json").read_text())
    matching = [e for e in entries if e["field_name"] == "Spall Strength (GPa)" and e["sample_id"] == "VA0-300"]
    assert len(matching) == 1
    assert matching[0]["value"] == 1.21
    assert matching[0]["confidence"] == "high"
