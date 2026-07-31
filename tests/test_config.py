import hashlib
import json

import pytest

from heldout_pipeline.config import ConfigError, load_evaluation_config, verify_prompt


def test_verify_prompt_returns_expected_hash(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_bytes(b"frozen")
    digest = hashlib.sha256(b"frozen").hexdigest()
    manifest = tmp_path / "prompt_manifest.json"
    manifest.write_text(json.dumps({"sha256": digest}), encoding="utf-8")

    assert verify_prompt(prompt, manifest) == digest


def test_verify_prompt_rejects_changed_text(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("changed", encoding="utf-8")
    manifest = tmp_path / "prompt_manifest.json"
    manifest.write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")

    with pytest.raises(ConfigError, match="hash mismatch"):
        verify_prompt(prompt, manifest)


def test_load_evaluation_config(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text(
        json.dumps(
            {
                "relative_tolerance": 0.005,
                "absolute_tolerance": 0.0,
                "numeric_floor": 1e-12,
                "null_values": ["", "-"],
                "categorical_fields": ["Treatment"],
                "external_reference_fields": ["Density"],
                "allowed_tiers": ["T1", "NA"],
                "allowed_provenance": ["paper_text", "unknown"],
            }
        ),
        encoding="utf-8",
    )
    config = load_evaluation_config(path)
    assert config.relative_tolerance == 0.005
    assert "Treatment" in config.categorical_fields
