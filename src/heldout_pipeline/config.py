from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationConfig:
    relative_tolerance: float
    absolute_tolerance: float
    numeric_floor: float
    null_values: frozenset[str]
    categorical_fields: frozenset[str]
    external_reference_fields: frozenset[str]
    allowed_tiers: frozenset[str]
    allowed_provenance: frozenset[str]
    ignored_fields: frozenset[str] = frozenset({"Verification", "source_file"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prompt(prompt_path: Path, manifest_path: Path) -> str:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = str(manifest.get("sha256", "")).lower()
    actual = sha256_file(prompt_path)
    if expected != actual:
        raise ConfigError(f"prompt hash mismatch: expected {expected}, got {actual}")
    return actual


def load_evaluation_config(path: Path) -> EvaluationConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvaluationConfig(
        relative_tolerance=float(data["relative_tolerance"]),
        absolute_tolerance=float(data["absolute_tolerance"]),
        numeric_floor=float(data["numeric_floor"]),
        null_values=frozenset(str(v).strip().casefold() for v in data["null_values"]),
        categorical_fields=frozenset(data["categorical_fields"]),
        external_reference_fields=frozenset(data["external_reference_fields"]),
        allowed_tiers=frozenset(data["allowed_tiers"]),
        allowed_provenance=frozenset(data["allowed_provenance"]),
        ignored_fields=frozenset(data.get("ignored_fields", ["Verification", "source_file"])),
    )
