from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .config import sha256_file, verify_prompt
from .manifest import PaperRecord


@dataclass(frozen=True)
class ClientResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    total_tokens: int | None = None
    remote_file_id: str | None = None
    cleanup_success: bool | None = None
    model_version: str | None = None
    response_id: str | None = None


class GeminiClient(Protocol):
    def generate(
        self,
        pdf_path: Path,
        prompt: str,
        model_id: str,
        generation_config: dict,
    ) -> ClientResponse: ...


@dataclass(frozen=True)
class RunSettings:
    artifact_root: Path
    prompt_path: Path
    prompt_manifest_path: Path
    model_id: str
    generation_config: dict
    provider: str = "gemini_api"
    project: str | None = None
    location: str | None = None
    max_attempts: int = 2
    retry_delay_seconds: float = 2.0
    force: bool = False
    input_price_per_million_tokens_usd: float | None = None
    output_price_per_million_tokens_usd: float | None = None


@dataclass(frozen=True)
class RunResult:
    paper_id: str
    split: str
    success: bool
    skipped: bool
    artifact_dir: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _estimated_cost(response: ClientResponse, settings: RunSettings) -> float | None:
    if (
        response.input_tokens is None
        or response.output_tokens is None
        or settings.input_price_per_million_tokens_usd is None
        or settings.output_price_per_million_tokens_usd is None
    ):
        return None
    return (
        response.input_tokens * settings.input_price_per_million_tokens_usd
        + response.output_tokens * settings.output_price_per_million_tokens_usd
    ) / 1_000_000


def run_paper(
    record: PaperRecord,
    settings: RunSettings,
    client: GeminiClient,
) -> RunResult:
    prompt_hash = verify_prompt(settings.prompt_path, settings.prompt_manifest_path)
    artifact_dir = Path(settings.artifact_root) / record.split / record.paper_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = artifact_dir / "response_metadata.json"
    raw_path = artifact_dir / "raw_response.md"
    resume_identity = {
        "paper_id": record.paper_id,
        "split": record.split,
        "pdf_sha256": sha256_file(record.pdf_path),
        "prompt_sha256": prompt_hash,
        "provider": settings.provider,
        "project": settings.project,
        "location": settings.location,
        "model_id": settings.model_id,
        "generation_config": settings.generation_config,
    }
    previous = None
    if metadata_path.exists() and not settings.force:
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"malformed response metadata for {record.paper_id}; "
                "inspect it and rerun with --force"
            ) from exc
    if not settings.force:
        if raw_path.exists() and not (previous and previous.get("success")):
            raise RuntimeError(
                f"incomplete response artifact for {record.paper_id}; "
                "inspect it and rerun with --force"
            )
        if previous and previous.get("success") and not raw_path.exists():
            raise RuntimeError(
                f"incomplete response artifact for {record.paper_id}; "
                "inspect it and rerun with --force"
            )
        if previous and previous.get("success"):
            previous_identity = {
                key: previous.get(key) for key in resume_identity
            }
            previous_identity["provider"] = previous.get(
                "provider", "gemini_api"
            )
            if previous_identity == resume_identity:
                return RunResult(
                    record.paper_id,
                    record.split,
                    True,
                    True,
                    artifact_dir,
                )
            raise RuntimeError(
                f"request identity changed for {record.paper_id}; "
                "use a separate --artifact-root or rerun with --force"
            )
    elif raw_path.exists():
        raw_path.unlink()

    prompt = Path(settings.prompt_path).read_text(encoding="utf-8")
    started = time.monotonic()
    request = {
        **resume_identity,
        "pdf_path": str(record.pdf_path),
        "started_at": _utc_now(),
    }
    _write_json(artifact_dir / "request.json", request)

    error: Exception | None = None
    response: ClientResponse | None = None
    attempts = 0
    for attempt in range(1, settings.max_attempts + 1):
        attempts = attempt
        try:
            response = client.generate(
                record.pdf_path,
                prompt,
                settings.model_id,
                settings.generation_config,
            )
        except Exception as exc:
            error = exc
            if attempt < settings.max_attempts:
                time.sleep(settings.retry_delay_seconds)
        else:
            break

    if response is None:
        metadata = {
            **request,
            "success": False,
            "attempts": attempts,
            "retries": max(attempts - 1, 0),
            "input_tokens": None,
            "output_tokens": None,
            "thinking_tokens": None,
            "total_tokens": None,
            "model_version": None,
            "response_id": None,
            "duration_seconds": time.monotonic() - started,
            "estimated_cost_usd": None,
            "finished_at": _utc_now(),
            "error_type": type(error).__name__ if error else "UnknownError",
            "error_message": "Model request failed",
        }
        _write_json(metadata_path, metadata)
        return RunResult(record.paper_id, record.split, False, False, artifact_dir)

    raw_path.write_text(response.text, encoding="utf-8")
    metadata = {
        **request,
        "success": True,
        "attempts": attempts,
        "retries": attempts - 1,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "thinking_tokens": response.thinking_tokens,
        "total_tokens": response.total_tokens,
        "model_version": response.model_version,
        "response_id": response.response_id,
        "duration_seconds": time.monotonic() - started,
        "remote_file_id": response.remote_file_id,
        "cleanup_success": response.cleanup_success,
        "raw_response_sha256": hashlib.sha256(
            response.text.encode("utf-8")
        ).hexdigest(),
        "estimated_cost_usd": _estimated_cost(response, settings),
        "finished_at": _utc_now(),
        "error_type": None,
        "error_message": None,
    }
    _write_json(metadata_path, metadata)
    return RunResult(record.paper_id, record.split, True, False, artifact_dir)
