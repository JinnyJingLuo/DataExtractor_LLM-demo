import hashlib
import json
from dataclasses import replace
from pathlib import Path

import heldout_pipeline.api_runner as api_runner
import pytest
from heldout_pipeline.api_runner import (
    ClientResponse,
    RunSettings,
    run_paper,
)
from heldout_pipeline.clients import GoogleGenerativeAIClient
from heldout_pipeline.manifest import PaperRecord


class FakeClient:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = 0

    def generate(self, pdf_path, prompt, model_id, generation_config):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary failure")
        return ClientResponse(
            text="raw model response",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            remote_file_id="files/1",
            cleanup_success=True,
        )


def settings(tmp_path: Path, prompt: Path, prompt_manifest: Path):
    return RunSettings(
        artifact_root=tmp_path / "artifacts",
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
        model_id="gemini-test",
        generation_config={},
        max_attempts=2,
        retry_delay_seconds=0,
    )


def setup_prompt(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("frozen", encoding="utf-8")
    digest = hashlib.sha256(b"frozen").hexdigest()
    manifest = tmp_path / "prompt_manifest.json"
    manifest.write_text(json.dumps({"sha256": digest}), encoding="utf-8")
    return prompt, manifest


def test_saves_raw_response_and_complete_metadata(tmp_path):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    record = PaperRecord("H1", pdf, "heldout", True, "unseen")

    result = run_paper(record, settings(tmp_path, prompt, prompt_manifest), FakeClient(failures=1))

    paper_dir = tmp_path / "artifacts" / "heldout" / "H1"
    metadata = json.loads((paper_dir / "response_metadata.json").read_text())
    assert result.success is True
    assert (paper_dir / "raw_response.md").read_text() == "raw model response"
    assert metadata["attempts"] == 2
    assert metadata["retries"] == 1
    assert metadata["total_tokens"] == 120
    assert metadata["raw_response_sha256"]


def test_successful_run_is_resumable_without_new_api_call(tmp_path):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    record = PaperRecord("D1", pdf, "development", True, "old")
    client = FakeClient()
    run_paper(record, settings(tmp_path, prompt, prompt_manifest), client)
    result = run_paper(record, settings(tmp_path, prompt, prompt_manifest), client)
    assert result.skipped is True
    assert client.calls == 1


def test_provider_change_requires_force_or_separate_artifact_root(tmp_path):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    record = PaperRecord("D1", pdf, "development", True, "old")
    original = settings(tmp_path, prompt, prompt_manifest)
    run_paper(record, original, FakeClient())
    vertex = replace(
        original,
        provider="vertex",
        project="dataextractionllm-503420",
        location="global",
    )
    client = FakeClient()

    with pytest.raises(RuntimeError, match="request identity changed"):
        run_paper(record, vertex, client)

    metadata = json.loads(
        (
            tmp_path
            / "artifacts"
            / "development"
            / "D1"
            / "response_metadata.json"
        ).read_text()
    )
    assert client.calls == 0
    assert metadata["provider"] == "gemini_api"


def test_identical_pdf_at_new_path_is_resumable(tmp_path):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    first_pdf = tmp_path / "first" / "paper.pdf"
    first_pdf.parent.mkdir()
    first_pdf.write_bytes(b"%PDF-same")
    first_record = PaperRecord(
        "D1", first_pdf, "development", True, "old"
    )
    configured = settings(tmp_path, prompt, prompt_manifest)
    run_paper(first_record, configured, FakeClient())
    moved_pdf = tmp_path / "moved" / "renamed.pdf"
    moved_pdf.parent.mkdir()
    moved_pdf.write_bytes(b"%PDF-same")
    moved_record = PaperRecord(
        "D1", moved_pdf, "development", True, "old"
    )
    client = FakeClient()

    result = run_paper(moved_record, configured, client)

    assert result.skipped is True
    assert client.calls == 0


def test_legacy_metadata_without_provider_resumes_as_gemini_api(tmp_path):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    record = PaperRecord("D1", pdf, "development", True, "old")
    configured = settings(tmp_path, prompt, prompt_manifest)
    run_paper(record, configured, FakeClient())
    metadata_path = (
        tmp_path
        / "artifacts"
        / "development"
        / "D1"
        / "response_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    for key in ("provider", "project", "location"):
        metadata.pop(key)
    metadata_path.write_text(json.dumps(metadata))
    client = FakeClient()

    result = run_paper(record, configured, client)

    assert result.skipped is True
    assert client.calls == 0


def test_failed_run_persists_error_metadata(tmp_path):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    record = PaperRecord("H1", pdf, "heldout", True, "unseen")
    result = run_paper(record, settings(tmp_path, prompt, prompt_manifest), FakeClient(failures=5))
    metadata = json.loads(
        (tmp_path / "artifacts" / "heldout" / "H1" / "response_metadata.json").read_text()
    )
    assert result.success is False
    assert metadata["error_type"] == "RuntimeError"
    assert metadata["attempts"] == 2


def test_failed_run_does_not_persist_exception_secrets(tmp_path):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    record = PaperRecord("H1", pdf, "heldout", True, "unseen")

    class SecretFailureClient:
        def generate(self, pdf_path, prompt, model_id, generation_config):
            raise RuntimeError(
                "Authorization: Bearer super-secret-token "
                "api_key=private-api-key"
            )

    run_paper(
        record,
        settings(tmp_path, prompt, prompt_manifest),
        SecretFailureClient(),
    )
    metadata = json.loads(
        (
            tmp_path
            / "artifacts"
            / "heldout"
            / "H1"
            / "response_metadata.json"
        ).read_text()
    )
    serialized = json.dumps(metadata)
    assert "super-secret-token" not in serialized
    assert "private-api-key" not in serialized
    assert metadata["error_message"] == "Model request failed"


def test_persistence_failure_does_not_generate_second_response(
    tmp_path, monkeypatch
):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    record = PaperRecord("D1", pdf, "development", True, "old")
    client = FakeClient()
    original_write_json = api_runner._write_json

    def fail_success_metadata(path, value):
        if path.name == "response_metadata.json" and value.get("success"):
            raise OSError("disk full")
        original_write_json(path, value)

    monkeypatch.setattr(api_runner, "_write_json", fail_success_metadata)

    with pytest.raises(OSError, match="disk full"):
        run_paper(
            record,
            settings(tmp_path, prompt, prompt_manifest),
            client,
        )

    assert client.calls == 1


def test_incomplete_success_artifact_requires_explicit_force(tmp_path):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    record = PaperRecord("D1", pdf, "development", True, "old")
    artifact_dir = tmp_path / "artifacts" / "development" / "D1"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "raw_response.md").write_text("completed response")
    client = FakeClient()

    with pytest.raises(RuntimeError, match="incomplete response artifact"):
        run_paper(
            record,
            settings(tmp_path, prompt, prompt_manifest),
            client,
        )

    assert client.calls == 0


def test_force_recovers_malformed_metadata_without_mixing_old_raw(tmp_path):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    record = PaperRecord("D1", pdf, "development", True, "old")
    artifact_dir = tmp_path / "artifacts" / "development" / "D1"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "raw_response.md").write_text("old provider response")
    (artifact_dir / "response_metadata.json").write_text("{truncated")
    configured = replace(
        settings(tmp_path, prompt, prompt_manifest),
        force=True,
    )

    result = run_paper(record, configured, FakeClient(failures=5))

    metadata = json.loads(
        (artifact_dir / "response_metadata.json").read_text()
    )
    assert result.success is False
    assert metadata["success"] is False
    assert not (artifact_dir / "raw_response.md").exists()


def test_saves_provider_and_resolved_model_metadata(tmp_path):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    record = PaperRecord("D1", pdf, "development", True, "old")
    configured = replace(
        settings(tmp_path, prompt, prompt_manifest),
        provider="vertex",
        project="dataextractionllm-503420",
        location="global",
    )

    class ProvenanceClient(FakeClient):
        def generate(self, pdf_path, prompt, model_id, generation_config):
            return ClientResponse(
                text="raw",
                input_tokens=100,
                output_tokens=20,
                thinking_tokens=5,
                total_tokens=125,
                model_version="gemini-3.1-pro-preview-20260701",
                response_id="response-1",
            )

    run_paper(record, configured, ProvenanceClient())
    metadata = json.loads(
        (
            tmp_path
            / "artifacts"
            / "development"
            / "D1"
            / "response_metadata.json"
        ).read_text()
    )
    assert metadata["provider"] == "vertex"
    assert metadata["project"] == "dataextractionllm-503420"
    assert metadata["location"] == "global"
    assert metadata["thinking_tokens"] == 5
    assert metadata["model_version"] == "gemini-3.1-pro-preview-20260701"
    assert metadata["response_id"] == "response-1"


def test_real_client_reports_remote_cleanup_failure(tmp_path):
    class Usage:
        prompt_token_count = 10
        candidates_token_count = 2
        thoughts_token_count = 1
        total_token_count = 13

    class Response:
        text = "response"
        usage_metadata = Usage()
        model_version = "gemini-3.1-pro-preview"
        response_id = "developer-response"

    class Uploaded:
        name = "files/test"

    class Model:
        def generate_content(self, content):
            return Response()

    class SDK:
        def configure(self, api_key):
            pass

        def upload_file(self, **kwargs):
            return Uploaded()

        def GenerativeModel(self, *args, **kwargs):
            return Model()

        def delete_file(self, name):
            raise RuntimeError("cleanup failed")

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    client = GoogleGenerativeAIClient(api_key="test", sdk=SDK())
    response = client.generate(pdf, "prompt", "model", {})
    assert response.cleanup_success is False
    assert response.thinking_tokens == 1
    assert response.model_version == "gemini-3.1-pro-preview"
    assert response.response_id == "developer-response"
