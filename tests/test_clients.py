from types import SimpleNamespace

import pytest

from heldout_pipeline.clients import (
    ClaudeAPIClient,
    ClaudeVertexClient,
    GeminiFileAPIClient,
    VertexAIClient,
    build_client,
)


class FakeTypes:
    class Part:
        @staticmethod
        def from_text(*, text):
            return SimpleNamespace(text=text, data=None, mime_type=None)

        @staticmethod
        def from_bytes(*, data, mime_type):
            return SimpleNamespace(text=None, data=data, mime_type=mime_type)

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.values = kwargs


class FakeModelService:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_vertex_client_sends_one_pdf_request_and_maps_metadata(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")
    response = SimpleNamespace(
        text="result",
        usage_metadata=SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=7,
            thoughts_token_count=3,
            total_token_count=21,
        ),
        model_version="gemini-3.1-pro-preview-20260701",
        response_id="response-7",
    )
    service = FakeModelService(response)
    client = VertexAIClient(
        project="dataextractionllm-503420",
        location="global",
        client=SimpleNamespace(models=service),
        types_module=FakeTypes,
    )

    result = client.generate(
        pdf,
        "frozen prompt",
        "gemini-3.1-pro-preview",
        {},
    )

    assert len(service.calls) == 1
    assert service.calls[0]["model"] == "gemini-3.1-pro-preview"
    assert service.calls[0]["config"] is None
    assert service.calls[0]["contents"][0].text == "frozen prompt"
    assert service.calls[0]["contents"][1].data == b"%PDF-test"
    assert service.calls[0]["contents"][1].mime_type == "application/pdf"
    assert result.text == "result"
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert result.thinking_tokens == 3
    assert result.total_tokens == 21
    assert result.model_version == "gemini-3.1-pro-preview-20260701"
    assert result.response_id == "response-7"


def test_vertex_client_maps_explicit_generation_config(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")
    service = FakeModelService(SimpleNamespace(text="result", usage_metadata=None))
    client = VertexAIClient(
        project="dataextractionllm-503420",
        location="global",
        client=SimpleNamespace(models=service),
        types_module=FakeTypes,
    )

    client.generate(pdf, "prompt", "model", {"temperature": 0.2})

    assert service.calls[0]["config"].values == {"temperature": 0.2}


class FakeGeminiFiles:
    def __init__(self):
        self.uploads = []
        self.deletes = []

    def upload(self, **kwargs):
        self.uploads.append(kwargs)
        return SimpleNamespace(name="files/paper-123", uri="file-uri")

    def delete(self, **kwargs):
        self.deletes.append(kwargs)


def test_gemini_file_api_client_uploads_pdf_and_deletes_remote_file(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-file-api")
    response = SimpleNamespace(
        text="file api result",
        usage_metadata=SimpleNamespace(
            prompt_token_count=19,
            candidates_token_count=8,
            thoughts_token_count=2,
            total_token_count=29,
        ),
        model_version="gemini-3.1-pro-preview-20260701",
        response_id="response-file-7",
    )
    files = FakeGeminiFiles()
    service = FakeModelService(response)
    client = GeminiFileAPIClient(
        api_key="test-key",
        client=SimpleNamespace(files=files, models=service),
        types_module=FakeTypes,
    )

    result = client.generate(
        pdf,
        "frozen prompt",
        "gemini-3.1-pro-preview",
        {"temperature": 0},
    )

    assert files.uploads == [{"file": str(pdf)}]
    assert files.deletes == [{"name": "files/paper-123"}]
    assert service.calls[0]["model"] == "gemini-3.1-pro-preview"
    assert service.calls[0]["contents"][0].name == "files/paper-123"
    assert service.calls[0]["contents"][1] == "frozen prompt"
    assert service.calls[0]["config"].values == {"temperature": 0}
    assert result.text == "file api result"
    assert result.input_tokens == 19
    assert result.output_tokens == 8
    assert result.thinking_tokens == 2
    assert result.total_tokens == 29
    assert result.remote_file_id == "files/paper-123"
    assert result.cleanup_success is True
    assert result.model_version == "gemini-3.1-pro-preview-20260701"
    assert result.response_id == "response-file-7"


class FakeClaudeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_claude_vertex_client_sends_base64_pdf_document_and_maps_metadata(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-claude")
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="table part 1"),
            SimpleNamespace(type="text", text="table part 2"),
        ],
        usage=SimpleNamespace(input_tokens=101, output_tokens=31),
        model="claude-opus-5",
        id="msg_123",
    )
    messages = FakeClaudeMessages(response)
    client = ClaudeVertexClient(
        project="dataextractionllm-503420",
        location="global",
        client=SimpleNamespace(messages=messages),
    )

    result = client.generate(
        pdf,
        "frozen prompt",
        "claude-opus-5",
        {"max_tokens": 4096, "temperature": 0},
    )

    assert len(messages.calls) == 1
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["max_tokens"] == 4096
    assert call["temperature"] == 0
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "application/pdf"
    assert content[0]["source"]["data"]
    assert content[1] == {"type": "text", "text": "frozen prompt"}
    assert result.text == "table part 1\n\ntable part 2"
    assert result.input_tokens == 101
    assert result.output_tokens == 31
    assert result.total_tokens == 132
    assert result.model_version == "claude-opus-5"
    assert result.response_id == "msg_123"


def test_claude_api_client_sends_base64_pdf_document_and_maps_metadata(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-claude")
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="api result")],
        usage=SimpleNamespace(input_tokens=50, output_tokens=12),
        model="claude-sonnet-4-5",
        id="msg_api_123",
    )
    messages = FakeClaudeMessages(response)
    client = ClaudeAPIClient(
        api_key="test-key",
        client=SimpleNamespace(messages=messages),
    )

    result = client.generate(
        pdf,
        "frozen prompt",
        "claude-sonnet-4-5",
        {"max_tokens": 2048, "temperature": 0},
    )

    call = messages.calls[0]
    assert call["model"] == "claude-sonnet-4-5"
    assert call["max_tokens"] == 2048
    assert call["temperature"] == 0
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "application/pdf"
    assert content[1] == {"type": "text", "text": "frozen prompt"}
    assert result.text == "api result"
    assert result.input_tokens == 50
    assert result.output_tokens == 12
    assert result.total_tokens == 62
    assert result.model_version == "claude-sonnet-4-5"
    assert result.response_id == "msg_api_123"


def test_build_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        build_client({"provider": "unknown", "model_id": "model"})


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {
                "provider": "vertex",
                "location": "global",
                "model_id": "gemini-3.1-pro-preview",
            },
            "project",
        ),
        (
            {
                "provider": "vertex",
                "project": "dataextractionllm-503420",
                "model_id": "gemini-3.1-pro-preview",
            },
            "location",
        ),
        (
            {
                "provider": "claude_vertex",
                "location": "global",
                "model_id": "claude-opus-5",
            },
            "project",
        ),
        (
            {
                "provider": "claude_vertex",
                "project": "dataextractionllm-503420",
                "model_id": "claude-opus-5",
            },
            "location",
        ),
    ],
)
def test_build_client_requires_vertex_identity(config, message):
    with pytest.raises(ValueError, match=message):
        build_client(config)


def test_build_client_requires_explicit_provider():
    with pytest.raises(ValueError, match="provider"):
        build_client({"model_id": "gemini-3.1-pro-preview"})


def test_build_client_requires_anthropic_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_client({"provider": "claude_api", "model_id": "claude-sonnet-4-5"})
