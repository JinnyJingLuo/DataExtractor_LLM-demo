from __future__ import annotations

import base64
import os
from pathlib import Path

from .api_runner import ClientResponse


class GoogleGenerativeAIClient:
    def __init__(self, api_key: str | None = None, sdk=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.sdk = sdk

    def generate(self, pdf_path, prompt, model_id, generation_config):
        if self.sdk is None:
            import google.generativeai as genai
        else:
            genai = self.sdk

        genai.configure(api_key=self.api_key)
        uploaded = genai.upload_file(path=str(pdf_path), mime_type="application/pdf")
        response = None
        try:
            model = genai.GenerativeModel(
                model_id,
                generation_config=generation_config or None,
            )
            response = model.generate_content([prompt, uploaded])
        finally:
            try:
                genai.delete_file(uploaded.name)
                cleanup_success = True
            except Exception:
                cleanup_success = False
        usage = getattr(response, "usage_metadata", None)
        return ClientResponse(
            text=response.text or "",
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            thinking_tokens=getattr(usage, "thoughts_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
            remote_file_id=uploaded.name,
            cleanup_success=cleanup_success,
            model_version=getattr(response, "model_version", None),
            response_id=getattr(response, "response_id", None),
        )


class GeminiFileAPIClient:
    def __init__(self, api_key: str | None = None, client=None, types_module=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")

        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            types_module = types
        elif types_module is None:
            from google.genai import types

            types_module = types

        self.client = client
        self.types = types_module

    def generate(self, pdf_path, prompt, model_id, generation_config):
        uploaded = self.client.files.upload(file=str(pdf_path))
        response = None
        try:
            config = (
                self.types.GenerateContentConfig(**generation_config)
                if generation_config
                else None
            )
            response = self.client.models.generate_content(
                model=model_id,
                contents=[uploaded, prompt],
                config=config,
            )
        finally:
            try:
                self.client.files.delete(name=uploaded.name)
                cleanup_success = True
            except Exception:
                cleanup_success = False
        usage = getattr(response, "usage_metadata", None)
        return ClientResponse(
            text=getattr(response, "text", "") or "",
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            thinking_tokens=getattr(usage, "thoughts_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
            remote_file_id=getattr(uploaded, "name", None),
            cleanup_success=cleanup_success,
            model_version=getattr(response, "model_version", None),
            response_id=getattr(response, "response_id", None),
        )


class VertexAIClient:
    def __init__(
        self,
        project: str,
        location: str,
        client=None,
        types_module=None,
    ):
        if not project:
            raise ValueError("Vertex provider requires project")
        if not location:
            raise ValueError("Vertex provider requires location")

        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=types.HttpOptions(api_version="v1"),
            )
            types_module = types
        elif types_module is None:
            from google.genai import types

            types_module = types

        self.client = client
        self.types = types_module

    def generate(self, pdf_path, prompt, model_id, generation_config):
        contents = [
            self.types.Part.from_text(text=prompt),
            self.types.Part.from_bytes(
                data=Path(pdf_path).read_bytes(),
                mime_type="application/pdf",
            ),
        ]
        config = (
            self.types.GenerateContentConfig(**generation_config)
            if generation_config
            else None
        )
        response = self.client.models.generate_content(
            model=model_id,
            contents=contents,
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        return ClientResponse(
            text=getattr(response, "text", "") or "",
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            thinking_tokens=getattr(usage, "thoughts_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
            model_version=getattr(response, "model_version", None),
            response_id=getattr(response, "response_id", None),
        )


def _claude_pdf_message(pdf_path, prompt):
    pdf_data = base64.b64encode(Path(pdf_path).read_bytes()).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _claude_response(response) -> ClientResponse:
    text_parts = [
        getattr(block, "text", "")
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
    ]
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return ClientResponse(
        text="\n\n".join(part for part in text_parts if part),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        model_version=getattr(response, "model", None),
        response_id=getattr(response, "id", None),
    )


class ClaudeAPIClient:
    def __init__(
        self,
        api_key: str | None = None,
        client=None,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")

        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key)

        self.client = client

    def generate(self, pdf_path, prompt, model_id, generation_config):
        config = dict(generation_config or {})
        max_tokens = int(config.pop("max_tokens", 8192))
        response = self.client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=_claude_pdf_message(pdf_path, prompt),
            **config,
        )
        return _claude_response(response)


class ClaudeVertexClient:
    def __init__(
        self,
        project: str,
        location: str,
        client=None,
    ):
        if not project:
            raise ValueError("Claude Vertex provider requires project")
        if not location:
            raise ValueError("Claude Vertex provider requires location")

        if client is None:
            from anthropic import AnthropicVertex

            client = AnthropicVertex(project_id=project, region=location)

        self.client = client

    def generate(self, pdf_path, prompt, model_id, generation_config):
        config = dict(generation_config or {})
        max_tokens = int(config.pop("max_tokens", 8192))
        response = self.client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=_claude_pdf_message(pdf_path, prompt),
            **config,
        )
        return _claude_response(response)


def build_client(config: dict):
    provider = config.get("provider")
    if not provider:
        raise ValueError("model config requires provider")
    if provider == "gemini_api":
        return GoogleGenerativeAIClient()
    if provider == "gemini_file_api":
        return GeminiFileAPIClient()
    if provider == "claude_api":
        return ClaudeAPIClient()
    if provider == "vertex":
        project = config.get("project")
        location = config.get("location")
        if not project:
            raise ValueError("Vertex provider requires project")
        if not location:
            raise ValueError("Vertex provider requires location")
        return VertexAIClient(project=project, location=location)
    if provider == "claude_vertex":
        project = config.get("project")
        location = config.get("location")
        if not project:
            raise ValueError("Claude Vertex provider requires project")
        if not location:
            raise ValueError("Claude Vertex provider requires location")
        return ClaudeVertexClient(project=project, location=location)
    raise ValueError(f"Unsupported provider: {provider}")
