# Vertex Dual Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the Gemini Developer API backend and add a reproducible Vertex AI ADC backend using `gemini-3.1-pro-preview`.

**Architecture:** Keep extraction orchestration in `api_runner.py`, move provider-specific clients into a focused `clients.py`, and select the client from an explicit model configuration. Extend request/response metadata with provider and resolved-model provenance while keeping parsing and evaluation interfaces unchanged.

**Tech Stack:** Python 3.12, `google-generativeai`, `google-genai`, pandas, pytest, Google Vertex AI ADC.

---

## File Map

- Create `src/heldout_pipeline/clients.py`: Developer API and Vertex AI client implementations plus provider factory.
- Modify `src/heldout_pipeline/api_runner.py`: shared response model, provider provenance, and run metadata.
- Modify `src/heldout_pipeline/cli.py`: validated provider selection and client construction.
- Modify `configs/models.json`: active Vertex AI experiment configuration.
- Create `configs/models.gemini_api.example.json`: preserved Developer API configuration.
- Modify `pyproject.toml`: add the supported Google Gen AI SDK.
- Modify `README.md`: document both authentication modes and Vertex execution.
- Modify `tests/test_api_runner.py`: response provenance and artifact metadata tests.
- Create `tests/test_clients.py`: provider factory and Vertex request/response mapping tests.
- Modify `tests/test_cli.py`: CLI model configuration and client-selection tests.

### Task 1: Extend Run Metadata

**Files:**
- Modify: `src/heldout_pipeline/api_runner.py`
- Test: `tests/test_api_runner.py`

- [ ] **Step 1: Write failing response-provenance tests**

Add assertions that `ClientResponse` accepts `thinking_tokens`,
`model_version`, and `response_id`, and that `run_paper` persists provider,
project, location, and these returned fields:

```python
def test_saves_provider_and_resolved_model_metadata(
    tmp_path, monkeypatch
):
    prompt, prompt_manifest = setup_prompt(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    record = PaperRecord("D1", pdf, "development", True, "old")
    configured = settings(tmp_path, prompt, prompt_manifest)
    configured = replace(
        configured,
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
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest tests/test_api_runner.py::test_saves_provider_and_resolved_model_metadata -q
```

Expected: FAIL because `RunSettings` and `ClientResponse` do not yet expose
the new fields.

- [ ] **Step 3: Add minimal metadata fields**

Extend the dataclasses:

```python
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
```

Add the provider fields to `request.json` and the returned response fields to
successful `response_metadata.json`. Add null values for the response-only
fields to failed metadata.

- [ ] **Step 4: Run runner tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_api_runner.py -q
```

Expected: all runner tests PASS.

- [ ] **Step 5: Commit metadata support**

```bash
git add src/heldout_pipeline/api_runner.py tests/test_api_runner.py
git commit -m "feat: record provider and model provenance"
```

### Task 2: Add the Vertex AI Client

**Files:**
- Create: `src/heldout_pipeline/clients.py`
- Modify: `src/heldout_pipeline/api_runner.py`
- Create: `tests/test_clients.py`
- Modify: `tests/test_api_runner.py`

- [ ] **Step 1: Write failing Vertex request and response tests**

Build lightweight fake `types` and model-service objects, then specify the
desired interface:

```python
def test_vertex_client_sends_one_pdf_request_and_maps_metadata(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")
    service = FakeModelService(
        FakeVertexResponse(
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
    )
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
    assert result.thinking_tokens == 3
    assert result.model_version == "gemini-3.1-pro-preview-20260701"
    assert result.response_id == "response-7"
```

Add provider-factory tests:

```python
def test_build_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        build_client({"provider": "unknown", "model_id": "model"})


def test_build_client_requires_vertex_project_and_location():
    with pytest.raises(ValueError, match="project"):
        build_client(
            {
                "provider": "vertex",
                "location": "global",
                "model_id": "gemini-3.1-pro-preview",
            }
        )
```

- [ ] **Step 2: Run client tests and verify RED**

Run:

```bash
python -m pytest tests/test_clients.py -q
```

Expected: collection FAIL because `heldout_pipeline.clients` does not exist.

- [ ] **Step 3: Move the Developer client and implement Vertex**

Create `clients.py`. It imports `ClientResponse` from `api_runner.py`;
`api_runner.py` does not import `clients.py`, which prevents a circular import.
Implement Vertex with:

```python
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
        if client is None or types_module is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=types.HttpOptions(api_version="v1"),
            )
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
```

Move `GoogleGenerativeAIClient` without behavior changes. Implement
`build_client(config)` that validates a required `provider` and returns the
corresponding concrete client. Update tests and CLI imports to load concrete
clients from `heldout_pipeline.clients`; keep only the shared protocol,
`ClientResponse`, run settings, and orchestration in `api_runner.py`.

- [ ] **Step 4: Run provider and runner tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_clients.py tests/test_api_runner.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 5: Commit provider implementations**

```bash
git add src/heldout_pipeline/clients.py src/heldout_pipeline/api_runner.py tests/test_clients.py tests/test_api_runner.py
git commit -m "feat: add Vertex AI ADC client"
```

### Task 3: Select Providers Through the CLI

**Files:**
- Modify: `src/heldout_pipeline/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `configs/models.json`
- Create: `configs/models.gemini_api.example.json`

- [ ] **Step 1: Write failing CLI provider tests**

Test `_run_command` with a monkeypatched `build_client` and `run_paper`:

```python
def test_run_command_passes_vertex_identity_to_runner(tmp_path, monkeypatch):
    model_config = tmp_path / "model.json"
    model_config.write_text(
        json.dumps(
            {
                "provider": "vertex",
                "project": "dataextractionllm-503420",
                "location": "global",
                "model_id": "gemini-3.1-pro-preview",
                "generation_config": {},
            }
        )
    )
    captured = {}

    def fake_build_client(model):
        captured["client_config"] = model
        return object()

    def fake_run_paper(record, settings, client):
        captured["settings"] = settings
        return RunResult(
            record.paper_id,
            record.split,
            True,
            True,
            tmp_path,
        )

    monkeypatch.setattr(cli, "build_client", fake_build_client)
    monkeypatch.setattr(cli, "run_paper", fake_run_paper)

    assert cli.main(
        [
            "run",
            "--manifest",
            str(manifest),
            "--split",
            "development",
            "--model-config",
            str(model_config),
        ]
    ) == 0
    assert captured["settings"].provider == "vertex"
    assert captured["settings"].project == "dataextractionllm-503420"
    assert captured["settings"].location == "global"
```

- [ ] **Step 2: Run the focused CLI test and verify RED**

Run:

```bash
python -m pytest tests/test_cli.py::test_run_command_passes_vertex_identity_to_runner -q
```

Expected: FAIL because the CLI still constructs the Developer API client
directly and does not populate provider identity.

- [ ] **Step 3: Wire the factory and update configurations**

Replace direct `GoogleGenerativeAIClient()` construction with:

```python
client = build_client(model)
```

Populate `RunSettings` with:

```python
provider=model["provider"],
project=model.get("project"),
location=model.get("location"),
```

Make `configs/models.json` the active Vertex configuration:

```json
{
  "provider": "vertex",
  "project": "dataextractionllm-503420",
  "location": "global",
  "model_id": "gemini-3.1-pro-preview",
  "max_attempts": 2,
  "retry_delay_seconds": 2.0,
  "inter_paper_delay_seconds": 2.5,
  "generation_config": {},
  "input_price_per_million_tokens_usd": null,
  "output_price_per_million_tokens_usd": null
}
```

Add `configs/models.gemini_api.example.json` with `provider` set to
`gemini_api` and the same non-secret operational fields.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: all CLI tests PASS.

- [ ] **Step 5: Commit CLI and configuration**

```bash
git add src/heldout_pipeline/cli.py tests/test_cli.py configs/models.json configs/models.gemini_api.example.json
git commit -m "feat: select Gemini provider from config"
```

### Task 4: Dependencies and Documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Add the supported SDK dependency**

Keep `google-generativeai` for the old backend and add:

```toml
"google-genai>=1.0,<2",
```

- [ ] **Step 2: Update the README with both authentication paths**

Document Vertex as the active experiment:

```bash
gcloud auth application-default login --no-launch-browser
gcloud auth application-default set-quota-project dataextractionllm-503420
heldout-pipeline run \
  --manifest manifests/papers.csv \
  --split development \
  --model-config configs/models.json
```

Keep the Developer API instructions with:

```bash
export GEMINI_API_KEY="..."
heldout-pipeline run \
  --manifest manifests/papers.csv \
  --split development \
  --model-config configs/models.gemini_api.example.json
```

Explain that empty `generation_config` means provider defaults, each paper has
one accepted successful response, and new Vertex results use Gemini 3.1 Pro
Preview rather than the retired original Gemini 3 Pro endpoint.

- [ ] **Step 3: Recreate the main worktree environment**

Run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Expected: installation completes with both Google SDKs present.

- [ ] **Step 4: Run dependency and full offline verification**

Run:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
```

Expected: no broken requirements, all tests PASS, and compileall exits zero.

- [ ] **Step 5: Commit dependencies and documentation**

```bash
git add pyproject.toml README.md
git commit -m "docs: document Vertex and Developer API runs"
```

### Task 5: Live Vertex Smoke Test and Final Verification

**Files:**
- No tracked file changes required.

- [ ] **Step 1: Verify ADC without printing credentials**

Run:

```bash
gcloud auth application-default print-access-token >/dev/null
```

Expected: exit code zero and no token printed.

- [ ] **Step 2: Send one minimal Vertex request**

Run a Python snippet with the installed Google Gen AI SDK:

```python
from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    project="dataextractionllm-503420",
    location="global",
    http_options=types.HttpOptions(api_version="v1"),
)
response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents="Reply with exactly: VERTEX_OK",
)
assert response.text.strip() == "VERTEX_OK"
print(response.model_version)
```

Expected: `VERTEX_OK` assertion passes and the service-reported model version
is printed. Do not save response content or credentials in the repository.

- [ ] **Step 3: Run final repository verification**

Run:

```bash
.venv/bin/python -m pytest -q
git status --short
git log --oneline -6
```

Expected: all tests PASS and the worktree contains no uncommitted files.

- [ ] **Step 4: Push main**

```bash
git push origin main
```

Expected: `origin/main` advances to the final documentation commit.
