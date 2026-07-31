# Dual Gemini Provider and Vertex AI Design

## Purpose

Extend the extraction runner to support both existing Gemini Developer API
authentication and Gemini on Vertex AI through Application Default Credentials
(ADC). New development and held-out experiments will use Gemini 3.1 Pro Preview
on Vertex AI while preserving the existing Developer API path.

## Experiment Identity

The active Vertex experiment is defined by:

- Provider: `vertex`
- Project: `dataextractionllm-503420`
- Location: `global`
- Requested model ID: `gemini-3.1-pro-preview`
- Prompt: the existing hash-verified frozen prompt
- Generation parameters: provider defaults (`generation_config` is empty)
- Sampling: one successful response per paper

The manuscript should identify the original chat and API experiments as Gemini
3.1 Pro only if their execution dates were on or after March 9, 2026. New
Vertex results must always be labeled separately from the original chatbox
results.

## Architecture

Keep the existing `GeminiClient` protocol and add a provider factory selected
from the model configuration.

Two concrete clients will be available:

1. `GoogleGenerativeAIClient` preserves the current Gemini Developer API
   implementation and reads `GEMINI_API_KEY`.
2. `VertexAIClient` uses the Google Gen AI SDK with `vertexai=True`, the
   configured project and location, and ADC from the local gcloud login.

The Vertex client sends the PDF and frozen prompt in a single independent
`generate_content` request. It does not use a chat session, cached
conversation, best-of selection, or cross-paper context.

## Configuration

The model configuration gains a required `provider` field:

```json
{
  "provider": "vertex",
  "project": "dataextractionllm-503420",
  "location": "global",
  "model_id": "gemini-3.1-pro-preview",
  "generation_config": {}
}
```

`provider` accepts only `vertex` or `gemini_api`. Vertex requires `project` and
`location`; Developer API continues to require `GEMINI_API_KEY`. Separate
example configuration files will document both modes, while the default
configuration will select Vertex for the new experiments.

## Artifacts and Provenance

Each request and response metadata file will record:

- provider
- project and location for Vertex
- requested model ID
- model version returned by the service, when available
- response ID, when available
- prompt and PDF hashes
- explicit generation configuration
- input, output, thinking, and total token counts when returned
- request start and finish timestamps
- duration, attempts, retries, and errors

The code will never write ADC credentials, access tokens, API keys, or local
credential-file contents into repository artifacts.

## Retry Semantics

A completed model response is accepted as the paper's sole response. Retries
are permitted only when an attempt fails before a usable response is returned,
for example because of transport errors, rate limiting, or server errors. The
runner never generates multiple successful candidates and selects the best
one.

## Compatibility

The existing `run`, parse, provenance, evaluation, and report commands remain
unchanged for users. Existing Developer API configuration continues to work.
Old artifacts remain readable; newly added metadata fields are optional when
summarizing earlier runs.

## Testing and Verification

Unit tests will cover:

- provider selection and configuration validation
- Vertex client construction with project, location, and ADC
- PDF and prompt request construction
- Vertex usage, model version, and response ID mapping
- preservation of the Developer API client
- provider metadata in saved artifacts
- rejection of unknown or incomplete provider configurations

After all offline tests pass, one minimal Vertex request will verify ADC,
project access, location, and model availability. Full extraction will not
start until the development PDF manifest is available and validated.

## Non-Goals

- Reproducing the retired Gemini 3 Pro model endpoint
- Changing the frozen extraction prompt
- Treating the original 30 development papers as held-out data
- Reporting framework tests as experimental results
