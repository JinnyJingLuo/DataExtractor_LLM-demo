# DataExtractor LLM: API and Held-Out Evaluation

This repository implements three separate experiments for the paper revision:

1. **Original chatbox development evaluation**: the historical 30-paper result
   reported in the submission.
2. **API development evaluation**: the same 30 development papers processed
   through reproducible, independent Gemini API requests.
3. **API held-out evaluation**: additional papers that were never used for
   prompt writing, examples, debugging, or error analysis.

The extraction prompt is frozen unchanged in `prompts/prompty_frozen.md`.
Development and held-out results are never merged into one headline metric.

## What This Corrects

The received evaluator could score a missing prediction column against the
ground-truth value itself, ignored extra prediction rows, and defaulted missing
provenance to T1. This implementation instead:

- outer-joins `paper_id + sample_id + field_name`;
- counts extra non-null predictions as false extractions;
- counts absent predictions as missing extractions;
- rejects duplicate normalized keys;
- uses `NA/unknown` rather than defaulting provenance to T1;
- separates external reference values from paper extraction;
- reports precision, recall, F1, all-field accuracy, non-null accuracy, and all
  five field outcomes.

## WSL Setup

```bash
git clone https://github.com/JinnyJingLuo/DataExtractor_LLM.git
cd DataExtractor_LLM
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

### Vertex AI authentication (default)

The default `configs/models.json` uses Vertex AI with ADC:

```bash
gcloud auth application-default login --no-launch-browser
gcloud auth application-default set-quota-project dataextractionllm-503420
```

The active experiment requests `gemini-3.1-pro-preview` from project
`dataextractionllm-503420` through the `global` endpoint. ADC credentials and
access tokens are never copied into run artifacts.

### Claude on Vertex AI

Claude can be run through the same Google Cloud project using
`configs/models.claude_vertex.example.json`. Enable the intended Claude model
in Vertex AI Model Garden first, then keep using ADC:

```bash
gcloud auth application-default login --no-launch-browser
gcloud auth application-default set-quota-project dataextractionllm-503420
gcloud config set project dataextractionllm-503420
```

The Claude Vertex backend sends each PDF as a base64 `application/pdf`
document in a single model request. Use a distinct artifact root from Gemini.

### Claude direct API authentication

If Vertex Claude quota is unavailable, the direct Anthropic backend can be used
with `configs/models.claude_api.example.json`:

```bash
export ANTHROPIC_API_KEY="..."
```

Do not commit the key. The direct Claude API backend sends the same base64
`application/pdf` document payload as the Vertex Claude backend.

### Gemini Developer API authentication

The Developer API backend remains available through
`configs/models.gemini_api.example.json`. Set its key only in the environment:

```bash
export GEMINI_API_KEY="..."
```

Do not commit the key.

## 1. Register Papers

Copy the example:

```bash
cp manifests/papers.example.csv manifests/papers.csv
```

Use absolute WSL paths. PDF filenames may be arbitrary.

```csv
paper_id,pdf_path,split,include,selection_note
Paper1,/mnt/c/path/to/Paper1.pdf,development,true,Original development paper
Heldout001,/mnt/c/path/to/unseen-alloy-study.pdf,heldout,true,Selected before prompt freeze and never used for prompt development
```

Rules:

- `paper_id` is unique and must match the `sheet` value in ground truth.
- Original papers use `development`.
- New unseen papers use `heldout`.
- Every held-out paper requires a selection note.
- Do not inspect held-out predictions before ground truth is completed.

Validate:

```bash
heldout-pipeline validate-manifest --manifest manifests/papers.csv
```

## 2. Verify the Frozen Prompt

```bash
heldout-pipeline freeze-prompt \
  --check \
  --prompt prompts/prompty_frozen.md \
  --prompt-manifest prompts/prompt_manifest.json
```

Any prompt change causes the API run to stop with a hash mismatch.

## 3. Prepare Ground Truth

`data/Ground_truth Table.xlsx` is the received 30-paper ground-truth workbook.
It uses the existing format:

- Excel row 2 contains headers.
- `sheet` is the paper ID.
- `Sample ID` is the experimental record ID.
- Other columns are scored fields.

For a fully manual held-out ground truth, complete this before viewing held-out
predictions:

1. Add rows for each held-out paper.
2. Set `sheet` to the corresponding manifest `paper_id`.
3. Complete and adjudicate the labels.
4. Save a versioned copy of the workbook and record its SHA-256.

For the Gemini + Claude consensus workflow, first generate the candidate table
in Section 5, then manually fill the review items before using it as final
held-out ground truth.

## 4. Run the API

Vertex development:

```bash
heldout-pipeline run \
  --manifest manifests/papers.csv \
  --split development \
  --model-config configs/models.json
```

Vertex held-out:

```bash
heldout-pipeline run \
  --manifest manifests/papers.csv \
  --split heldout \
  --model-config configs/models.json
```

Claude Vertex held-out:

```bash
heldout-pipeline run \
  --manifest manifests/papers.csv \
  --split heldout \
  --model-config configs/models.claude_vertex.example.json \
  --artifact-root artifacts/claude_vertex_heldout
```

Claude direct API held-out:

```bash
heldout-pipeline run \
  --manifest manifests/papers.csv \
  --split heldout \
  --model-config configs/models.claude_api.example.json \
  --artifact-root artifacts/claude_api_heldout
```

Developer API development:

```bash
heldout-pipeline run \
  --manifest manifests/papers.csv \
  --split development \
  --model-config configs/models.gemini_api.example.json \
  --artifact-root artifacts/gemini_api
```

Use a distinct `--artifact-root` for each provider when retaining results from
both backends. A run is resumable only when provider, project, location, model,
generation configuration, prompt hash, and PDF hash all match the saved
metadata.

Each paper is sent in an independent API request. Artifacts are written to:

```text
artifacts/<split>/<paper_id>/
  request.json
  raw_response.md
  response_metadata.json
  extracted_data.csv
  evidence_source.csv
  parse_report.json
```

The metadata records provider, requested model ID, service-resolved model
version, response ID, project and location, PDF and prompt hashes, input,
output, thinking and total tokens, duration, attempts, retries, errors,
cleanup status, and cost when prices are configured. It never records
credentials.

An empty `generation_config` means provider defaults, matching the submitted
paper's API description. A completed model response is the paper's only
accepted response. Retries occur only after a failed attempt; the runner never
generates multiple successful responses and selects the best one. Successful
papers are skipped on rerun unless `--force` is supplied.

The original Gemini 3 Pro Preview endpoint was retired on March 9, 2026. New
Vertex development and held-out runs therefore use its official successor,
Gemini 3.1 Pro Preview, and must be labeled separately from historical chatbox
results.

## 5. Build Gemini + Claude Held-Out Candidate Ground Truth

Run Gemini and Claude into separate artifact roots before looking at the
combined candidate table:

```bash
heldout-pipeline run \
  --manifest manifests/papers.csv \
  --split heldout \
  --model-config configs/models.json \
  --artifact-root artifacts/gemini_vertex_heldout

heldout-pipeline run \
  --manifest manifests/papers.csv \
  --split heldout \
  --model-config configs/models.claude_vertex.example.json \
  --artifact-root artifacts/claude_vertex_heldout
```

If Vertex Claude quota is unavailable, use direct Claude API instead:

```bash
heldout-pipeline run \
  --manifest manifests/papers.csv \
  --split heldout \
  --model-config configs/models.claude_api.example.json \
  --artifact-root artifacts/claude_api_heldout
```

Then compare them:

```bash
heldout-pipeline compare-consensus \
  --manifest manifests/papers.csv \
  --split heldout \
  --gemini-artifact-root artifacts/gemini_vertex_heldout \
  --claude-artifact-root artifacts/claude_api_heldout \
  --output-dir artifacts/heldout_consensus
```

Outputs:

- `consensus_candidates.csv`: field-level Gemini/Claude agreements.
- `candidate_ground_truth.xlsx`: partial held-out ground-truth workbook with
  agreed fields filled and disagreements left blank.
- `disagreement_review.csv`: fields that require human checking.
- `record_review.csv`: unmatched or ambiguous experimental records.

Agreement uses exact normalized Sample ID first, then unique experimental
anchor fields. Numerical fields use the evaluator tolerance from
`configs/evaluation.json`; text fields use only mechanical normalization
(case, whitespace, punctuation), so semantic differences such as `Annealing`
versus `Annealed` remain review items.

## 6. Review Provenance

Generate a row for every predicted sample-field:

```bash
heldout-pipeline prepare-provenance \
  --manifest manifests/papers.csv \
  --split heldout \
  --artifact-root artifacts/gemini_vertex_heldout \
  --output artifacts/gemini_vertex_heldout/heldout/provenance_review.csv
```

Review `tier`, `provenance`, evidence, and source location. Set
`review_status=approved` only after verification. Missing evidence remains
`tier=NA, provenance=unknown`; it is never silently treated as T1.

Standard or assumed room temperature, density, and melting point are labeled
`external_reference`. A directly reported value in one of these fields remains
a paper extraction when its evidence is explicit.

Repeat for development:

```bash
heldout-pipeline prepare-provenance \
  --manifest manifests/papers.csv \
  --split development \
  --output artifacts/development/provenance_review.csv
```

For the separate Developer API root:

```bash
heldout-pipeline prepare-provenance \
  --manifest manifests/papers.csv \
  --split development \
  --artifact-root artifacts/gemini_api \
  --output artifacts/gemini_api/development/provenance_review.csv
```

## 7. Run Corrected Evaluation

Development:

```bash
heldout-pipeline evaluate \
  --manifest manifests/papers.csv \
  --split development \
  --ground-truth "data/Ground_truth Table.xlsx" \
  --provenance artifacts/development/provenance_review.csv \
  --output-dir results/development
```

Held-out:

```bash
heldout-pipeline evaluate \
  --manifest manifests/papers.csv \
  --split heldout \
  --ground-truth "data/Heldout_ground_truth_reviewed.xlsx" \
  --artifact-root artifacts/gemini_vertex_heldout \
  --provenance artifacts/gemini_vertex_heldout/heldout/provenance_review.csv \
  --output-dir results/heldout
```

Developer API development:

```bash
heldout-pipeline evaluate \
  --manifest manifests/papers.csv \
  --split development \
  --ground-truth "data/Ground_truth Table.xlsx" \
  --artifact-root artifacts/gemini_api \
  --provenance artifacts/gemini_api/development/provenance_review.csv \
  --output-dir results/gemini_api/development
```

Each result directory contains:

- `field_outcomes.csv`
- `overall_metrics.csv`
- `paper_metrics.csv`
- `field_metrics.csv`
- `field_type_metrics.csv`
- `tier_metrics.csv`
- `provenance_metrics.csv`
- `error_breakdown.csv`

Tier metrics include only provenance rows explicitly marked `approved`.
External reference rows remain auditable but are excluded from primary metrics.

## 8. Generate Paper Tables

```bash
heldout-pipeline report \
  --artifact-root artifacts \
  --output-dir results/comparison
```

For Developer API artifacts:

```bash
heldout-pipeline report \
  --artifact-root artifacts/gemini_api \
  --output-dir results/gemini_api/comparison
```

This produces:

- `cost_runtime.csv`
- `experiment_overview.csv`

The overview keeps chatbox development, API development, and API held-out
experiments in separate rows.

## Metric Definitions

```text
precision = correct_value /
            (correct_value + wrong_value + false_extraction)

recall = correct_value /
         (correct_value + wrong_value + missing_extraction)

f1 = 2 * precision * recall / (precision + recall)

accuracy_all = sum(graded cell scores) / all_scored_fields

accuracy_non_null = sum(graded scores over non-null ground-truth fields) /
                    (correct_value + wrong_value + missing_extraction)

strict_accuracy_all = (correct_value + correct_null) / all_scored_fields

strict_accuracy_non_null = correct_value /
                           (correct_value + wrong_value + missing_extraction)
```

Categorical fields use exact equality after case and whitespace normalization.
No fuzzy or semantic matching is used. Numeric fields use the project graded
rubric: 1.00 for exact or <0.5% relative difference, 0.95 for <2%, 0.90 for
format-only numeric differences, 0.80 for <5%, 0.60 for <10%, 0.30 for <20%,
and 0.00 for missing or >20%. The strict columns preserve the previous binary
accuracy under the 0.5% tolerance.
`field_type_metrics.csv` separates numeric and categorical fields so numeric-only
accuracy can be reported without relabeling it as the overall extraction
accuracy.

## Rebuttal Reporting Rule

Do not describe the code framework as an experimental result. Report API
accuracy, held-out accuracy, tokens, runtime, retries, failures, and cost only
after the corresponding artifacts have been generated and audited.
