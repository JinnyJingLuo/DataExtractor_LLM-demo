# API and Held-Out Evaluation Pipeline Design

## Objective

Build an independent, reproducible Gemini API evaluation pipeline under
`rebuttal_heldout_eval/heldout_pipeline` that supports the three experiments
required for the paper revision:

1. API development-set results on the original 30 papers.
2. Corrected evaluation of API outputs.
3. Strict held-out evaluation on additional unseen papers.

The original 30-paper chatbox results remain the historical development
baseline. They are not combined with API results or held-out results.

## Fixed Experimental Decisions

- The received `prompty.md` is used without textual modification.
- The prompt is copied into the new pipeline and frozen by SHA-256.
- Original papers are labeled `development`.
- Additional papers are labeled `heldout`.
- Held-out PDFs use arbitrary filenames and are registered through a manifest.
- Ground truth continues to use the existing Excel layout, with new papers
  added as additional `PaperN` rows.
- API development and API held-out metrics are reported separately.
- The corrected evaluator replaces the received evaluator for primary results.
- The received evaluator may be run only to produce a clearly labeled
  `legacy_accuracy` comparison.

## Approaches Considered

### Patch the received scripts

This minimizes file count but retains brittle parsing, incomplete logs, and a
scoring design that cannot correctly represent false extractions or missing
columns.

### Independent held-out pipeline

This preserves the received artifact for audit while introducing explicit
interfaces for paper registration, API runs, normalized records, provenance,
and corrected metrics. This is the selected approach.

### Rewrite extraction with a new structured-output prompt

This could simplify parsing, but it would change the evaluated extraction
method and invalidate the decision to freeze the received prompt unchanged.

## Directory Structure

```text
rebuttal_heldout_eval/
  heldout_pipeline/
    README.md
    pyproject.toml
    configs/
      evaluation.json
      models.json
    prompts/
      prompty_frozen.md
      prompt_manifest.json
    manifests/
      papers.csv
      papers.example.csv
    src/
      heldout_pipeline/
        __init__.py
        cli.py
        config.py
        manifest.py
        api_runner.py
        response_parser.py
        normalize.py
        provenance.py
        evaluator.py
        metrics.py
        reporting.py
        legacy.py
    tests/
      test_manifest.py
      test_response_parser.py
      test_normalize.py
      test_provenance.py
      test_evaluator.py
      test_metrics.py
      test_reporting.py
    artifacts/
      development/
      heldout/
    results/
      development/
      heldout/
      comparison/
```

The received `Gemini_Pipeline` directory remains unchanged.

## Paper Manifest

`manifests/papers.csv` is the source of truth for experiment membership:

```text
paper_id,pdf_path,split,include,selection_note
Paper1,../../gemini_pipeline_received/Gemini_Pipeline/papers/Paper1.pdf,development,true,Original development paper
Heldout001,D:/heldout/example.pdf,heldout,true,Selected before prompt freeze
```

Rules:

- `paper_id` must be unique.
- `split` must be `development` or `heldout`.
- Every included PDF must exist.
- Held-out records require a non-empty `selection_note`.
- A paper cannot occur in both splits.
- Arbitrary PDF filenames are supported.
- Metrics are always grouped by manifest split.

## Prompt Freeze

`prompty_frozen.md` is an exact byte-for-byte copy of the received
`prompty.md`. `prompt_manifest.json` records:

- Prompt filename.
- SHA-256.
- Source path.
- Freeze timestamp.
- Git commit.
- A statement that the text was not changed for the held-out experiment.

Every API run verifies the prompt hash before sending a request. A mismatch
stops the run.

## API Experiment

The API runner processes one paper per independent request. It uses the model
identifier specified in `configs/models.json`, initially
`gemini-2.5-pro`, matching the received runner.

For each paper, the runner stores:

```text
artifacts/<split>/<paper_id>/
  request.json
  raw_response.md
  response_metadata.json
  extracted_data.csv
  evidence_source.csv
  parse_report.json
```

`request.json` records paper ID, split, PDF SHA-256, prompt SHA-256, model ID,
generation configuration, attempt number, and timestamps.

`response_metadata.json` records:

- Success or failure.
- Input, output, and total tokens when returned by the API.
- Wall-clock duration.
- Number of attempts and retries.
- Error type and message.
- Uploaded-file identifier and cleanup result.
- Raw-response SHA-256.

Raw response text is saved before parsing. A parse failure therefore does not
destroy the model output.

The CLI supports resumable runs. A successful existing artifact is skipped
unless `--force` is supplied. Failed papers can be selected with
`--retry-failures`.

## Response Parsing

The parser expects the two Markdown tables required by the frozen prompt:

- Table 1: extracted records.
- Table 2: evidence source.

Parsing behavior:

- Preserve the raw response.
- Validate the exact 37-column Table 1 schema.
- Reject duplicate column names.
- Report missing and unexpected columns.
- Preserve all extracted rows, including extra rows that may later become
  false extractions.
- Preserve evidence rows even when their field name cannot be mapped.
- Write a machine-readable parse report.
- Never silently truncate cells or rows to force a schema match.

## Ground-Truth Input

The evaluator reads the existing `Ground_truth Table.xlsx` format:

- Header row is the second Excel row.
- `sheet` identifies the paper.
- `Sample ID` identifies the experimental record.
- All remaining columns are fields.

Held-out ground-truth rows are added to the same workbook using paper IDs that
match the manifest. The evaluator converts both ground truth and predictions
to long form with the key:

```text
paper_id + sample_id + field_name
```

Duplicate normalized keys are fatal. The evaluator does not silently choose
one row.

## Provenance Review

The frozen prompt's evidence table is paper-field scoped and may not
distinguish different samples within the same paper. The new pipeline creates
`provenance_review.csv` with one row per prediction key:

```text
paper_id,sample_id,field_name,tier,provenance,evidence_location,evidence_text,review_status
```

Allowed tiers:

- `T1`
- `T2`
- `T3`
- `NA`

Allowed provenance:

- `paper_text`
- `paper_table`
- `paper_equation`
- `paper_figure`
- `derived_from_paper`
- `external_reference`
- `not_available`
- `unknown`

Missing or unrecognized evidence is `unknown`; it never defaults to T1.
Primary tier metrics require `review_status=approved`. Unapproved rows remain
in overall field metrics but are reported as unknown provenance and excluded
from tier-specific claims.

Room temperature, standard density, and standard melting point values inserted
because the paper did not report them are labeled `external_reference`.
External references are excluded from primary extraction metrics and reported
in a separate provenance table.

## Corrected Evaluation

Ground truth and predictions are outer-joined on the complete key. Every key
receives exactly one outcome:

- `correct_value`: non-null ground truth and matching non-null prediction.
- `wrong_value`: non-null ground truth and non-matching non-null prediction.
- `missing_extraction`: non-null ground truth and null or absent prediction.
- `false_extraction`: null or absent ground truth and non-null prediction.
- `correct_null`: null ground truth and null prediction.

An extracted row with no matching ground-truth sample generates
`false_extraction` outcomes for each non-null field. A ground-truth field with
an absent prediction column generates `missing_extraction`.

Categorical fields use exact equality after trimming, case folding, and
collapsing whitespace. No fuzzy or semantic matching is used.

Numeric fields match when:

```text
absolute_error <= absolute_tolerance
or
absolute_error / max(abs(ground_truth), numeric_floor) <= relative_tolerance
```

The default relative tolerance remains `0.005`, matching the paper. Default
absolute tolerance and field-specific overrides are stored in
`configs/evaluation.json`.

Primary metrics:

```text
precision = correct_value /
            (correct_value + wrong_value + false_extraction)

recall = correct_value /
         (correct_value + wrong_value + missing_extraction)

f1 = 2 * precision * recall / (precision + recall)

accuracy_all = (correct_value + correct_null) / all_outcomes

accuracy_non_null = correct_value /
                    (correct_value + wrong_value + missing_extraction)
```

Zero denominators return `0.0` with support counts.

Metrics are written:

- Overall by split.
- By paper.
- By field.
- By approved tier.
- By provenance.
- By model and prompt hash.
- As an error breakdown with row-level outcomes.

## Legacy Evaluation

The received evaluator is not used for primary conclusions because controlled
tests confirm that it can:

- Score an absent extracted column against the ground-truth column itself.
- Ignore extra extracted rows.
- Default missing provenance to T1.

The new pipeline may reproduce its aggregate result under the label
`legacy_accuracy` solely to explain differences from the submitted paper. The
report must place legacy and corrected results in separate columns.

## Paper Outputs

The reporting command creates:

```text
results/development/api_run_summary.csv
results/development/corrected_metrics.csv
results/development/error_breakdown.csv
results/heldout/api_run_summary.csv
results/heldout/corrected_metrics.csv
results/heldout/error_breakdown.csv
results/comparison/experiment_overview.csv
results/comparison/legacy_vs_corrected.csv
results/comparison/tier_metrics.csv
results/comparison/field_metrics.csv
results/comparison/cost_runtime.csv
```

`experiment_overview.csv` contains three separate rows:

1. Original chatbox development evaluation.
2. API development evaluation.
3. API held-out evaluation.

No combined development-plus-held-out headline metric is produced.

## CLI

Commands:

```powershell
python -m heldout_pipeline.cli freeze-prompt
python -m heldout_pipeline.cli validate-manifest --manifest manifests/papers.csv
python -m heldout_pipeline.cli run --split development
python -m heldout_pipeline.cli run --split heldout
python -m heldout_pipeline.cli prepare-provenance --split heldout
python -m heldout_pipeline.cli evaluate --split development --ground-truth "Ground_truth Table.xlsx"
python -m heldout_pipeline.cli evaluate --split heldout --ground-truth "Ground_truth Table.xlsx"
python -m heldout_pipeline.cli report
```

The API commands require `GEMINI_API_KEY`. Evaluation and report commands do
not require network access.

## Testing Strategy

Implementation is test-first. Tests cover:

- Manifest validation and arbitrary PDF paths.
- Prompt-hash mismatch rejection.
- Raw-response persistence before parsing.
- Two-table parsing and schema rejection.
- Ground-truth Excel conversion.
- Duplicate paper/sample/field keys.
- All five corrected outcomes.
- Missing extracted columns.
- Extra extracted samples and fields.
- Null normalization.
- Numeric tolerance and categorical exact match.
- No fuzzy matching for semantic variants.
- Unknown provenance rather than T1 default.
- External-reference exclusion.
- Development/held-out separation.
- Precision, recall, F1, all-field accuracy, and non-null accuracy.
- Run summaries with success, failure, token, time, retry, and cost fields.

Tests use synthetic fixtures and do not call the Gemini API.

## Error Handling

- Invalid manifests, duplicate keys, missing PDFs, prompt hash changes, and
  malformed ground truth stop execution with a nonzero exit code.
- API failures are recorded per paper and do not delete successful artifacts.
- Parsing failures preserve raw responses and create parse reports.
- Missing token or price data is reported as unavailable, never as zero.
- Held-out evaluation refuses to run if the selected paper IDs are absent from
  the manifest or are not labeled `heldout`.

## Acceptance Criteria

The implementation is complete when:

- The received prompt is frozen unchanged and hash-verified.
- Original and held-out papers are registered through one manifest.
- API runs preserve raw outputs and complete run metadata.
- The corrected evaluator passes regression tests for every confirmed legacy
  scoring defect.
- Development and held-out metrics are generated separately.
- Precision, recall, F1, all-field accuracy, non-null accuracy, tier metrics,
  provenance metrics, and error categories are auditable from row-level files.
- External reference values are separated from paper extraction.
- The README provides exact commands for adding arbitrary held-out PDFs,
  extending the existing ground-truth workbook, running the API, reviewing
  provenance, evaluating, and generating manuscript tables.
