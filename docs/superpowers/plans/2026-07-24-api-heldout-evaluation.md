# API Held-Out Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a reproducible Gemini API pipeline that separately evaluates the original 30-paper development set and additional strict held-out papers with corrected information-extraction metrics.

**Architecture:** A Python package under `src/heldout_pipeline` owns manifest validation, prompt freezing, API artifact persistence, response parsing, normalization, provenance review, corrected scoring, and report generation. The received prompt is copied unchanged and hash-verified. API execution is isolated behind a client protocol so all non-network behavior is testable without a Gemini key.

**Tech Stack:** Python 3.10+, pandas, openpyxl, google-generativeai, pytest, standard-library argparse/dataclasses/json/csv/hashlib.

---

### Task 1: Project skeleton and frozen inputs

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `configs/evaluation.json`
- Create: `configs/models.json`
- Create: `prompts/prompty_frozen.md`
- Create: `prompts/prompt_manifest.json`
- Create: `manifests/papers.example.csv`
- Create: `src/heldout_pipeline/__init__.py`

- [x] **Step 1: Add packaging and test configuration**

Define a Python package with runtime dependencies on pandas, openpyxl, and
google-generativeai, plus pytest as a development dependency.

- [x] **Step 2: Copy the received prompt byte-for-byte**

Copy `prompty.md` from the received Gemini archive to
`prompts/prompty_frozen.md`. Record its SHA-256 and source path in
`prompt_manifest.json`.

- [x] **Step 3: Add explicit evaluation/model configuration**

Store the 0.5% numeric tolerance, null vocabulary, categorical fields,
provenance labels, external-reference exclusions, model ID, retry count, and
delay in JSON configuration.

- [x] **Step 4: Verify the package can be imported**

Run:

```bash
python3 -m pytest --collect-only
python3 -c "import heldout_pipeline"
```

Expected: package imports and pytest collection exits successfully.

### Task 2: Manifest validation

**Files:**
- Create: `src/heldout_pipeline/manifest.py`
- Create: `tests/test_manifest.py`

- [x] **Step 1: Write failing manifest tests**

Tests must cover arbitrary PDF filenames, duplicate IDs, invalid splits,
missing files, invalid booleans, and missing held-out selection notes.

- [x] **Step 2: Run the tests and verify RED**

```bash
python3 -m pytest tests/test_manifest.py -v
```

Expected: import or missing-function failures.

- [x] **Step 3: Implement manifest loading and validation**

Expose:

```python
def load_manifest(path: Path) -> list[PaperRecord]: ...
def select_papers(records: Iterable[PaperRecord], split: str) -> list[PaperRecord]: ...
```

`PaperRecord` contains `paper_id`, `pdf_path`, `split`, `include`, and
`selection_note`.

- [x] **Step 4: Run the tests and verify GREEN**

```bash
python3 -m pytest tests/test_manifest.py -v
```

Expected: all manifest tests pass.

### Task 3: Prompt freeze and API artifact runner

**Files:**
- Create: `src/heldout_pipeline/config.py`
- Create: `src/heldout_pipeline/api_runner.py`
- Create: `tests/test_config.py`
- Create: `tests/test_api_runner.py`

- [x] **Step 1: Write failing hash and artifact tests**

Tests must prove that a prompt hash mismatch stops a run, raw response text is
written before parsing, token/runtime/retry metadata is persisted, successful
runs are resumable, and failed attempts retain error metadata.

- [x] **Step 2: Run targeted tests and verify RED**

```bash
python3 -m pytest tests/test_config.py tests/test_api_runner.py -v
```

- [x] **Step 3: Implement configuration and API client boundary**

Expose:

```python
def verify_prompt(prompt_path: Path, manifest_path: Path) -> str: ...
def run_paper(record: PaperRecord, settings: RunSettings, client: GeminiClient) -> RunResult: ...
```

The real client uses one uploaded PDF and one `generate_content` request per
paper. Test clients return deterministic responses without network calls.

- [x] **Step 4: Run targeted tests and verify GREEN**

```bash
python3 -m pytest tests/test_config.py tests/test_api_runner.py -v
```

### Task 4: Lossless response parsing

**Files:**
- Create: `src/heldout_pipeline/response_parser.py`
- Create: `tests/test_response_parser.py`

- [x] **Step 1: Write failing parser tests**

Cover valid two-table output, pipes inside escaped cells, missing tables,
duplicate headers, missing required Table 1 columns, unexpected columns, and
preservation of every row.

- [x] **Step 2: Run parser tests and verify RED**

```bash
python3 -m pytest tests/test_response_parser.py -v
```

- [x] **Step 3: Implement parser and parse report**

Expose:

```python
def parse_response(text: str, required_columns: Sequence[str]) -> ParsedResponse: ...
def write_parsed_artifacts(parsed: ParsedResponse, output_dir: Path) -> None: ...
```

Never truncate cells or rows to force a schema match.

- [x] **Step 4: Run parser tests and verify GREEN**

```bash
python3 -m pytest tests/test_response_parser.py -v
```

### Task 5: Ground-truth and prediction normalization

**Files:**
- Create: `src/heldout_pipeline/normalize.py`
- Create: `tests/test_normalize.py`

- [x] **Step 1: Write failing normalization tests**

Cover the existing second-row Excel header, arbitrary held-out paper IDs,
wide-to-long conversion, normalized sample IDs, duplicate keys, missing
prediction columns, and extra extracted samples.

- [x] **Step 2: Run normalization tests and verify RED**

```bash
python3 -m pytest tests/test_normalize.py -v
```

- [x] **Step 3: Implement strict long-form conversion**

Expose:

```python
def load_ground_truth(path: Path, allowed_papers: set[str]) -> pd.DataFrame: ...
def load_predictions(artifact_root: Path, records: Sequence[PaperRecord]) -> pd.DataFrame: ...
def wide_to_long(frame: pd.DataFrame, paper_col: str, sample_col: str) -> pd.DataFrame: ...
```

The normalized key is `paper_id`, `sample_id`, `field_name`. Duplicate keys
raise a validation error.

- [x] **Step 4: Run normalization tests and verify GREEN**

```bash
python3 -m pytest tests/test_normalize.py -v
```

### Task 6: Provenance preparation and validation

**Files:**
- Create: `src/heldout_pipeline/provenance.py`
- Create: `tests/test_provenance.py`

- [x] **Step 1: Write failing provenance tests**

Cover P1/P2/P3 parsing, unknown evidence, no T1 default, external room
temperature/density/melting-point classification, sample-level expansion, and
approved-tier filtering.

- [x] **Step 2: Run provenance tests and verify RED**

```bash
python3 -m pytest tests/test_provenance.py -v
```

- [x] **Step 3: Implement provenance review generation**

Expose:

```python
def prepare_provenance(predictions: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame: ...
def validate_provenance(records: pd.DataFrame) -> list[ValidationIssue]: ...
```

Unrecognized or missing evidence is `tier=NA`, `provenance=unknown`.

- [x] **Step 4: Run provenance tests and verify GREEN**

```bash
python3 -m pytest tests/test_provenance.py -v
```

### Task 7: Corrected evaluator and metrics

**Files:**
- Create: `src/heldout_pipeline/evaluator.py`
- Create: `src/heldout_pipeline/metrics.py`
- Create: `tests/test_evaluator.py`
- Create: `tests/test_metrics.py`

- [x] **Step 1: Write failing regression tests**

Tests must reproduce the received evaluator's absent-column false-correct bug
and ignored-extra-row bug, then assert corrected outcomes. Add tests for all
five outcomes, null normalization, 0.5% numeric tolerance, categorical exact
match, semantic mismatch, external-reference exclusion, and split isolation.

- [x] **Step 2: Run evaluator tests and verify RED**

```bash
python3 -m pytest tests/test_evaluator.py tests/test_metrics.py -v
```

- [x] **Step 3: Implement outer-join scoring**

Expose:

```python
def evaluate_fields(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    provenance: pd.DataFrame,
    config: EvaluationConfig,
) -> pd.DataFrame: ...

def summarize_metrics(outcomes: pd.DataFrame, group_by: Sequence[str]) -> pd.DataFrame: ...
```

Produce precision, recall, F1, all-field accuracy, non-null accuracy, outcome
counts, and support.

- [x] **Step 4: Run evaluator tests and verify GREEN**

```bash
python3 -m pytest tests/test_evaluator.py tests/test_metrics.py -v
```

### Task 8: Reporting and CLI

**Files:**
- Create: `src/heldout_pipeline/reporting.py`
- Create: `src/heldout_pipeline/cli.py`
- Create: `tests/test_reporting.py`
- Create: `tests/test_cli.py`
- Modify: `README.md`

- [x] **Step 1: Write failing report and CLI tests**

Cover separate development/held-out directories, experiment overview rows,
legacy labeling, cost/runtime aggregation, manifest selection, and CLI exit
codes.

- [x] **Step 2: Run report tests and verify RED**

```bash
python3 -m pytest tests/test_reporting.py tests/test_cli.py -v
```

- [x] **Step 3: Implement commands**

Implement:

```text
freeze-prompt
validate-manifest
run
prepare-provenance
evaluate
report
```

- [x] **Step 4: Document the complete workflow**

Document adding arbitrary held-out PDFs, extending the existing ground-truth
workbook, API environment variables, provenance approval, evaluation, and
paper-table generation.

- [x] **Step 5: Run report tests and verify GREEN**

```bash
python3 -m pytest tests/test_reporting.py tests/test_cli.py -v
```

### Task 9: Full verification

**Files:**
- Verify all created files.

- [x] **Step 1: Run the complete test suite**

```bash
python3 -m pytest -v
```

Expected: all tests pass with no network access.

- [x] **Step 2: Run a synthetic end-to-end workflow**

Use temporary development and held-out PDFs, synthetic API responses, and a
small ground-truth workbook. Verify separate reports and row-level outcomes.

- [x] **Step 3: Validate frozen prompt integrity**

```bash
sha256sum prompts/prompty_frozen.md
python3 -m heldout_pipeline.cli freeze-prompt --check
```

- [x] **Step 4: Inspect Git changes**

```bash
git status --short
git diff --check
```

Expected: only intentional project files and no whitespace errors.
