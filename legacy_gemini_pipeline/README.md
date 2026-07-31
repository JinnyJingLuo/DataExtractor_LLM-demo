# Legacy Gemini Pipeline Reproduction

This directory preserves the extraction and evaluation scripts received in
`Gemini_Pipeline.zip`. The extraction script uses the Gemini Developer API
model configured in the received code:

```text
gemini-2.5-pro
```

This experiment reproduces the received code package. It is not an exact
reproduction of the paper's historical Gemini 3 Pro chatbox run.

The copied `prompty.md` is byte-identical to `prompts/prompty_frozen.md`. The
ground-truth workbook in the received ZIP is byte-identical to
`data/Ground_truth Table.xlsx`.

Google no longer makes `gemini-2.5-pro` available to new Developer API users.
The preserved `extract_paper3.py` therefore documents the received code but
cannot run with a new account. `extract_gemini31.py` keeps the frozen prompt,
table parser, and workbook format while using the current `google-genai` SDK
and `gemini-3.1-pro-preview`. Results from this compatibility run must be
reported separately from an exact legacy reproduction.

## Server Setup

Run from the repository root:

```bash
python -m venv .venv-legacy
source .venv-legacy/bin/activate
python -m pip install --upgrade pip
python -m pip install -r legacy_gemini_pipeline/requirements.txt
ln -s "../Papers/Sample Papers" legacy_gemini_pipeline/papers
mkdir -p legacy_gemini_pipeline/outputs
```

The legacy code uses the Developer API, not Vertex ADC:

```bash
export GEMINI_API_KEY="your-developer-api-key"
```

Record the environment before the run:

```bash
python --version > legacy_gemini_pipeline/outputs/environment.txt
python -m pip freeze >> legacy_gemini_pipeline/outputs/environment.txt
sha256sum legacy_gemini_pipeline/prompty.md \
  "data/Ground_truth Table.xlsx" \
  legacy_gemini_pipeline/papers/Paper*.pdf \
  > legacy_gemini_pipeline/outputs/input_sha256.txt
```

## Extraction

The script always processes `Paper1.pdf` through `Paper30.pdf` and writes to
`legacy_gemini_pipeline/outputs/`.

```bash
cd legacy_gemini_pipeline
PYTHONUNBUFFERED=1 python extract_paper3.py 2>&1 | tee outputs/legacy_run.log
cd ..
```

Do not edit or manually align the extracted workbooks before evaluation.

## Gemini 3.1 Compatibility Run

Create a new Gemini Developer API key in Google AI Studio and load it without
putting the key in shell history:

```bash
unset GOOGLE_API_KEY GEMINI_API_KEY
read -rsp "Gemini API key: " GEMINI_API_KEY
echo
export GEMINI_API_KEY
```

Run Paper 1 first:

```bash
cd legacy_gemini_pipeline
python extract_gemini31.py \
  --papers 1 \
  --output-dir outputs_gemini31_api
```

After inspecting the Paper 1 workbook, process the complete development set:

```bash
python extract_gemini31.py \
  --papers 1 2 3 4 5 6 7 8 9 10 \
           11 12 13 14 15 16 17 18 19 20 \
           21 22 23 24 25 26 27 28 29 30 \
  --output-dir outputs_gemini31_api
```

The current key is redacted from exceptions before they are printed or
written to `processing_log.txt`. Revoke any key that has appeared in terminal
output, a saved log, source control, or a shared message.

## Combine And Evaluate

```bash
cd legacy_gemini_pipeline

python combine_extraction_outputs.py \
  --input-dir outputs \
  --extracted-output outputs/Extracted_Data_Combined.xlsx \
  --evidence-output outputs/Evidence_Source_Combined.xlsx

python evaluate_extraction.py \
  --ground-truth "../data/Ground_truth Table.xlsx" \
  --extracted outputs/Extracted_Data_Combined.xlsx \
  --evidence outputs/Evidence_Source_Combined.xlsx \
  --output outputs/evaluation_results.xlsx \
  2>&1 | tee outputs/legacy_evaluation.log

cd ..
```

The legacy evaluator:

- matches rows only by normalized `Sample ID`;
- ignores prediction-only records because it uses a ground-truth left join;
- counts null-null cells as correct;
- defaults missing priority evidence to T1;
- does not parse uncertainty or range notation.

Keep this result separate from `artifacts/vertex_30` and the corrected
evaluation under `results/vertex_30`.
