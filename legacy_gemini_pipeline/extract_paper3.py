from __future__ import annotations

import getpass
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import google.generativeai as genai
import pandas as pd

MODEL_NAME = "gemini-2.5-pro"
PAPER_NAMES = [f"Paper{i}.pdf" for i in range(1, 31)]
PAPERS_DIR = "papers"
DELAY_SECONDS = 2.5


def load_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key
    try:
        api_key = getpass.getpass("Enter GEMINI_API_KEY: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
    return api_key


def extract_markdown_tables(text: str) -> List[str]:
    table_re = re.compile(
        r"(?:^|\n)(\|.*\|\n\|[ \-:|]+\|\n(?:\|.*\|\n?)*)",
        re.MULTILINE,
    )
    return [match.group(1).strip() for match in table_re.finditer(text)]


def parse_markdown_table(block: str) -> pd.DataFrame:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Markdown table is too short to parse.")

    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    data_rows = []
    for line in lines[2:]:
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if len(row) < len(header):
            row.extend([""] * (len(header) - len(row)))
        elif len(row) > len(header):
            row = row[: len(header)]
        data_rows.append(row)

    return pd.DataFrame(data_rows, columns=header)


def log_message(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(exist_ok=True)
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def estimate_remaining(avg_seconds: float, remaining: int) -> str:
    if avg_seconds <= 0 or remaining <= 0:
        return "estimating..."
    total_seconds = int(avg_seconds * remaining)
    minutes, seconds = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def call_gemini_with_retry(
    model: genai.GenerativeModel,
    protocol: str,
    uploaded_file: genai.File,
) -> Tuple[str, Optional[object]]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, 3):
        try:
            response = model.generate_content([protocol, uploaded_file])
            return response.text or "", response
        except Exception as exc:
            last_exc = exc
            if attempt == 1:
                time.sleep(2)
    raise last_exc or RuntimeError("Gemini call failed.")


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    prompt_path = base_dir / "prompty.md"
    papers_dir = base_dir / PAPERS_DIR
    output_dir = base_dir / "outputs"
    log_path = output_dir / "processing_log.txt"

    print("[1/6] Loading API key...")
    api_key = load_api_key()
    if not api_key:
        print("Error: GEMINI_API_KEY is required.", file=sys.stderr)
        return 1

    print("[2/6] Reading extraction protocol...")
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
        return 1
    protocol = prompt_path.read_text(encoding="utf-8")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    successes: List[str] = []
    failures: List[str] = []
    durations: List[float] = []
    total_papers = len(PAPER_NAMES)

    for idx, paper_name in enumerate(PAPER_NAMES, start=1):
        pdf_path = papers_dir / paper_name
        print(f"\nProcessing Paper {idx} of {total_papers}... ({paper_name})")
        if not pdf_path.exists():
            message = f"{paper_name}: missing file"
            print(f"Error: {message}", file=sys.stderr)
            log_message(log_path, message)
            failures.append(paper_name)
            continue

        start_time = time.time()
        uploaded_file = None
        try:
            print("Uploading PDF to Gemini File API...")
            uploaded_file = genai.upload_file(
                path=str(pdf_path),
                mime_type="application/pdf",
            )

            print("Sending protocol + PDF to Gemini...")
            response_text, response_obj = call_gemini_with_retry(
                model,
                protocol,
                uploaded_file,
            )

            print("Parsing markdown tables...")
            tables = extract_markdown_tables(response_text)
            if len(tables) < 2:
                raise ValueError(
                    "Expected at least two markdown tables in the response, "
                    f"found {len(tables)}."
                )

            table1 = parse_markdown_table(tables[0])
            table2 = parse_markdown_table(tables[1])

            output_dir.mkdir(exist_ok=True)
            stem = Path(paper_name).stem
            extracted_path = output_dir / f"{stem}_extracted_data.xlsx"
            evidence_path = output_dir / f"{stem}_evidence_source.xlsx"

            table1.to_excel(extracted_path, index=False)
            table2.to_excel(evidence_path, index=False)

            if response_obj is not None and hasattr(response_obj, "usage_metadata"):
                usage = response_obj.usage_metadata
                print(
                    "Token usage:",
                    getattr(usage, "prompt_token_count", "n/a"),
                    getattr(usage, "candidates_token_count", "n/a"),
                    getattr(usage, "total_token_count", "n/a"),
                )

            successes.append(paper_name)
            log_message(log_path, f"{paper_name}: success")
            print(f"Saved outputs for {paper_name}.")

        except Exception as exc:
            failures.append(paper_name)
            error_message = f"{paper_name}: failed - {exc}"
            print(f"Error: {error_message}", file=sys.stderr)
            log_message(log_path, error_message)

        finally:
            if uploaded_file is not None:
                try:
                    print("Cleaning up uploaded file...")
                    genai.delete_file(uploaded_file.name)
                except Exception as exc:
                    log_message(log_path, f"{paper_name}: cleanup failed - {exc}")
                    print(f"Warning: failed to delete uploaded file: {exc}")

        elapsed = time.time() - start_time
        durations.append(elapsed)
        avg_seconds = sum(durations) / len(durations)
        remaining = total_papers - idx
        print(f"Completed in {elapsed:.1f}s. ETA: {estimate_remaining(avg_seconds, remaining)}")

        if remaining:
            time.sleep(DELAY_SECONDS)

    print("\nBatch summary:")
    print(f"Successes ({len(successes)}): {', '.join(successes) if successes else 'None'}")
    print(f"Failures ({len(failures)}): {', '.join(failures) if failures else 'None'}")
    log_message(log_path, f"Summary: successes={len(successes)}, failures={len(failures)}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
