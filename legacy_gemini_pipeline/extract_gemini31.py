from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from google import genai

from extract_paper3 import extract_markdown_tables, parse_markdown_table


DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_PAPERS = [f"Paper{i}.pdf" for i in range(1, 31)]
DELAY_SECONDS = 2.5


def redact_secret(message: str, secret: str) -> str:
    return message.replace(secret, "[REDACTED]") if secret else message


def parse_paper_names(values: Iterable[str]) -> list[str]:
    papers: list[str] = []
    for value in values:
        token = value.strip()
        if token.isdigit():
            token = f"Paper{int(token)}.pdf"
        if not re.fullmatch(r"Paper\d+\.pdf", token):
            raise ValueError(
                f"invalid paper {value!r}; use a number or PaperN.pdf"
            )
        papers.append(token)
    return papers


def load_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return getpass.getpass("Enter GEMINI_API_KEY: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def log_message(path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def generate_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
    uploaded_file: object,
) -> object:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            return client.models.generate_content(
                model=model,
                contents=[prompt, uploaded_file],
            )
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(2)
    raise last_error or RuntimeError("Gemini request failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen legacy prompt with the current Gemini SDK."
    )
    parser.add_argument(
        "--papers",
        nargs="+",
        default=DEFAULT_PAPERS,
        help="Paper numbers or filenames; default: Paper1.pdf through Paper30.pdf",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--papers-dir", default="papers")
    parser.add_argument("--output-dir", default="outputs_gemini31_api")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paper_names = parse_paper_names(args.papers)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    base_dir = Path(__file__).resolve().parent
    papers_dir = base_dir / args.papers_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "processing_log.txt"

    api_key = load_api_key()
    if not api_key:
        print("Error: GEMINI_API_KEY is required.", file=sys.stderr)
        return 1

    prompt_path = base_dir / "prompty.md"
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
        return 1
    prompt = prompt_path.read_text(encoding="utf-8")
    client = genai.Client(api_key=api_key)

    failures: list[str] = []
    for index, paper_name in enumerate(paper_names, start=1):
        print(f"Processing {index}/{len(paper_names)}: {paper_name}")
        pdf_path = papers_dir / paper_name
        if not pdf_path.exists():
            message = f"{paper_name}: missing file"
            print(f"Error: {message}", file=sys.stderr)
            log_message(log_path, message)
            failures.append(paper_name)
            continue

        uploaded_file = None
        try:
            uploaded_file = client.files.upload(file=pdf_path)
            response = generate_with_retry(
                client,
                args.model,
                prompt,
                uploaded_file,
            )
            response_text = response.text or ""

            stem = pdf_path.stem
            (output_dir / f"{stem}_raw_response.md").write_text(
                response_text,
                encoding="utf-8",
            )
            tables = extract_markdown_tables(response_text)
            if len(tables) < 2:
                raise ValueError(
                    "expected at least two markdown tables, "
                    f"found {len(tables)}"
                )

            extracted = parse_markdown_table(tables[0])
            evidence = parse_markdown_table(tables[1])
            extracted.to_excel(
                output_dir / f"{stem}_extracted_data.xlsx",
                index=False,
            )
            evidence.to_excel(
                output_dir / f"{stem}_evidence_source.xlsx",
                index=False,
            )
            log_message(log_path, f"{paper_name}: success")
            print(
                f"Saved {len(extracted)} extracted rows and "
                f"{len(evidence)} evidence rows."
            )
        except Exception as exc:
            failures.append(paper_name)
            message = redact_secret(str(exc), api_key)
            log_message(log_path, f"{paper_name}: failed - {message}")
            print(f"Error: {paper_name}: {message}", file=sys.stderr)
        finally:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception as exc:
                    message = redact_secret(str(exc), api_key)
                    log_message(
                        log_path,
                        f"{paper_name}: cleanup failed - {message}",
                    )

        if index < len(paper_names):
            time.sleep(DELAY_SECONDS)

    print(
        f"Processed {len(paper_names)} papers; "
        f"failures={len(failures)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
