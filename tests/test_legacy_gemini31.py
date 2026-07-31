from __future__ import annotations

import sys
from pathlib import Path


LEGACY_DIR = Path(__file__).resolve().parents[1] / "legacy_gemini_pipeline"
sys.path.insert(0, str(LEGACY_DIR))

from extract_gemini31 import parse_paper_names, redact_secret  # noqa: E402


def test_redact_secret_removes_api_key_from_error_message() -> None:
    api_key = "secret-api-key-value"
    message = f"https://example.test?key={api_key}: request failed"

    assert redact_secret(message, api_key) == (
        "https://example.test?key=[REDACTED]: request failed"
    )


def test_parse_paper_names_accepts_numbers_and_pdf_names() -> None:
    assert parse_paper_names(["1", "Paper2.pdf"]) == [
        "Paper1.pdf",
        "Paper2.pdf",
    ]
