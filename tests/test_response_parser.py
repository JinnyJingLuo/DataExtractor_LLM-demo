import pytest

from heldout_pipeline.response_parser import ParseError, parse_response


def response(table1_header="| Sample ID | Value |", table1_row="| S1 | 10 |"):
    return f"""## Table 1: Extracted Data

{table1_header}
|---|---|
{table1_row}

## Table 2: Evidence Source

| Column Name | Source Location | Notes |
|---|---|---|
| Value | Table 1 | DIRECT (P1): measured |
"""


def test_parses_two_tables_without_losing_rows():
    text = response().replace("| S1 | 10 |\n", "| S1 | 10 |\n| S2 | 20 |\n")
    parsed = parse_response(text, ["Sample ID", "Value"])
    assert parsed.extracted_data["Sample ID"].tolist() == ["S1", "S2"]
    assert parsed.evidence_source.iloc[0]["Column Name"] == "Value"
    assert parsed.report["valid"] is True


def test_preserves_escaped_pipe_in_cell():
    parsed = parse_response(
        response(table1_row=r"| S1 | alpha \| beta |"),
        ["Sample ID", "Value"],
    )
    assert parsed.extracted_data.iloc[0]["Value"] == "alpha | beta"


def test_rejects_missing_required_column():
    with pytest.raises(ParseError, match="missing required"):
        parse_response(response(), ["Sample ID", "Value", "Treatment"])


def test_rejects_missing_second_table():
    with pytest.raises(ParseError, match="two Markdown tables"):
        parse_response("| Sample ID | Value |\n|---|---|\n| S1 | 1 |\n", ["Sample ID", "Value"])


def test_rejects_duplicate_headers():
    with pytest.raises(ParseError, match="duplicate headers"):
        parse_response(
            response(table1_header="| Sample ID | Sample ID |"),
            ["Sample ID"],
        )
