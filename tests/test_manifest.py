from pathlib import Path

import pytest

from heldout_pipeline.manifest import ManifestError, load_manifest, select_papers


def write_manifest(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "papers.csv"
    path.write_text(
        "paper_id,pdf_path,split,include,selection_note\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_loads_arbitrary_heldout_filename(tmp_path):
    pdf = tmp_path / "unseen article final.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    manifest = write_manifest(
        tmp_path,
        [f"H001,{pdf},heldout,true,Chosen before prompt freeze"],
    )

    records = load_manifest(manifest)

    assert records[0].paper_id == "H001"
    assert records[0].pdf_path == pdf.resolve()
    assert select_papers(records, "heldout") == records


@pytest.mark.parametrize(
    "rows,message",
    [
        (
            ["P1,a.pdf,development,true,old", "P1,b.pdf,heldout,true,new"],
            "duplicate paper_id",
        ),
        (["P1,a.pdf,test,true,note"], "invalid split"),
        (["P1,a.pdf,heldout,true,"], "selection_note"),
        (["P1,a.pdf,development,maybe,note"], "include"),
    ],
)
def test_rejects_invalid_manifest(tmp_path, rows, message):
    with pytest.raises(ManifestError, match=message):
        load_manifest(write_manifest(tmp_path, rows))


def test_rejects_missing_included_pdf(tmp_path):
    manifest = write_manifest(
        tmp_path,
        ["P1,missing.pdf,development,true,Original development paper"],
    )
    with pytest.raises(ManifestError, match="does not exist"):
        load_manifest(manifest)
