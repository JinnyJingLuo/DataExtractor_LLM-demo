import json
import struct
from pathlib import Path
from unittest import mock

import pytest

import run_figure_refine as m
from run_figure_refine import LocalizationError, locate_figure_bbox


class _FakeUploadedFile:
    name = "files/fake"


class _FakeClient:
    class files:
        @staticmethod
        def delete(name):
            pass


def _fake_upload_file(api_key, path):
    return _FakeClient(), _FakeUploadedFile()


def _make_fake_png(path: Path, width: int, height: int) -> Path:
    # png_size() only reads the first 24 bytes (signature + IHDR's width/
    # height fields) -- no need for a real, fully-valid PNG body.
    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)
    path.write_bytes(header)
    return path


def test_locate_figure_bbox_success_returns_usage(tmp_path):
    page_image = _make_fake_png(tmp_path / "fake.png", 1000, 1000)

    def fake_run_gemini(api_key, model_id, contents, media_resolution="MEDIA_RESOLUTION_HIGH"):
        return json.dumps({"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}), {
            "input_tokens": 100, "output_tokens": 10, "thinking_tokens": 0, "total_tokens": 110,
        }

    with mock.patch.object(m, "run_gemini", fake_run_gemini), mock.patch.object(m, "upload_file", _fake_upload_file):
        bbox, usage = locate_figure_bbox("fake-key", "fake-model", page_image, "5")

    assert bbox == pytest.approx((0.1, 0.1, 0.9, 0.9))
    assert usage == {"input_tokens": 100, "output_tokens": 10, "thinking_tokens": 0, "total_tokens": 110}


def test_locate_figure_bbox_out_of_range_raises_localization_error_with_usage(tmp_path):
    # y values (594, 792) exceed the fake page image's own height (500), so
    # they can't be normalized as pixel coordinates either -- genuinely
    # out of range, not just mixed fraction/pixel units.
    page_image = _make_fake_png(tmp_path / "fake.png", 600, 500)

    def fake_run_gemini(api_key, model_id, contents, media_resolution="MEDIA_RESOLUTION_HIGH"):
        return json.dumps({"x_min": 0.0, "y_min": 594.0, "x_max": 1.0, "y_max": 792.0}), {
            "input_tokens": 50, "output_tokens": 5, "thinking_tokens": 0, "total_tokens": 55,
        }

    with mock.patch.object(m, "run_gemini", fake_run_gemini), mock.patch.object(m, "upload_file", _fake_upload_file):
        with pytest.raises(LocalizationError) as exc_info:
            locate_figure_bbox("fake-key", "fake-model", page_image, "4")

    # The token spend from the failed call is not lost -- it's on the exception.
    assert exc_info.value.usage == {"input_tokens": 50, "output_tokens": 5, "thinking_tokens": 0, "total_tokens": 55}


def test_locate_figure_bbox_unparseable_response_raises_localization_error_with_usage():
    def fake_run_gemini(api_key, model_id, contents, media_resolution="MEDIA_RESOLUTION_HIGH"):
        return "sorry, I could not find that figure", {
            "input_tokens": 30, "output_tokens": 8, "thinking_tokens": 0, "total_tokens": 38,
        }

    with mock.patch.object(m, "run_gemini", fake_run_gemini), mock.patch.object(m, "upload_file", _fake_upload_file):
        with pytest.raises(LocalizationError) as exc_info:
            locate_figure_bbox("fake-key", "fake-model", Path("/tmp/fake.png"), "4")

    assert exc_info.value.usage == {"input_tokens": 30, "output_tokens": 8, "thinking_tokens": 0, "total_tokens": 38}


def test_localization_error_is_a_runtime_error():
    # Existing exception-handling call sites catch RuntimeError -- must stay compatible.
    assert issubclass(LocalizationError, RuntimeError)
