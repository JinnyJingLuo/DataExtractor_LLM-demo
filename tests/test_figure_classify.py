# tests/test_figure_classify.py
from pathlib import Path

from figure_classify import classify_figure_type, cv_available_for


class _FakeUploadedFile:
    name = "files/fake"


def _fake_call_upload(api_key, path):
    class _FakeClient:
        class files:
            @staticmethod
            def delete(name):
                pass

    return _FakeClient(), _FakeUploadedFile()


def test_classify_figure_type_exact_label():
    def fake_call_llm(api_key, model_id, contents):
        return "discrete-marker", {"input_tokens": 10, "output_tokens": 2, "thinking_tokens": 0, "total_tokens": 12}

    result, usage = classify_figure_type(
        "fake-key", "fake-model", Path("/tmp/fake.png"),
        call_llm=fake_call_llm, call_upload=_fake_call_upload,
    )
    assert result == "discrete-marker"
    assert usage == {"input_tokens": 10, "output_tokens": 2, "thinking_tokens": 0, "total_tokens": 12}


def test_classify_figure_type_extracts_label_from_extra_text():
    def fake_call_llm(api_key, model_id, contents):
        return "I think this is a Bar Chart.", {}

    result, _usage = classify_figure_type(
        "fake-key", "fake-model", Path("/tmp/fake.png"),
        call_llm=fake_call_llm, call_upload=_fake_call_upload,
    )
    assert result == "bar"


def test_classify_figure_type_unparseable_defaults_to_non_chart():
    def fake_call_llm(api_key, model_id, contents):
        return "unrelated gibberish response", {}

    result, _usage = classify_figure_type(
        "fake-key", "fake-model", Path("/tmp/fake.png"),
        call_llm=fake_call_llm, call_upload=_fake_call_upload,
    )
    assert result == "non-chart"


def test_cv_available_only_for_discrete_marker():
    assert cv_available_for("discrete-marker") is True
    assert cv_available_for("line-with-markers") is False
    assert cv_available_for("continuous-curve") is False
    assert cv_available_for("bar") is False
    assert cv_available_for("box-plot") is False
    assert cv_available_for("non-chart") is False
