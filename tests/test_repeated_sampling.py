# tests/test_repeated_sampling.py
from pathlib import Path

import pandas as pd

from repeated_sampling import collect_llm_draws


def test_collect_llm_draws_across_multiple_calls():
    call_count = {"n": 0}

    def fake_call_refine(api_key, model_id, base_prompt, field_name, figure_number, figure_image, pass1_extracted):
        call_count["n"] += 1
        value = 3.500 + 0.001 * call_count["n"]
        frame = pd.DataFrame(
            [
                {"Sample ID": "VA0.6-300", field_name: f"{value:.3f}", "Evidence Note": "calc-derived"},
                {"Sample ID": "VA0-300", field_name: "-", "Evidence Note": "not plotted for 0% pre-strain"},
            ]
        )
        return frame, {"input_tokens": 100, "output_tokens": 20, "thinking_tokens": 50, "total_tokens": 170}

    per_sample, usage = collect_llm_draws(
        api_key="fake-key",
        model_id="fake-model",
        base_prompt="base prompt text",
        field_name="Longitudinal Stress at HEL (GPa)",
        figure_number="5",
        figure_image=Path("/tmp/fake.png"),
        pass1_extracted=pd.DataFrame([{"Sample ID": "VA0.6-300"}, {"Sample ID": "VA0-300"}]),
        n_draws=3,
        call_refine=fake_call_refine,
    )

    assert len(per_sample["VA0.6-300"]) == 3
    assert per_sample["VA0.6-300"][0]["value"] == 3.501
    assert per_sample["VA0.6-300"][0]["evidence"] == "calc-derived"
    # "-" draws are skipped for that sample, not recorded as a bad float
    assert "VA0-300" not in per_sample
    # usage is summed across all 3 draws, not just the last one
    assert usage == {"input_tokens": 300, "output_tokens": 60, "thinking_tokens": 150, "total_tokens": 510}


def test_collect_llm_draws_skips_one_failing_draw_without_losing_the_rest():
    call_count = {"n": 0}

    def fake_call_refine(api_key, model_id, base_prompt, field_name, figure_number, figure_image, pass1_extracted):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise ValueError("model returned an unparseable table this draw")
        frame = pd.DataFrame([{"Sample ID": "VA0.6-300", field_name: "3.500", "Evidence Note": "figure read"}])
        return frame, {"input_tokens": 100, "output_tokens": 20, "thinking_tokens": 50, "total_tokens": 170}

    per_sample, usage = collect_llm_draws(
        api_key="fake-key",
        model_id="fake-model",
        base_prompt="base prompt text",
        field_name="Spall Strength (GPa)",
        figure_number="5",
        figure_image=Path("/tmp/fake.png"),
        pass1_extracted=pd.DataFrame([{"Sample ID": "VA0.6-300"}]),
        n_draws=3,
        call_refine=fake_call_refine,
    )

    # draw 2 failed, but draws 1 and 3 still succeeded -- not discarded
    assert len(per_sample["VA0.6-300"]) == 2
    # usage only reflects the 2 successful calls, not the failed one
    assert usage == {"input_tokens": 200, "output_tokens": 40, "thinking_tokens": 100, "total_tokens": 340}


def test_collect_llm_draws_returns_empty_when_every_draw_fails():
    def always_fails(api_key, model_id, base_prompt, field_name, figure_number, figure_image, pass1_extracted):
        raise RuntimeError("network error")

    per_sample, usage = collect_llm_draws(
        api_key="fake-key",
        model_id="fake-model",
        base_prompt="base prompt text",
        field_name="Spall Strength (GPa)",
        figure_number="5",
        figure_image=Path("/tmp/fake.png"),
        pass1_extracted=pd.DataFrame([{"Sample ID": "VA0.6-300"}]),
        n_draws=3,
        call_refine=always_fails,
    )

    assert per_sample == {}
    assert usage == {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}


from repeated_sampling import collect_match_draws


def test_collect_match_draws_across_multiple_calls():
    call_count = {"n": 0}

    def fake_call_match(api_key, model_id, image_path, prompt):
        call_count["n"] += 1
        # first two calls agree, third disagrees -- exercises majority-vote input shape
        marker = "1" if call_count["n"] != 3 else "2"
        return (
            f"| Sample ID | Marker Number |\n| --- | --- |\n| VA0.6-300 | {marker} |\n",
            {"input_tokens": 5, "output_tokens": 1, "thinking_tokens": 0, "total_tokens": 6},
        )

    filled_markers = [
        {"marker_number": 1, "cx": 10.0, "cy": 20.0, "value": 3.5},
        {"marker_number": 2, "cx": 30.0, "cy": 40.0, "value": 1.5},
    ]

    per_sample, usage = collect_match_draws(
        api_key="fake-key",
        model_id="fake-model",
        image_path=Path("/tmp/fake.png"),
        filled_markers=filled_markers,
        field_name="Spall Strength (GPa)",
        figure_number="4",
        pass1_extracted=pd.DataFrame([{"Sample ID": "VA0.6-300"}]),
        n_draws=3,
        call_match=fake_call_match,
    )

    assert per_sample["VA0.6-300"] == ["1", "1", "2"]
    assert usage == {"input_tokens": 15, "output_tokens": 3, "thinking_tokens": 0, "total_tokens": 18}


def test_collect_match_draws_skips_one_failing_draw_without_losing_the_rest():
    call_count = {"n": 0}

    def fake_call_match(api_key, model_id, image_path, prompt):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise ValueError("model returned an unparseable table this draw")
        return (
            f"| Sample ID | Marker Number |\n| --- | --- |\n| VA0.6-300 | 1 |\n",
            {"input_tokens": 5, "output_tokens": 1, "thinking_tokens": 0, "total_tokens": 6},
        )

    filled_markers = [{"marker_number": 1, "cx": 10.0, "cy": 20.0, "value": 3.5}]

    per_sample, usage = collect_match_draws(
        api_key="fake-key",
        model_id="fake-model",
        image_path=Path("/tmp/fake.png"),
        filled_markers=filled_markers,
        field_name="Spall Strength (GPa)",
        figure_number="4",
        pass1_extracted=pd.DataFrame([{"Sample ID": "VA0.6-300"}]),
        n_draws=3,
        call_match=fake_call_match,
    )

    assert per_sample["VA0.6-300"] == ["1", "1"]
    assert usage == {"input_tokens": 10, "output_tokens": 2, "thinking_tokens": 0, "total_tokens": 12}
