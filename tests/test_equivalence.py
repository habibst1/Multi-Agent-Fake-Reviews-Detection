"""The LangGraph pipeline must give the same results as the original one when it gets the same answers.

Each scenario runs twice in a fake world (fake LLM, fake web services, fake embedding models):
once through the original code from master (tests/legacy), once through the migrated code.
Only these documented differences are allowed:
- error messages may end with " [reason: ...]", and say "LLM" where they said "Groq";
- additional_data now carries the opinion checker's aggregation_results and matched_hotel;
- the stance prompt asks for a JSON answer instead of "classification,confidence".
"""

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

SCENARIOS = ["readme_example", "web_claims", "failures", "unknown_hotel", "no_claims"]

LEGACY_STANCE_FORMAT = (
    "Respond with ONLY the classification and a confidence score from 0.0 to 1.0, separated by a comma.\n"
    "Format: classification,confidence_score\n"
    "Example: supports,0.85\n"
)
NEW_STANCE_FORMAT = (
    "Respond with ONLY a JSON object with the classification and a confidence score from 0.0 to 1.0.\n"
    'Format: {"classification": "...", "confidence_score": 0.0}\n'
    'Example: {"classification": "supports", "confidence_score": 0.85}\n'
)

REASON_SUFFIX = re.compile(r" \[reason: .*\]$", re.S)
LLM_FAILURE_TEXTS = [
    (("Groq API error:", "LLM error:"), "<llm error>"),
    (("Failed to parse Groq response:", "Failed to parse LLM response:"), "<llm parse error>"),
    (("Error in Groq API:", "Error in LLM call:"), "<llm error>"),
    (("Ranking failed due to error:",), "<ranking error>"),
]


def _normalize_text(text: str) -> str:
    text = REASON_SUFFIX.sub("", text)
    for prefixes, placeholder in LLM_FAILURE_TEXTS:
        if text.startswith(prefixes):
            return placeholder
    return text


def _normalize(value):
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items() if key != "additional_data"}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        return _normalize_text(value)
    return value


def _run_both(scenario: str, tmp_path: Path, *new_flags: str):
    """Run the scenario through both implementations at the same time and return their outputs."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    jobs = {}
    for impl, flags in (("legacy", ()), ("new", new_flags)):
        output_path = tmp_path / f"{impl}.json"
        log_path = tmp_path / f"{impl}.log"
        log = open(log_path, "w", encoding="utf-8")
        command = [sys.executable, str(TESTS_DIR / "run_scenario.py"), impl, scenario, str(output_path), *flags]
        jobs[impl] = (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), output_path, log, log_path)

    outputs = {}
    for impl, (process, output_path, log, log_path) in jobs.items():
        returncode = process.wait(timeout=900)
        log.close()
        assert returncode == 0, f"The {impl} run failed:\n{log_path.read_text(encoding='utf-8')[-4000:]}"
        outputs[impl] = json.loads(output_path.read_text(encoding="utf-8"))
    return outputs["legacy"], outputs["new"]


def _check_additional_data(result) -> None:
    for claim in result.get("claims_results", []):
        if claim["checker_used"] == "review":
            assert set(claim["additional_data"]) == {"aggregation_results", "matched_hotel"}, claim["claim_id"]
        else:
            assert claim["additional_data"] == {}, claim["claim_id"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_same_results_as_the_original_pipeline(scenario, tmp_path):
    legacy, new = _run_both(scenario, tmp_path)

    assert _normalize(new["result"]) == _normalize(legacy["result"])
    # Same prompts, in any order: the migrated pipeline checks claims in parallel
    legacy_prompts = [prompt.replace(LEGACY_STANCE_FORMAT, NEW_STANCE_FORMAT) for prompt in legacy["prompts"]]
    assert Counter(new["prompts"]) == Counter(legacy_prompts)
    _check_additional_data(new["result"])


def test_sequential_mode_gives_the_same_results(tmp_path):
    legacy, new = _run_both("readme_example", tmp_path, "--sequential")

    assert _normalize(new["result"]) == _normalize(legacy["result"])
    assert len(new["prompts"]) == len(legacy["prompts"])
