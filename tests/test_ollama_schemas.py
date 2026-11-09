"""The local Ollama model must answer in every format the pipeline asks for.

These tests call the real model, so they are skipped when Ollama is not running
or the configured model is not installed.
"""

import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_review_detection.agents.claim_extractor import ClassifiedClaim, ExtractedClaim, build_prompt_classify, build_prompt_extract  # noqa: E402
from fake_review_detection.agents.location_checker import ExtractedDestination, LocationJudgment  # noqa: E402
from fake_review_detection.agents.opinion_checker import StanceAnswer  # noqa: E402
from fake_review_detection.agents.web_search_checker import ClaimVerification, RankedEvidence, SufficiencyCheck  # noqa: E402
from fake_review_detection.clients import llm  # noqa: E402

pytestmark = pytest.mark.skipif(llm.check_ollama() is not None, reason="Ollama is not running with the configured model")

CASES = {
    "claim extraction": (
        build_prompt_extract("Staff service was excellent. Located 15 minutes from Dubai Mall."),
        List[ExtractedClaim],
    ),
    "claim classification": (
        build_prompt_classify([{"claim_text": "Staff service was excellent", "claim_type": "service_general"}]),
        List[ClassifiedClaim],
    ),
    "destination": (
        'Extract the destination of the claim "Located 15 minutes from Dubai Mall" as a JSON object '
        "with destination, type (specific or category) and category.",
        ExtractedDestination,
    ),
    "location judgment": (
        'The claim "15 minutes walk from the mall" has a calculated walking time of 12 minutes. '
        "Judge it as a JSON object with verdict, reason and confidence_score.",
        LocationJudgment,
    ),
    "stance": (
        'User Claim: "The staff were friendly". Review Claim: "The staff were rude". '
        "Classify the review claim as a JSON object with classification and confidence_score.",
        StanceAnswer,
    ),
    "evidence ranking": (
        'Rank the evidence for the claim "The hotel has a pool" as a JSON array of objects with index, score and reasoning. '
        "Evidence 1: The pool is open all year. Evidence 2: The lobby is large.",
        List[RankedEvidence],
    ),
    "evidence sufficiency": (
        'Is the evidence "The pool is open all year" sufficient to verify "The hotel has a pool"? '
        "Answer as a JSON object with is_sufficient, reasoning and additional_info_needed.",
        SufficiencyCheck,
    ),
    "claim verification": (
        'Verify the claim "The hotel has a pool" against the evidence "The pool is open all year". Answer as a JSON object '
        "with verdict, confidence_score, reasoning, the four evidence lists, red_flags and recommendation.",
        ClaimVerification,
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_model_answers_in_the_expected_format(name):
    prompt, schema = CASES[name]
    # ask_json raises if the answer does not match the schema
    assert llm.ask_json(prompt, schema) is not None
