"""API keys must never reach the logs or the saved results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_log import redact, redact_value  # noqa: E402

GOOGLE_ERROR = (
    "403 Client Error: Forbidden for url: "
    "https://www.googleapis.com/customsearch/v1?key=AIzaSyEXAMPLEKEY123&cx=engine-1&q=pool&num=10"
)


def test_keys_in_request_urls_are_hidden():
    message = redact(GOOGLE_ERROR)
    assert "AIzaSyEXAMPLEKEY123" not in message
    assert "key=[redacted]" in message
    assert "cx=engine-1&q=pool" in message


def test_geoapify_style_api_key_parameter_is_hidden():
    message = redact("Error for url: https://api.geoapify.com/v2/places?categories=airport&apiKey=geo-secret-42&limit=3")
    assert "geo-secret-42" not in message
    assert "&limit=3" in message


def test_configured_key_values_are_hidden_anywhere(monkeypatch):
    monkeypatch.setenv("ORS_API_KEY", "ors-secret-value")
    assert redact("Authorization: ors-secret-value was refused") == "Authorization: [redacted] was refused"


def test_results_are_redacted_recursively():
    result = {"claims_results": [{"evidence": {"error": f"No search results found [reason: {GOOGLE_ERROR}]"}}], "summary": {"total_claims": 1}}
    redacted = redact_value(result)
    assert "AIzaSyEXAMPLEKEY123" not in str(redacted)
    assert redacted["summary"] == {"total_claims": 1}
