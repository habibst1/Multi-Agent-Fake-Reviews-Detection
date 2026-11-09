"""The SerpApi client must return the result shape the web search checker used with Google Custom Search."""

import sys
import time
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_review_detection.clients import search  # noqa: E402


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.fixture
def serpapi(monkeypatch):
    """Answer SerpApi requests with canned payloads, keyed by engine, and record the parameters sent."""
    answers, sent = {}, []

    def fake_get(url, params=None, **kwargs):
        assert url == search.SERPAPI_URL
        sent.append(params)
        return answers[params["engine"]]

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(search.config, "SERPAPI_API_KEY", "key-for-tests")
    return answers, sent


def test_web_results_keep_title_url_and_snippet(serpapi):
    answers, sent = serpapi
    answers["google"] = _Response(200, {"organic_results": [
        {"title": "Pool", "link": "https://a.test", "snippet": "A rooftop pool."},
        {"title": "No snippet", "link": "https://b.test"},
        {"title": "Spa", "link": "https://c.test", "snippet": "A spa and sauna."},
    ]})
    assert search.web_search('"Hotel" has a pool', num_results=10) == [
        {"snippet": "A rooftop pool.", "title": "Pool", "url": "https://a.test"},
        {"snippet": "A spa and sauna.", "title": "Spa", "url": "https://c.test"},
    ]
    assert sent == [{"engine": "google", "q": '"Hotel" has a pool', "api_key": "key-for-tests"}]


def test_web_results_are_limited_to_num_results(serpapi):
    answers, _ = serpapi
    answers["google"] = _Response(200, {"organic_results": [
        {"title": f"Page {i}", "link": f"https://{i}.test", "snippet": f"Snippet {i}"} for i in range(20)
    ]})
    assert len(search.web_search("query", num_results=10)) == 10


def test_image_results_map_to_image_page_and_title(serpapi):
    answers, _ = serpapi
    answers["google_images"] = _Response(200, {"images_results": [
        {"title": "Rooftop pool", "original": "https://img.test/pool.jpg", "link": "https://site.test/pool", "source": "site.test"},
        {"title": "Broken result", "link": "https://site.test/none"},
    ]})
    assert search.image_search("query") == [{
        "image_url": "https://img.test/pool.jpg",
        "title": "Rooftop pool",
        "context_url": "https://site.test/pool",
        "snippet": "Rooftop pool",
    }]


def test_a_query_without_results_returns_an_empty_list(serpapi):
    answers, _ = serpapi
    answers["google"] = _Response(200, {"error": "Google hasn't returned any results for this query."})
    assert search.web_search("nothing matches this") == []


def test_refused_searches_raise_with_serpapis_message(serpapi):
    answers, _ = serpapi
    answers["google_images"] = _Response(401, {"error": "Invalid API key."})
    with pytest.raises(search.SearchError, match="HTTP 401: Invalid API key."):
        search.image_search("query")
