"""Live check of the SerpApi client against the real service.

It spends 2 searches of the SerpApi quota, so it only runs on request:
set SERPAPI_API_KEY in .env, then run  RUN_LIVE_SEARCH=1 python -m pytest tests/test_search_live.py
(in PowerShell:  $env:RUN_LIVE_SEARCH=1; python -m pytest tests/test_search_live.py).
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_review_detection import config  # noqa: E402
from fake_review_detection.clients import search  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_SEARCH") != "1" or not config.SERPAPI_API_KEY,
    reason="live search test: set RUN_LIVE_SEARCH=1 and SERPAPI_API_KEY to run it",
)


def test_web_search_returns_pages_with_snippets():
    results = search.web_search('"Burj Al Arab" helipad')
    assert results
    assert all(result["url"].startswith("http") and result["snippet"] for result in results)


def test_image_search_returns_image_and_page_urls():
    results = search.image_search('"Burj Al Arab" helipad')
    assert results
    assert all(result["image_url"].startswith("http") and result["context_url"].startswith("http") for result in results)
