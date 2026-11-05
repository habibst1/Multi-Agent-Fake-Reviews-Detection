# fake_review_detection/clients/search.py

"""
Web and image search through SerpApi, which returns Google results.

Both functions return the result shape the web search checker used with Google's
Custom Search JSON API, which Google shuts down on 1 January 2027.
"""

from typing import Any, Dict, List

from fake_review_detection import config
from fake_review_detection.clients import http_client

SERPAPI_URL = "https://serpapi.com/search"


class SearchError(RuntimeError):
    """SerpApi refused the search, for example because of a wrong key or an empty monthly quota."""


def web_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """Google web results as dicts with 'snippet', 'title' and 'url'; results without a snippet are skipped."""
    data = _serpapi({"engine": "google", "q": query})
    results = []
    for item in data.get("organic_results", [])[:num_results]:
        snippet = item.get("snippet", "")
        if snippet:
            results.append({
                'snippet': snippet,
                'title': item.get("title", ""),
                'url': item.get("link", "")
            })
    return results


def image_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """Google Images results as dicts with 'image_url', 'title', 'context_url' (the page showing it) and 'snippet'."""
    data = _serpapi({"engine": "google_images", "q": query})
    results = []
    for item in data.get("images_results", [])[:num_results]:
        image_url = item.get("original", "")
        if image_url:
            results.append({
                'image_url': image_url,
                'title': item.get("title", ""),
                'context_url': item.get("link", ""),
                # Google Images results carry no caption; the title stands in for the old image snippet
                'snippet': item.get("title", "")
            })
    return results


def _serpapi(params: Dict[str, str]) -> Dict[str, Any]:
    response = http_client.get(SERPAPI_URL, params={**params, "api_key": config.SERPAPI_API_KEY}, timeout=30)
    if response.status_code >= 400:
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        raise SearchError(f"SerpApi answered HTTP {response.status_code}: {message}")
    # A query without results comes back as HTTP 200 with an "error" field and no results
    return response.json()
