# http_client.py

import threading
import time
from typing import Callable

import requests

from retry import call_with_retries

RETRY_STATUSES = {429, 500, 502, 503, 504}

# Nominatim's usage policy allows at most one request per second
NOMINATIM_HOST = "https://nominatim.openstreetmap.org"
NOMINATIM_MIN_INTERVAL_SECONDS = 1.0
_nominatim_lock = threading.Lock()
_last_nominatim_request = 0.0


class _RetryableStatus(Exception):
    def __init__(self, response: requests.Response):
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def get(url: str, **kwargs) -> requests.Response:
    """requests.get, retried on network errors and 429/5xx answers."""
    return _send_with_retries(url, lambda: requests.get(url, **kwargs))


def post(url: str, **kwargs) -> requests.Response:
    """requests.post, retried on network errors and 429/5xx answers."""
    return _send_with_retries(url, lambda: requests.post(url, **kwargs))


def _send_with_retries(url: str, send: Callable[[], requests.Response]) -> requests.Response:
    def attempt() -> requests.Response:
        _wait_for_nominatim(url)
        response = send()
        if response.status_code in RETRY_STATUSES:
            raise _RetryableStatus(response)
        return response

    try:
        return call_with_retries(attempt, _is_transient)
    except _RetryableStatus as last:
        return last.response  # the caller's raise_for_status() reports it, as before


def _is_transient(error: Exception) -> bool:
    return isinstance(error, (_RetryableStatus, requests.ConnectionError, requests.Timeout))


def _wait_for_nominatim(url: str) -> None:
    """Space Nominatim requests at least one second apart, across parallel claims."""
    global _last_nominatim_request
    if not url.startswith(NOMINATIM_HOST):
        return
    with _nominatim_lock:
        wait = _last_nominatim_request + NOMINATIM_MIN_INTERVAL_SECONDS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_nominatim_request = time.monotonic()
