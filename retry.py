# retry.py

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

ATTEMPTS = 3
FIRST_DELAY_SECONDS = 0.5


def call_with_retries(fn: Callable[[], T], is_transient: Callable[[Exception], bool]) -> T:
    """Call `fn`, retrying transient failures with exponential backoff (0.5 s, then 1 s)."""
    delay = FIRST_DELAY_SECONDS
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return fn()
        except Exception as error:
            if attempt == ATTEMPTS or not is_transient(error):
                raise
            logger.warning(f"⚠️ Attempt {attempt} of {ATTEMPTS} failed ({error}); retrying in {delay:.1f}s")
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")
