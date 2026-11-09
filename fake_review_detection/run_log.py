# fake_review_detection/run_log.py

import logging
import os
import re
import sys
from contextvars import ContextVar
from typing import Any, List, Optional

from fake_review_detection import config

# The claim the current log lines belong to, and the failures caught while checking it.
# Claims run in parallel threads; each one sets its own values.
current_claim: ContextVar[Optional[str]] = ContextVar("current_claim", default=None)
_failures: ContextVar[Optional[List[str]]] = ContextVar("failures", default=None)

logger = logging.getLogger(__name__)

# HTTP errors quote the request URL, and Google and Geoapify take their API key as a URL parameter
_KEY_PARAMETER = re.compile(r"([?&](?:key|apikey|api_key|token)=)[^&\s'\"]+", re.IGNORECASE)


def redact(text: str) -> str:
    """Hide API keys in a message, whether they come from a request URL or from the configuration."""
    text = _KEY_PARAMETER.sub(r"\1[redacted]", text)
    for name in config.SECRET_KEYS:
        secret = os.getenv(name)
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def redact_value(value: Any) -> Any:
    """Apply `redact` to every string inside a result (dicts, lists and strings)."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


def start_scope(claim_id: Optional[str]) -> None:
    """Tag the next log lines with `claim_id` and start an empty list of failure reasons."""
    current_claim.set(claim_id)
    _failures.set([])


def note_failure(message: str) -> None:
    """Log a caught failure and keep it, so the result can say why it failed."""
    message = redact(message)
    logger.warning(message)
    failures = _failures.get()
    if failures is not None:
        failures.append(message)


def with_reasons(message: str) -> str:
    """Append the failures caught for the current claim to an error message."""
    failures = _failures.get()
    if not failures:
        return message
    return f"{message} [reason: {'; '.join(failures)}]"


class _ClaimTagFormatter(logging.Formatter):
    """Prefix each line with its claim id, after any leading blank lines, and hide API keys."""

    def format(self, record: logging.LogRecord) -> str:
        text = redact(super().format(record))
        claim = current_claim.get()
        if not claim:
            return text
        body = text.lstrip("\n")
        return f"{text[:len(text) - len(body)]}[{claim}] {body}"


def setup_logging(level: int = logging.INFO) -> None:
    """Send log lines to stdout, tagged with the claim they belong to."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ClaimTagFormatter("%(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
