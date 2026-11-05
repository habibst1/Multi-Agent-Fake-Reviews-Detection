# llm.py

from typing import Any, Optional

import httpx
import requests
from langchain_ollama import ChatOllama
from ollama import ResponseError
from pydantic import TypeAdapter

import config
from retry import call_with_retries


def ask_json(prompt: str, schema: Any) -> Any:
    """
    Send one prompt to the local Ollama model and return its answer as plain Python data.

    Ollama constrains the output to the JSON schema of `schema` (a TypedDict or a list of
    them), so the answer is always valid JSON with the expected fields and allowed values.
    """
    adapter = TypeAdapter(schema)
    chat = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        seed=config.LLM_SEED,
        num_ctx=config.LLM_NUM_CTX,
        format=adapter.json_schema(),
        client_kwargs={"timeout": config.LLM_TIMEOUT_SECONDS},
    )
    message = call_with_retries(lambda: chat.invoke(prompt), _is_transient)
    return adapter.validate_json(message.content)


def _is_transient(error: Exception) -> bool:
    if isinstance(error, ResponseError):
        return error.status_code >= 500
    return isinstance(error, (ConnectionError, httpx.TransportError))


def check_ollama() -> Optional[str]:
    """Describe what keeps Ollama from serving the configured model, or return None if it can."""
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Ollama is not reachable at {config.OLLAMA_BASE_URL} ({e}). Start Ollama and try again."
    installed = {model["name"] for model in response.json().get("models", [])}
    if config.OLLAMA_MODEL not in installed:
        return f"Model '{config.OLLAMA_MODEL}' is not installed in Ollama. Run: ollama pull {config.OLLAMA_MODEL}"
    return None
