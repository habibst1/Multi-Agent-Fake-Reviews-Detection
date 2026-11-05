# fake_review_detection/config.py

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Load environment variables ---
load_dotenv(PROJECT_ROOT / ".env")

# --- Local LLM served by Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
LLM_TEMPERATURE = 0.1  # same temperature as the original Groq calls
LLM_SEED = 42  # fixed seed so runs repeat
LLM_NUM_CTX = 4096  # fits every prompt; a larger context costs VRAM
LLM_TIMEOUT_SECONDS = 300

# Claims checked at the same time; every LLM call goes to the same local GPU
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))

# --- External services ---
# Nominatim refuses generic User-Agents; its usage policy asks for one that identifies the application
NOMINATIM_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT",
    "MultiAgentFakeReviewsDetection/1.0 (+https://github.com/habibst1/Multi-Agent-Fake-Reviews-Detection)",
)
ORS_API_KEY = os.getenv("ORS_API_KEY")
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# --- Data ---
# Reviews with extracted claims, used by the opinion checker
REVIEWS_DIR = Path(os.getenv("REVIEWS_DIR", str(PROJECT_ROOT / "data" / "reviews_with_claims")))
# Where main.py saves the results of each run
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(PROJECT_ROOT / "results")))

# Environment variables holding secrets; their values never go into logs or results
SECRET_KEYS = ("ORS_API_KEY", "GEOAPIFY_API_KEY", "GOOGLE_API_KEY")

REQUIRED_KEYS = {
    "ORS_API_KEY": "location checker",
    "GEOAPIFY_API_KEY": "location checker",
    "GOOGLE_API_KEY": "web search checker",
    "SEARCH_ENGINE_ID": "web search checker",
}


def missing_api_keys() -> list[str]:
    """Describe every required API key that is not set."""
    return [f"{key} is not set (needed by the {checker})" for key, checker in REQUIRED_KEYS.items() if not os.getenv(key)]
