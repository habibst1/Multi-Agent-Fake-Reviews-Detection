# fake_review_detection/agents/claim_extractor.py

import json
import logging
from typing import Any, Dict, List, Literal, TypedDict

from fake_review_detection import config
from fake_review_detection.clients import llm
from fake_review_detection.run_log import note_failure

logger = logging.getLogger(__name__)

# Predefined claim types with brief definitions (can be expanded)
CLAIM_TYPES = [
    "hotel_general",
    "hotel_prices",
    "hotel_design_and_features",
    "hotel_cleanliness",
    "hotel_comfort",
    "hotel_quality",
    "hotel_miscellaneous",
    "rooms_general",
    "rooms_prices",
    "rooms_design_and_features",
    "rooms_cleanliness",
    "rooms_comfort",
    "rooms_quality",
    "rooms_miscellaneous",
    "room_amenities_general",
    "room_amenities_prices",
    "room_amenities_design_and_features",
    "room_amenities_cleanliness",
    "room_amenities_comfort",
    "room_amenities_quality",
    "room_amenities_miscellaneous",
    "facilities_general",
    "facilities_prices",
    "facilities_design_and_features",
    "facilities_cleanliness",
    "facilities_comfort",
    "facilities_quality",
    "facilities_miscellaneous",
    "service_general",
    "location_general",
    "food_drinks_prices",
    "food_drinks_quality",
    "food_drinks_style_and_options",
    "food_drinks_miscellaneous"
]


# ---------- Answer formats (Ollama constrains the output to these) ----------
class ExtractedClaim(TypedDict):
    claim_text: str
    claim_type: Literal[tuple(CLAIM_TYPES)]


class ClassifiedClaim(ExtractedClaim):
    claim_nature: Literal["factual", "opinion"]
    claim_polarity: Literal["positive", "negative", "neutral"]


# ---------- Low-level LLM call ----------
def call_llm(prompt: str, schema: Any) -> Any:
    """Calls the local LLM with the given prompt; the answer follows `schema`."""
    try:
        return llm.ask_json(prompt, schema)
    except Exception as e:
        note_failure(f"Error calling LLM: {e}")
        return []

# ---------- STEP 1: Extract claims + claim_type (preserve phrasing) ----------
def build_prompt_extract(review_text: str) -> str:
    """Builds the prompt for claim extraction."""
    return f"""
You are a claim extractor. Your job is to identify individual fact-checkable claims from the hotel review below.
For each claim, assign a claim_type from this list:
{', '.join(CLAIM_TYPES)}
CRITICAL INSTRUCTIONS:
1. When multiple elements together form one complete idea, or has a cause-effect or premise-conclusion relationship, keep them as a single claim.
   Example: "The hotel advertised free breakfast but charged me for it" should remain one claim, not two.
2. Never rewrite, summarize, or split the original claim text.
3. If a claim is in a section "Positive" or "Negative", keep the section tag in claim_text.
4. Extract only fact-checkable claims that can be checked by a google search / google maps api / others' reviews only , else don't extract.
Return ONLY a valid JSON array of claim objects in the form (no extra text):
[
  {{
    "claim_text": "...",
    "claim_type": "..."
  }}
]
Review:
\"\"\"{review_text}\"\"\"
""".strip()

def extract_claims_and_types(review_text: str) -> List[Dict[str, Any]]:
    """Extracts claims and their types from the review text."""
    prompt = build_prompt_extract(review_text)
    return call_llm(prompt, List[ExtractedClaim])

# ---------- STEP 2: Add claim_nature + claim_polarity (keep text unchanged) ----------
def build_prompt_classify(claims: List[Dict[str, Any]]) -> str:
    """
    Builds the prompt for classifying claims into factual/opinion and positive/negative.
    """
    # NOTE: We serialize the claims exactly as produced by step 1.
    claims_json = json.dumps(claims, ensure_ascii=False, indent=2)
    return f"""
You are a claim classifier. For each claim below, add:
- claim_nature: "factual" or "opinion"
- claim_polarity: "positive", "negative", or "neutral"
Use these rules:
- factual: This claim can be objectively verified using external, structured data or web sources (e.g., maps, hotel website, official records, Google search). It's a statement about a verifiable state of affairs.
- opinion: This claim is based on a guest's personal experience, subjective perception, or requires comparison with other users' reviews to assess its general validity. It cannot be verified by a single external, objective source. These are often about quality, comfort, service experience, or performance.
- positive: favorable/complimentary sentiment
- negative: unfavorable/critical sentiment
- neutral: neither clearly positive nor negative (or mixed/ambiguous)
Return ONLY a valid JSON array (no extra text).
Input claims:
{claims_json}
Expected output:
[
  {{
    "claim_text":,                # unchanged
    "claim_type":,                # unchanged
    "claim_nature": ,
    "claim_polarity":
  }}
]
""".strip()

def classify_nature_and_polarity(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classifies the nature and polarity of the provided claims."""
    if not claims:
        return []
    prompt = build_prompt_classify(claims)
    out = call_llm(prompt, List[ClassifiedClaim])
    # Ensure output length matches input length; a mismatched list is still returned, as before
    if len(out) != len(claims):
        logger.warning("⚠️ Step 2 returned an unexpected result (empty or length mismatch).")
    return out

# ---------- Convenience: Full two-stage pipeline ----------
def process_review_two_stage(review_text: str) -> List[Dict[str, Any]]:
    """
    Processes the review text through the two-stage claim extraction and classification pipeline.
    """
    logger.info(f"\n--- Processing Review (model = {config.OLLAMA_MODEL}) ---\n{review_text}\n")
    # Step 1: Extract claims
    step1 = extract_claims_and_types(review_text)
    if not step1:
        logger.info("❌ No claims extracted or parsing failed in Step 1.")
        return []

    # Step 2: Classify claims
    step2 = classify_nature_and_polarity(step1)
    if not step2:
        logger.info("⚠️ No claims classified or parsing failed in Step 2.")
    return step2

# Modification For pipeline
def extract_claims_pipeline(hotel_name: str, hotel_review: str) -> Dict[str, Any]:
    """
    Final standardized function to extract claims from a hotel review.

    Args:
        hotel_name (str): Name of the hotel (not used directly here but kept for consistency)
        hotel_review (str): The text of the review.

    Returns:
        dict: A dictionary containing extracted claims in standardized format.
              e.g., {"claims": [{"claim_text": "...", "claim_type": "...", ...}, ...]}
    """
    extracted_claims = process_review_two_stage(hotel_review)
    return {
        "claims": extracted_claims
    }
