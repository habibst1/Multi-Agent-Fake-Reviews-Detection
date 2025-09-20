# agents/claim_extractor.py

import os
import json
import re
from typing import Dict, List, Any


from groq import Groq

# Initialize Groq client using the environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please check your .env file.")
client = Groq(api_key=GROQ_API_KEY)

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

# ---------- Robust JSON extraction (handles minor formatting) ----------
def extract_json_array(text: str) -> Any:
    """
    Try to parse a JSON array from the model output. Falls back to regex to
    grab the outermost [...] block if the model wrapped it with extra text.
    """
    text = text.strip()
    # Fast path
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        pass
    # Fallback: find the first bracketed JSON array
    match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            return []
    return []

# ---------- Low-level LLM call ----------
def call_llm(prompt: str, model: str, temperature: float) -> Any:
    """Calls the Groq LLM with the given prompt and parameters."""
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        return extract_json_array(content)
    except Exception as e:
        print(f"Error calling LLM: {e}")
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

def extract_claims_and_types(review_text: str, model_extract: str) -> List[Dict[str, Any]]:
    """Extracts claims and their types from the review text."""
    prompt = build_prompt_extract(review_text)
    # Default model if not specified, matching the original code
    model = model_extract if model_extract else "llama-3.3-70b-versatile"
    out = call_llm(prompt, model=model, temperature=0.1)
    return out if isinstance(out, list) else []

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

def classify_nature_and_polarity(
    claims: List[Dict[str, Any]],
    model_classify: str
) -> List[Dict[str, Any]]:
    """Classifies the nature and polarity of the provided claims."""
    if not claims:
        return []
    prompt = build_prompt_classify(claims)
    # Default model if not specified, matching the original code
    model = model_classify if model_classify else "llama-3.1-8b-instant"
    out = call_llm(prompt, model=model, temperature=0.1)
    # Ensure output length matches input length
    if not isinstance(out, list) or len(out) != len(claims):
         print("⚠️ Step 2 returned an unexpected result (empty or length mismatch).")
         # Return input claims with empty nature/polarity if mismatch
         # Or return the output if it's a list (even if wrong length)
         # Original code returned `out or []`. Let's be slightly more robust.
         if isinstance(out, list):
             # Pad or truncate if necessary, or just return as is.
             # For now, let's return the classified list, even if length mismatch,
             # as the pipeline might handle it. Log the issue.
             return out
         else:
             return [] # If output is not a list, return empty list
    return out

# ---------- Convenience: Full two-stage pipeline ----------
def process_review_two_stage(
    review_text: str,
    model_extract: str = "llama-3.3-70b-versatile",
    model_classify: str = "llama-3.1-8b-instant"
) -> List[Dict[str, Any]]:
    """
    Processes the review text through the two-stage claim extraction and classification pipeline.
    """
    print(f"\n--- Processing Review (extract model = {model_extract}) (classify model = {model_classify}) ---\n{review_text}\n")
    # Step 1: Extract claims
    step1 = extract_claims_and_types(review_text, model_extract=model_extract)
    if not step1:
        print("❌ No claims extracted or parsing failed in Step 1.")
        return []
    # print("Step 1 - Extracted claims with claim_type:")
    # for c in step1:
    #     print(json.dumps(c, ensure_ascii=False, indent=2))

    # Step 2: Classify claims
    step2 = classify_nature_and_polarity(step1, model_classify=model_classify)
    # The original code checked for length mismatch and warned.
    # We've handled that in classify_nature_and_polarity.
    # If step2 is empty after classification, return it.
    if not step2:
         print("⚠️ No claims classified or parsing failed in Step 2.")
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

# Example usage (if run directly for testing):
# if __name__ == "__main__":
#     # Ensure .env is loaded if running this script directly
#     from dotenv import load_dotenv # pip install python-dotenv
#     load_dotenv()
#
#     test_review = "The hotel was dirty, but the location was great near the Eiffel Tower."
#     result = extract_claims_pipeline("Test Hotel", test_review)
#     print(json.dumps(result, indent=2))
