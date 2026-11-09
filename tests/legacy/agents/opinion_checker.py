# agents/opinion_checker.py

import os
import json
import re
from groq import Groq
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import process
from typing import Dict, List, Any, Tuple, Optional

# --- Configuration and Client Initialization ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please check your .env file.")
client = Groq(api_key=GROQ_API_KEY)

# --- Model Initialization ---
try:
    print("🔧 Loading SentenceTransformer model for opinion checker...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer model loaded.")
except Exception as e:
    print(f"❌ Failed to load SentenceTransformer model: {e}")
    sbert_model = None

# --- Core Opinion Checking Functions ---

def find_closest_json_file(hotel_name: str, folder: str = "Hotel Reviews with claims") -> Tuple[str, str]:
    """
    Finds the JSON file in the folder that best matches the hotel name.
    Returns a tuple of (full_file_path, matched_hotel_name_for_display).
    """
    # Normalize hotel name for matching
    normalized_hotel_name = hotel_name.lower().replace(" ", "_")
    
    # Check if folder exists
    if not os.path.exists(folder):
        print(f"⚠️ Review data folder '{folder}' not found.")
        return "", hotel_name # Return original name if folder missing

    try:
        json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
    except Exception as e:
        print(f"⚠️ Error reading review data folder '{folder}': {e}")
        return "", hotel_name

    if not json_files:
        print(f"⚠️ No JSON files found in review data folder '{folder}'.")
        return "", hotel_name

    # Use fuzzy matching to find the best file
    try:
        best_match, score = process.extractOne(normalized_hotel_name, json_files)
        matched_file_path = os.path.join(folder, best_match)
        # Convert back to display name
        matched_display_name = best_match.replace('.json', '').replace('_', ' ')
        print(f"📂 Matched hotel '{hotel_name}' to file: {best_match} (score={score})")
        return matched_file_path, matched_display_name
    except Exception as e:
        print(f"⚠️ Error during fuzzy matching for hotel '{hotel_name}': {e}")
        # Fallback: try exact match or first file
        for f in json_files:
            if normalized_hotel_name in f.lower() or f.lower().startswith(normalized_hotel_name):
                 matched_file_path = os.path.join(folder, f)
                 matched_display_name = f.replace('.json', '').replace('_', ' ')
                 print(f"📂 Fallback match for hotel '{hotel_name}' to file: {f}")
                 return matched_file_path, matched_display_name
        # If no good match, return the first file or empty
        fallback_file = json_files[0] if json_files else ""
        matched_display_name = fallback_file.replace('.json', '').replace('_', ' ') if fallback_file else hotel_name
        print(f"📂 No good match found, using fallback file: {fallback_file}")
        return os.path.join(folder, fallback_file) if fallback_file else "", matched_display_name


def extract_claims(reviews: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Extracts positive and negative claims from a list of reviews.
    Each claim object includes text, type, nature, and polarity.
    """
    positive_claims = []
    negative_claims = []
    for review in reviews:
        if "claims" in review:
            for claim in review["claims"]:
                text = claim.get("claim_text", "").strip()
                # Basic cleaning
                text = re.sub(r'\s+', ' ', text).strip()
                polarity = claim.get("claim_polarity", "").lower()
                # Filter out very short or generic text
                if len(text) > 10 and "no comments" not in text.lower():
                    claim_obj = {
                        "text": text,
                        "type": claim.get("claim_type", ""),
                        "nature": claim.get("claim_nature", ""),
                        "polarity": polarity
                    }
                    if polarity == "positive":
                        positive_claims.append(claim_obj)
                    elif polarity == "negative":
                        negative_claims.append(claim_obj)
    print(f"🧹 Extracted {len(positive_claims)} positive claims and {len(negative_claims)} negative claims")
    return positive_claims, negative_claims


def find_top_similar_claims(target_claim_text: str, claims_list: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    Finds the top_k most similar claims to the target claim text using SBERT embeddings.
    """
    if not claims_list or not sbert_model:
        return []

    print(f"🔍 Finding top {top_k} similar claims to the target claim...")
    claim_texts = [c["text"] for c in claims_list]
    
    try:
        claim_emb = sbert_model.encode(target_claim_text, convert_to_tensor=True)
        review_embs = sbert_model.encode(claim_texts, convert_to_tensor=True)
        scores = util.cos_sim(claim_emb, review_embs)[0]
        top_indices = scores.argsort(descending=True)[:top_k]
        
        results = []
        for i in top_indices:
            original_claim = claims_list[i]
            results.append({
                "text": original_claim["text"],
                "type": original_claim["type"],
                "nature": original_claim["nature"],
                "polarity": original_claim["polarity"],
                "similarity": float(scores[i])
            })
        return results
    except Exception as e:
        print(f"⚠️ Error finding similar claims: {e}")
        return []


def classify_review_stance(user_claim: str, review_claim: str) -> Tuple[str, float, Optional[bool]]:
    """
    Classifies the stance of a review claim towards a user claim using the Groq LLM.
    Returns a tuple: (stance: str, confidence: float, had_api_error: Optional[bool]).
    Stance is one of 'supports', 'contradicts', 'neutral'.
    Confidence is between 0.0 and 1.0.
    had_api_error is True if there was an API error, False/None otherwise.
    """
    prompt = f"""
User Claim: "{user_claim}"
Review Claim: "{review_claim}"
Classify the review claim:
- "supports": The review claim supports or agrees with the user claim
- "contradicts": The review claim contradicts or disagrees with the user claim  
- "neutral": The review claim doesn't relate enough to make a stance judgment or is neither supporting nor contradicting the user claim
Respond with ONLY the classification and a confidence score from 0.0 to 1.0, separated by a comma.
Format: classification,confidence_score
Example: supports,0.85
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", # Using Groq's Llama model
            temperature=0.1,
        )
        result_text = chat_completion.choices[0].message.content.strip().lower()
        # Parse the response
        parts = result_text.split(',')
        if len(parts) >= 2:
            stance = parts[0].strip()
            try:
                confidence = float(parts[1].strip())
            except ValueError:
                confidence = 0.5 # Default if parsing fails
            
            # Ensure stance is one of the expected values
            if stance not in ['supports', 'contradicts', 'neutral']:
                # Fallback parsing if format is different
                if 'support' in result_text:
                    stance = 'supports'
                elif 'contradict' in result_text:
                    stance = 'contradicts'
                else:
                    stance = 'neutral'
                confidence = 0.5  # Default confidence
            
            # Override to neutral if confidence is below 0.75 (as per original logic)
            if confidence < 0.75:
                stance = 'neutral'
                
        else:
            # Fallback parsing
            if 'support' in result_text:
                stance = 'supports'
            elif 'contradict' in result_text:
                stance = 'contradicts'
            else:
                stance = 'neutral'
            confidence = 0.5
            
        return stance, confidence, None  # Third value indicates error status (None = no error)

    except Exception as e:
        print(f"⚠️ Error with Groq API in classify_review_stance: {e}")
        # Return neutral with low confidence and mark as error
        return "neutral", 0.3, True


def determine_verdict(support: int, contradict: int, neutral: int) -> Tuple[str, float]:
    """
    Determines the final verdict and confidence based on stance counts.
    Returns: (verdict: str, confidence: float)
    Verdict is one of "true", "false", "unable_to_verify".
    """
    total_relevant = support + contradict
    # Handle cases with insufficient data
    if total_relevant == 0:
        return "unable_to_verify", 0.1
    # Calculate support ratio and total evidence
    support_ratio = support / total_relevant
    total_evidence = total_relevant + (neutral * 0.2)  # Neutral claims contribute slightly
    # 40-60% range is now INCONCLUSIVE instead of binary true/false
    if support_ratio >= 0.6:
        verdict = "true"  # LIKELY TRUE maps to true
        confidence = min(0.9, 0.6 + (support_ratio * 0.3) + (total_evidence / 20))
    elif support_ratio <= 0.4:
        verdict = "false"  # LIKELY FALSE maps to false
        confidence = min(0.9, 0.6 + ((1 - support_ratio) * 0.3) + (total_evidence / 20))
    else:
        verdict = "unable_to_verify"  # INCONCLUSIVE maps to unable_to_verify
        # Confidence decreases as we approach 50/50 split
        distance_from_50 = abs(support_ratio - 0.5)
        confidence = 0.4 + (distance_from_50 * 0.8) + (total_evidence / 25)
    # Cap confidence and ensure minimum
    confidence = max(0.1, min(0.95, confidence))
    return verdict, confidence


def check_opinion_claim(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to evaluate a single opinion claim using other hotel reviews.
    
    Expected input format:
    {
        "claim_id": "claim_1",
        "claim_text": "The staff was incredibly friendly.",
        "hotel_name": "Grand Plaza Hotel"
    }

    Returns:
    {
        "claim_id": "claim_1",
        "claim_text": "...",
        "verdict": "true|false|unable_to_verify",
        "aggregation_results": {...},
        "matched_hotel": "...",
        "confidence_score": 0.85,
        "evidence": {...}
    }
    """
    try:
        claim_id = input_data.get("claim_id")
        claim_text = input_data.get("claim_text")
        hotel_name = input_data.get("hotel_name")
        
        if not all([claim_id, claim_text, hotel_name]):
             return {
                "claim_id": claim_id or "unknown",
                "claim_text": claim_text or "",
                "verdict": "unable_to_verify",
                "aggregation_results": {
                    "supports": 0,
                    "contradicts": 0,
                    "neutral": 0
                },
                "matched_hotel": hotel_name or "",
                "confidence_score": 0.0,
                "evidence": {"error": "Missing required input parameters"}
            }

        print(f"\n=== 📋 Evaluating Opinion Claim: '{claim_text}' @ Hotel: '{hotel_name}' ===")
        
        if not sbert_model:
             return {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "verdict": "unable_to_verify",
                "aggregation_results": {
                    "supports": 0,
                    "contradicts": 0,
                    "neutral": 0
                },
                "matched_hotel": hotel_name,
                "confidence_score": 0.0,
                "evidence": {"error": "SentenceTransformer model not loaded"}
            }
        
        json_file_path, matched_hotel = find_closest_json_file(hotel_name)
        
        if not json_file_path or not os.path.exists(json_file_path):
            print(f"❌ File not found: {json_file_path}")
            return {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "verdict": "unable_to_verify",
                "aggregation_results": {
                    "supports": 0,
                    "contradicts": 0,
                    "neutral": 0
                },
                "matched_hotel": hotel_name, # Use original if matching failed
                "confidence_score": 0.0,
                "evidence": {"error": f"No JSON file found for '{hotel_name}'"}
            }

        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                reviews = json.load(f)
        except Exception as e:
            print(f"❌ Error reading JSON file '{json_file_path}': {e}")
            return {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "verdict": "unable_to_verify",
                "aggregation_results": {
                    "supports": 0,
                    "contradicts": 0,
                    "neutral": 0
                },
                "matched_hotel": matched_hotel,
                "confidence_score": 0.0,
                "evidence": {"error": f"Failed to read review data: {str(e)}"}
            }

        # Extract claims from reviews
        positive_claims, negative_claims = extract_claims(reviews)
        if not positive_claims and not negative_claims:
            print("❌ No valid claims found in review data")
            return {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "verdict": "unable_to_verify",
                "aggregation_results": {
                    "supports": 0,
                    "contradicts": 0,
                    "neutral": 0
                },
                "matched_hotel": matched_hotel,
                "confidence_score": 0.0,
                "evidence": {"error": "No valid claims found in review data"}
            }

        # Process both positive and negative claims separately
        support = contradict = neutral = 0
        api_error_count = 0
        all_evidence = []
        
        # Process positive claims
        print("\n🔍 Analyzing POSITIVE claims...")
        if positive_claims:
            top_positive_claims = find_top_similar_claims(claim_text, positive_claims)
            for idx, claim_data in enumerate(top_positive_claims, start=1):
                review_claim = claim_data["text"]
                print(f"\n🔹 Positive Claim {idx} (Similarity: {claim_data['similarity']:.4f}):\n\"{review_claim[:100]}...\"\n")
                stance, conf, had_error = classify_review_stance(claim_text, review_claim)
                print(f"➡️ Classified as: {stance.upper()} (Confidence: {conf:.2f})")
                if had_error:
                    api_error_count += 1
                    print("⚠️ This classification had an API error")
                all_evidence.append({
                    "text": review_claim,
                    "similarity_score": claim_data["similarity"],
                    "stance": stance,
                    "confidence": conf,
                    "polarity": "positive"
                })
                if stance == "supports":
                    support += 1
                elif stance == "contradicts":
                    contradict += 1
                else:
                    neutral += 1

        # Process negative claims
        print("\n🔍 Analyzing NEGATIVE claims...")
        if negative_claims:
            top_negative_claims = find_top_similar_claims(claim_text, negative_claims)
            for idx, claim_data in enumerate(top_negative_claims, start=1):
                review_claim = claim_data["text"]
                print(f"\n🔹 Negative Claim {idx} (Similarity: {claim_data['similarity']:.4f}):\n\"{review_claim[:100]}...\"\n")
                stance, conf, had_error = classify_review_stance(claim_text, review_claim)
                print(f"➡️ Classified as: {stance.upper()} (Confidence: {conf:.2f})")
                if had_error:
                    api_error_count += 1
                    print("⚠️ This classification had an API error")
                all_evidence.append({
                    "text": review_claim,
                    "similarity_score": claim_data["similarity"],
                    "stance": stance,
                    "confidence": conf,
                    "polarity": "negative"
                })
                if stance == "supports":
                    support += 1
                elif stance == "contradicts":
                    contradict += 1
                else:
                    neutral += 1

        print(f"\n✅ Summary for Claim: '{claim_text}'")
        print(f"✔️ Support: {support} | ❌ Contradict: {contradict} | ⚖️ Neutral: {neutral}")
        print(f"⚠️ API Errors: {api_error_count}")

        # Updated verdict determination logic
        if api_error_count == (len(positive_claims) + len(negative_claims)) and api_error_count > 0:
            verdict = "unable_to_verify"
            confidence = 0.0
        elif api_error_count > 0 and support + contradict == 0:
            verdict = "unable_to_verify"
            confidence = 0.0
        elif support + contradict == 0:
            verdict = "unable_to_verify"
            confidence = 0.3
        else:
            verdict_str, confidence = determine_verdict(support, contradict, neutral)
            verdict = verdict_str  # Keep as string: "true", "false", or "unable_to_verify"

        print(f"⚖️ VERDICT: {'TRUE' if verdict == 'true' else 'FALSE' if verdict == 'false' else 'UNABLE_TO_VERIFY'}")
        print(f"🎯 CONFIDENCE: {confidence:.2f}")

        return {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "verdict": verdict,  
            "aggregation_results": {
                "supports": support,
                "contradicts": contradict,
                "neutral": neutral,
                "api_errors": api_error_count
            },
            "matched_hotel": matched_hotel,
            "confidence_score": confidence,
            "evidence": {
                "matched_reviews": all_evidence,  
                "api_error_count": api_error_count
            }
        }

    except Exception as e:
        print(f"❌ Error in check_opinion_claim: {e}")
        return {
            "claim_id": input_data.get("claim_id", "unknown"),
            "claim_text": input_data.get("claim_text", ""),
            "verdict": "unable_to_verify",
            "aggregation_results": {
                "supports": 0,
                "contradicts": 0,
                "neutral": 0
            },
            "matched_hotel": input_data.get("hotel_name", ""),
            "confidence_score": 0.0,
            "evidence": {"error": f"Processing error: {str(e)}"}
        }

# Example usage (if run directly for testing):
# if __name__ == "__main__":
#     # Ensure .env is loaded if running this script directly
#     # from dotenv import load_dotenv # pip install python-dotenv
#     # load_dotenv()
#     # test_data = {
#     #     "claim_id": "test_claim_1",
#     #     "claim_text": "The staff was incredibly friendly.",
#     #     "hotel_name": "Grand Plaza Hotel"
#     # }
#     # result = check_opinion_claim(test_data)
#     # print(json.dumps(result, indent=2))
#     pass
