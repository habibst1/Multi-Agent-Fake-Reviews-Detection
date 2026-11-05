# main.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv  

# --- Load environment variables ---
load_dotenv()

# --- Import agent functions ---
try:
    from agents.claim_extractor import extract_claims_pipeline
    from agents.location_checker import check_location_claim
    from agents.web_search_checker import check_web_search_claim
    from agents.opinion_checker import check_opinion_claim
    # Import the final aggregation function
    from agents.final_judge import aggregate_claim_results
    print("✅ All agent modules imported successfully.")
except ImportError as e:
    print(f"❌ Error importing agent modules: {e}")
    exit(1)

def hotel_review_fact_checker(hotel_name: str, hotel_review: str) -> Dict[str, Any]:
    """
    Main pipeline function to fact-check a hotel review.

    Args:
        hotel_name (str): Name of the hotel.
        hotel_review (str): The review text to fact-check.

    Returns:
        dict: Complete analysis results with verdicts for all claims.
    """
    print("🏨 HOTEL REVIEW FACT-CHECKING PIPELINE")
    print("=" * 60)
    print(f"🏢 Hotel: {hotel_name}")
    # Truncate review for display if too long
    display_review = hotel_review if len(hotel_review) <= 100 else hotel_review[:97] + "..."
    print(f"📝 Review: {display_review}")
    print("=" * 60)

    # --- STEP 1: Extract claims from the review ---
    print("\n📋 STEP 1: EXTRACTING CLAIMS")
    print("-" * 30)
    try:
        claims_result = extract_claims_pipeline(hotel_name, hotel_review)
        claims_data = claims_result.get("claims", [])
        if not claims_data:
            error_msg = "No claims extracted from the review."
            print(f"❌ {error_msg}")
            return {
                "hotel_name": hotel_name,
                "hotel_review": hotel_review,
                "overall_verdict": "unable_to_verify",
                "claims_results": [],
                "summary": {
                    "total_claims": 0,
                    "true_claims": 0,
                    "false_claims": 0,
                    "unable_to_verify_claims": 0,
                    "accuracy_rate": 0.0
                },
                "error": error_msg
            }
        print(f"✅ Extracted {len(claims_data)} claims:")
        for i, claim in enumerate(claims_data, 1):
            print(f"  {i}. [{claim.get('claim_type', 'unknown')}] {claim.get('claim_text', '')}")
            print(f"     Nature: {claim.get('claim_nature', 'unknown')}")
    except Exception as e:
        error_msg = f"Claim extraction failed: {str(e)}"
        print(f"❌ Error in claim extraction: {e}")
        return {
            "hotel_name": hotel_name,
            "hotel_review": hotel_review,
            "overall_verdict": "unable_to_verify",
            "claims_results": [],
            "summary": {
                "total_claims": 0,
                "true_claims": 0,
                "false_claims": 0,
                "unable_to_verify_claims": 0,
                "accuracy_rate": 0.0
            },
            "error": error_msg
        }

    # --- STEP 2: Process each claim through appropriate checker ---
    print(f"\n🔍 STEP 2: FACT-CHECKING {len(claims_data)} CLAIMS")
    print("-" * 40)
    checker_results = [] # Store results from individual checkers

    for i, claim in enumerate(claims_data, 1):
        claim_text = claim.get("claim_text", "")
        claim_type = claim.get("claim_type", "")
        claim_nature = claim.get("claim_nature", "")
        claim_id = f"claim_{i}"

        print(f"\n🔸 Processing Claim {i}: '{claim_text}'")
        print(f"   Type: {claim_type} | Nature: {claim_nature}")

        # Prepare input for checkers
        checker_input = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "hotel_name": hotel_name,
            "claim_type": claim_type
        }

        result = None
        checker_used = "unknown"

        try:
            # Route to appropriate checker based on claim type and nature
            if claim_type == "location_general":
                if claim_nature == "factual":
                    print("   🌍 → Location Checker (Factual Location)")
                    result = check_location_claim(checker_input)
                    checker_used = "location"
                else:  # opinion
                    print("   💭 → Review Checker (Opinion about Location)")
                    result = check_opinion_claim(checker_input)
                    checker_used = "review"
            elif claim_nature == "factual" and claim_type != "location_general":
                print("   🔍 → Web Search Checker")
                result = check_web_search_claim(checker_input)
                checker_used = "web_search"
            elif claim_nature == "opinion" and claim_type != "location_general":
                print("   💭 → Review Checker")
                result = check_opinion_claim(checker_input)
                checker_used = "review"
            else:
                print("   ❓ → Unknown claim type/nature, defaulting to Web Search")
                result = check_web_search_claim(checker_input)
                checker_used = "web_search"

            # Ensure result has essential fields
            if result is None:
                 result = {
                    "claim_id": claim_id,
                    "verdict": "unable_to_verify",
                    "confidence_score": 0.0,
                    "evidence": {"error": "Checker returned None"},
                }

            # Add checker_used information to the result
            result["checker_used"] = checker_used

            checker_results.append(result)
            print(f"   ✅ Checker '{checker_used}' finished for claim {i}.")

        except Exception as e:
            print(f"   ❌ Error processing claim with {checker_used} checker: {str(e)}")
            checker_results.append({
                "claim_id": claim_id,
                "verdict": "unable_to_verify",
                "confidence_score": 0.0,
                "checker_used": checker_used,
                "evidence": {"error": f"Processing error in {checker_used} checker: {str(e)}"},
            })

        # --- Add delay between API calls to avoid rate limits ---
        if i < len(claims_data):
            delay_seconds = 30 
            print(f"   ⏱️ Waiting {delay_seconds} seconds before next claim...")
            time.sleep(delay_seconds)

    # --- STEP 3: Final Judgment and Aggregation ---
    print(f"\n🏛️ STEP 3: FINAL JUDGMENT AND AGGREGATION")
    print("-" * 45)
    try:
        # Use the imported final judge function
        final_results = aggregate_claim_results(hotel_name, hotel_review, claims_data, checker_results)
        return final_results
    except Exception as e:
        error_msg = f"Error in final judgment: {str(e)}"
        print(f"❌ {error_msg}")
        # Fallback aggregation if final judge fails
        return {
            "hotel_name": hotel_name,
            "hotel_review": hotel_review,
            "overall_verdict": "unable_to_verify",
            "claims_results": checker_results, # Return raw checker results
            "summary": {
                "total_claims": len(claims_data),
                "true_claims": 0,
                "false_claims": 0,
                "unable_to_verify_claims": len(claims_data),
                "accuracy_rate": 0.0
            },
            "error": error_msg
        }


def display_results(results: Dict[str, Any]):
    """
    Display the results in a formatted way.
    """
    print("\n" + "="*80)
    print("🎯 FINAL FACT-CHECKING RESULTS")
    print("="*80)
    print(f"\n🏨 Hotel: {results['hotel_name']}")
    display_review = results['hotel_review'] if len(results['hotel_review']) <= 100 else results['hotel_review'][:97] + "..."
    print(f"📝 Review: {display_review}")

    if "error" in results and results["error"]:
        print(f"\n❌ Pipeline Error: {results['error']}")
        return

    summary = results.get('summary', {})
    overall_verdict = results.get('overall_verdict', 'unable_to_verify')

    # Determine display for overall verdict
    if overall_verdict == "true":
        overall_symbol = "✅"
        overall_display = "TRUE"
    elif overall_verdict == "false":
        overall_symbol = "❌"
        overall_display = "FALSE"
    else: # "unable_to_verify"
        overall_symbol = "❓"
        overall_display = "UNABLE_TO_VERIFY"

    print(f"\n📊 Summary:")
    print(f"   • Total Claims: {summary.get('total_claims', 0)}")
    print(f"   • True Claims: {summary.get('true_claims', 0)}")
    print(f"   • False Claims: {summary.get('false_claims', 0)}")
    print(f"   • Unable to Verify Claims: {summary.get('unable_to_verify_claims', 0)}")
    print(f"   • Accuracy Rate (of verifiable): {summary.get('accuracy_rate', 0.0)}%")
    print(f"\n🏆 Overall Review Verdict: {overall_symbol} {overall_display}")

    print(f"\n🔍 Detailed Claim Results:")
    claims_results = results.get('claims_results', [])
    if not claims_results:
        print("   No detailed results available.")
        return

    for claim_result in claims_results:
        claim_text = claim_result.get('claim_text', 'N/A')
        verdict = claim_result.get('verdict', 'unable_to_verify')
        confidence = claim_result.get('confidence_score', 0.0)
        checker_used = claim_result.get('checker_used', 'unknown')

        # Determine display for claim verdict
        if verdict == "true":
            verdict_symbol = "✅"
            verdict_display = "TRUE"
        elif verdict == "false":
            verdict_symbol = "❌"
            verdict_display = "FALSE"
        else: # "unable_to_verify"
            verdict_symbol = "❓"
            verdict_display = "UNABLE_TO_VERIFY"

        # Truncate claim text for display
        display_claim = claim_text if len(claim_text) <= 80 else claim_text[:77] + "..."

        print(f"\n{verdict_symbol} Claim: \"{display_claim}\"")
        print(f"   Verdict: {verdict_display} (Confidence: {confidence:.2f})")
        print(f"   Checked by: {checker_used.title()} Checker")

        # Display reasoning or key evidence if available
        evidence = claim_result.get('evidence', {})
        if 'reasoning' in evidence:
            print(f"   💭 Reasoning: {evidence['reasoning']}")
        elif 'recommendation' in evidence: # From web search checker
             print(f"   💭 Recommendation: {evidence['recommendation']}")
        elif 'matched_reviews' in evidence and isinstance(evidence['matched_reviews'], list): # From opinion checker
            matched_count = len(evidence['matched_reviews'])
            print(f"   📈 Matched {matched_count} similar claims in reviews.")

        # Display aggregation results if from opinion checker
        additional_data = claim_result.get('additional_data', {})
        if 'aggregation_results' in additional_data:
            agg = additional_data['aggregation_results']
            print(f"   📊 Review Analysis: {agg.get('supports', 0)} support, {agg.get('contradicts', 0)} contradict, {agg.get('neutral', 0)} neutral")

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: A review with mixed claims
    example_hotel_name = "Burj Al Arab"
    example_hotel_review = (
        "Overpriced hotel in Dubai. Located 15 minutes from Dubai Mall. Staff service was excellent but rooms felt outdated. Food quality was poor. 20 Kilometers from the airport"
    )

    # Run the fact-checker pipeline
    results = hotel_review_fact_checker(example_hotel_name, example_hotel_review)

    # Display the formatted results
    display_results(results)

    # Optionally, save the full results to a JSON file
    output_filename = f"{example_hotel_name}_fact_check_results_{int(time.time())}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n💾 Full results saved to '{output_filename}'")
