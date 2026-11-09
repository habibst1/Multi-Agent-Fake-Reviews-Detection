# agents/final_judge.py

import time
from typing import Dict, List, Any, Union

def aggregate_claim_results(
    hotel_name: str,
    hotel_review: str,
    claims_data: List[Dict[str, Any]],
    checker_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Final judge function to aggregate individual claim check results and determine
    the overall review verdict.

    Args:
        hotel_name (str): The name of the hotel.
        hotel_review (str): The original hotel review text.
        claims_data (List[Dict]): The list of claims extracted by the claim extractor.
                                  Each dict contains 'claim_text', 'claim_type', 'claim_nature'.
        checker_results (List[Dict]): The list of results from individual checkers.
                                      Each dict contains 'claim_id', 'verdict', 'confidence_score',
                                      'checker_used', 'evidence', 'additional_data'.

    Returns:
        dict: A dictionary containing the overall analysis results.
              {
                  "hotel_name": "...",
                  "hotel_review": "...",
                  "overall_verdict": "true|false|unable_to_verify",
                  "claims_results": [...], # Detailed results for each claim
                  "summary": {
                      "total_claims": int,
                      "true_claims": int,
                      "false_claims": int,
                      "unable_to_verify_claims": int,
                      "accuracy_rate": float
                  }
                  # Potentially an "error" key if something went wrong in this stage
              }
    """
    print(f"\n🏛️ STEP 3: FINAL JUDGMENT AND AGGREGATION")
    print("-" * 45)

    # Combine claims data with checker results
    # This assumes claim_id can be used to match them, or they are in the same order.
    # The structure provided suggests checker_results already contain claim details.
    # Let's reconstruct claims_results based on checker_results for clarity and completeness.
    claims_results = []
    true_claims = 0
    false_claims = 0
    unable_to_verify_claims = 0

    # Create a mapping of checker results by claim_id for easy lookup
    checker_result_map = {res.get("claim_id"): res for res in checker_results}

    # Iterate through the original claims data to build the final claims_results list
    for i, claim in enumerate(claims_data, 1):
        claim_id = f"claim_{i}" # Reconstruct claim_id if needed, or use from claim if available
        claim_text = claim.get("claim_text", "")
        claim_type = claim.get("claim_type", "")
        claim_nature = claim.get("claim_nature", "")

        # Find the corresponding checker result
        checker_result = checker_result_map.get(claim_id)

        if not checker_result:
            print(f"   ⚠️  No checker result found for claim ID: {claim_id}. Marking as unable to verify.")
            unable_to_verify_claims += 1
            claims_results.append({
                "claim_id": claim_id,
                "claim_text": claim_text,
                "claim_type": claim_type,
                "claim_nature": claim_nature,
                "verdict": "unable_to_verify",
                "checker_used": "missing",
                "confidence_score": 0.0,
                "evidence": {"error": "Checker result missing"},
                "additional_data": {}
            })
            continue

        # Extract verdict and confidence from checker result
        verdict = checker_result.get("verdict", "unable_to_verify")
        confidence = checker_result.get("confidence_score", 0.0)
        checker_used = checker_result.get("checker_used", "unknown")

        # Update counters
        if verdict == "true":
            true_claims += 1
            verdict_symbol = "✅"
            verdict_display = "TRUE"
        elif verdict == "false":
            false_claims += 1
            verdict_symbol = "❌"
            verdict_display = "FALSE"
        else: # "unable_to_verify"
            unable_to_verify_claims += 1
            verdict_symbol = "❓"
            verdict_display = "UNABLE_TO_VERIFY"

        print(f"   {verdict_symbol} Claim {i} Verdict: {verdict_display} (Confidence: {confidence:.2f})")

        # Append the checker result (which should already contain all necessary details)
        # Ensure it has the core claim metadata
        final_claim_result = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "claim_type": claim_type,
            "claim_nature": claim_nature,
            "verdict": verdict,
            "checker_used": checker_used,
            "confidence_score": confidence,
            "evidence": checker_result.get("evidence", {}),
            "additional_data": checker_result.get("additional_data", {})
        }
        claims_results.append(final_claim_result)


    # --- Calculate Overall Verdict ---
    total_claims = len(claims_data)
    verifiable_claims = true_claims + false_claims

    # Overall verdict logic: Based on majority of verifiable claims
    # If no claims were verifiable, overall verdict is 'unable_to_verify'
    if verifiable_claims == 0:
        overall_verdict = "unable_to_verify"
        print(f"\n⚖️  Overall Verdict Logic: No verifiable claims found.")
    elif true_claims > false_claims:
        overall_verdict = "true" # Mostly True
        print(f"\n⚖️  Overall Verdict Logic: More true claims ({true_claims}) than false ({false_claims}).")
    elif false_claims > true_claims:
        overall_verdict = "false" # Mostly False
        print(f"\n⚖️  Overall Verdict Logic: More false claims ({false_claims}) than true ({true_claims}).")
    else:
        # Tie case: true_claims == false_claims and both > 0
        overall_verdict = "unable_to_verify" # Inconclusive
        print(f"\n⚖️  Overall Verdict Logic: Equal true and false claims ({true_claims}). Inconclusive.")

    # Calculate accuracy rate
    accuracy_rate = round((true_claims / verifiable_claims * 100), 1) if verifiable_claims > 0 else 0.0

    # Compile summary
    summary = {
        "total_claims": total_claims,
        "true_claims": true_claims,
        "false_claims": false_claims,
        "unable_to_verify_claims": unable_to_verify_claims,
        "accuracy_rate": accuracy_rate
    }

    # Compile final results
    final_results = {
        "hotel_name": hotel_name,
        "hotel_review": hotel_review,
        "overall_verdict": overall_verdict,
        "claims_results": claims_results,
        "summary": summary
    }

    # --- Display Summary (can be moved to main.py if preferred) ---
    print(f"\n📊 Final Summary:")
    print(f"   • Total Claims: {summary['total_claims']}")
    print(f"   • True Claims: {summary['true_claims']}")
    print(f"   • False Claims: {summary['false_claims']}")
    print(f"   • Unable to Verify Claims: {summary['unable_to_verify_claims']}")
    print(f"   • Accuracy Rate (of verifiable): {summary['accuracy_rate']}%")

    overall_verdict_display = overall_verdict.upper()
    if overall_verdict == "true":
        overall_symbol = "✅"
    elif overall_verdict == "false":
        overall_symbol = "❌"
    else: # "unable_to_verify"
        overall_symbol = "❓"

    print(f"\n🏆 Overall Review Verdict: {overall_symbol} {overall_verdict_display}")

    return final_results

# Example usage (if run directly for testing):
# if __name__ == "__main__":
#     # Example data structures (simplified)
#     test_hotel_name = "Test Hotel"
#     test_hotel_review = "This is a test review."
#     test_claims_data = [
#         {"claim_text": "The hotel has a pool.", "claim_type": "hotel_design_and_features", "claim_nature": "factual"},
#         {"claim_text": "The staff was friendly.", "claim_type": "service_general", "claim_nature": "opinion"}
#     ]
#     test_checker_results = [
#         {
#             "claim_id": "claim_1",
#             "verdict": "true",
#             "confidence_score": 0.95,
#             "checker_used": "web_search",
#             "evidence": {"sources": ["http://example.com"]},
#             "additional_data": {}
#         },
#         {
#             "claim_id": "claim_2",
#             "verdict": "unable_to_verify",
#             "confidence_score": 0.2,
#             "checker_used": "review",
#             "evidence": {"matched_reviews": []},
#             "additional_data": {"aggregation_results": {"supports": 0, "contradicts": 0, "neutral": 5}}
#         }
#     ]
#     result = aggregate_claim_results(
#         test_hotel_name,
#         test_hotel_review,
#         test_claims_data,
#         test_checker_results
#     )
#     import json
#     print(json.dumps(result, indent=2))
