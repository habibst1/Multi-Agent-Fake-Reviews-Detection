# main.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import json
import sys
import time
from typing import Dict, Any

import config
import llm
from run_log import setup_logging

setup_logging()

# --- Import the review graph (it imports every agent) ---
try:
    from graph import run_review
    print("✅ All agent modules imported successfully.")
except ImportError as e:
    print(f"❌ Error importing agent modules: {e}")
    exit(1)

def hotel_review_fact_checker(hotel_name: str, hotel_review: str, sequential: bool = False) -> Dict[str, Any]:
    """
    Main pipeline function to fact-check a hotel review.

    Args:
        hotel_name (str): Name of the hotel.
        hotel_review (str): The review text to fact-check.
        sequential (bool): Check one claim at a time instead of in parallel.

    Returns:
        dict: Complete analysis results with verdicts for all claims.
    """
    max_concurrency = 1 if sequential else config.MAX_CONCURRENCY
    return run_review(hotel_name, hotel_review, max_concurrency=max_concurrency)


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

    parser = argparse.ArgumentParser(description="Fact-check a hotel review with the LangGraph pipeline and a local Ollama model.")
    parser.add_argument("--hotel", default=example_hotel_name, help="Name of the hotel (default: the README example).")
    parser.add_argument("--review", default=example_hotel_review, help="Review text to fact-check (default: the README example).")
    parser.add_argument("--sequential", action="store_true", help="Check one claim at a time, like the original loop.")
    args = parser.parse_args()

    # --- Check that every service the pipeline needs is ready ---
    problems = config.missing_api_keys()
    ollama_problem = llm.check_ollama()
    if ollama_problem:
        problems.append(ollama_problem)
    if problems:
        print("❌ Cannot start the pipeline:")
        for problem in problems:
            print(f"   • {problem}")
        sys.exit(1)

    # Run the fact-checker pipeline
    results = hotel_review_fact_checker(args.hotel, args.review, sequential=args.sequential)

    # Display the formatted results
    display_results(results)

    # Optionally, save the full results to a JSON file
    output_filename = f"{args.hotel}_fact_check_results_{int(time.time())}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n💾 Full results saved to '{output_filename}'")
