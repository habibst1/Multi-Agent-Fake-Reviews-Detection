# fake_review_detection/graph.py

import logging
from pathlib import Path
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from fake_review_detection import config
from fake_review_detection.agents.claim_extractor import extract_claims_pipeline
from fake_review_detection.agents.final_judge import aggregate_claim_results
from fake_review_detection.agents.location_checker import check_location_claim, location_graph
from fake_review_detection.agents.opinion_checker import check_opinion_claim, opinion_graph
from fake_review_detection.agents.web_search_checker import check_web_search_claim, web_search_graph
from fake_review_detection.run_log import redact_value, start_scope, with_reasons
from fake_review_detection.state import ClaimTask, ReviewState

logger = logging.getLogger(__name__)

# Graph node -> (name reported in "checker_used", checker function)
CHECKERS = {
    "location_check": ("location", check_location_claim),
    "web_check": ("web_search", check_web_search_claim),
    "opinion_check": ("review", check_opinion_claim),
}


def _failed_review(hotel_name: str, hotel_review: str, error_msg: str) -> Dict[str, Any]:
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


def extract_claims(state: ReviewState) -> Dict[str, Any]:
    """STEP 1: extract the claims from the review."""
    start_scope(None)
    hotel_name, hotel_review = state["hotel_name"], state["hotel_review"]
    logger.info("🏨 HOTEL REVIEW FACT-CHECKING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"🏢 Hotel: {hotel_name}")
    # Truncate review for display if too long
    display_review = hotel_review if len(hotel_review) <= 100 else hotel_review[:97] + "..."
    logger.info(f"📝 Review: {display_review}")
    logger.info("=" * 60)

    logger.info("\n📋 STEP 1: EXTRACTING CLAIMS")
    logger.info("-" * 30)
    try:
        claims_result = extract_claims_pipeline(hotel_name, hotel_review)
        claims_data = claims_result.get("claims", [])
        if not claims_data:
            error_msg = with_reasons("No claims extracted from the review.")
            logger.info(f"❌ {error_msg}")
            return {"final_results": _failed_review(hotel_name, hotel_review, error_msg)}
        logger.info(f"✅ Extracted {len(claims_data)} claims:")
        for i, claim in enumerate(claims_data, 1):
            logger.info(f"  {i}. [{claim.get('claim_type', 'unknown')}] {claim.get('claim_text', '')}")
            logger.info(f"     Nature: {claim.get('claim_nature', 'unknown')}")
    except Exception as e:
        error_msg = f"Claim extraction failed: {str(e)}"
        logger.error(f"❌ Error in claim extraction: {e}")
        return {"final_results": _failed_review(hotel_name, hotel_review, error_msg)}

    logger.info(f"\n🔍 STEP 2: FACT-CHECKING {len(claims_data)} CLAIMS")
    logger.info("-" * 40)
    return {"claims": claims_data}


def route_claims(state: ReviewState):
    """Send every claim to its checker. All the checks run at the same time."""
    if state.get("final_results") is not None:
        return END
    sends = []
    for i, claim in enumerate(state["claims"], 1):
        claim_type = claim.get("claim_type", "")
        claim_nature = claim.get("claim_nature", "")

        # Route to appropriate checker based on claim type and nature
        if claim_type == "location_general":
            if claim_nature == "factual":
                node, route_message = "location_check", "   🌍 → Location Checker (Factual Location)"
            else:  # opinion
                node, route_message = "opinion_check", "   💭 → Review Checker (Opinion about Location)"
        elif claim_nature == "factual" and claim_type != "location_general":
            node, route_message = "web_check", "   🔍 → Web Search Checker"
        elif claim_nature == "opinion" and claim_type != "location_general":
            node, route_message = "opinion_check", "   💭 → Review Checker"
        else:
            node, route_message = "web_check", "   ❓ → Unknown claim type/nature, defaulting to Web Search"

        task: ClaimTask = {
            "index": i,
            "claim_id": f"claim_{i}",
            "claim_text": claim.get("claim_text", ""),
            "hotel_name": state["hotel_name"],
            "claim_type": claim_type,
            "claim_nature": claim_nature,
            "route_message": route_message,
        }
        sends.append(Send(node, task))
    return sends


def _checker_node(node_name: str):
    checker_used, check = CHECKERS[node_name]

    def run_checker(task: ClaimTask) -> Dict[str, Any]:
        start_scope(task["claim_id"])
        logger.info(f"\n🔸 Processing Claim {task['index']}: '{task['claim_text']}'")
        logger.info(f"   Type: {task['claim_type']} | Nature: {task['claim_nature']}")
        logger.info(task["route_message"])

        # Prepare input for checkers
        checker_input = {key: task[key] for key in ("claim_id", "claim_text", "hotel_name", "claim_type")}
        try:
            result = check(checker_input)
            # Ensure result has essential fields
            if result is None:
                result = {
                    "claim_id": task["claim_id"],
                    "verdict": "unable_to_verify",
                    "confidence_score": 0.0,
                    "evidence": {"error": "Checker returned None"},
                }
            # Add checker_used information to the result
            result["checker_used"] = checker_used
            logger.info(f"   ✅ Checker '{checker_used}' finished for claim {task['index']}.")
        except Exception as e:
            logger.error(f"   ❌ Error processing claim with {checker_used} checker: {str(e)}")
            result = {
                "claim_id": task["claim_id"],
                "verdict": "unable_to_verify",
                "confidence_score": 0.0,
                "checker_used": checker_used,
                "evidence": {"error": f"Processing error in {checker_used} checker: {str(e)}"},
            }
        return {"checker_results": [result]}

    return run_checker


def _claim_number(claim_id: Any) -> int:
    try:
        return int(str(claim_id).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def final_judge(state: ReviewState) -> Dict[str, Any]:
    """STEP 3: combine the claim verdicts into the review verdict."""
    start_scope(None)
    hotel_name, hotel_review, claims_data = state["hotel_name"], state["hotel_review"], state["claims"]
    # Parallel checks finish in any order; keep the claim order of the original loop
    checker_results = sorted(state.get("checker_results", []), key=lambda r: _claim_number(r.get("claim_id")))
    try:
        return {"final_results": aggregate_claim_results(hotel_name, hotel_review, claims_data, checker_results)}
    except Exception as e:
        error_msg = f"Error in final judgment: {str(e)}"
        logger.error(f"❌ {error_msg}")
        # Fallback aggregation if final judge fails
        return {"final_results": {
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
        }}


def build_review_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("extract_claims", extract_claims)
    for node_name in CHECKERS:
        builder.add_node(node_name, _checker_node(node_name))
    builder.add_node("final_judge", final_judge)
    builder.add_edge(START, "extract_claims")
    builder.add_conditional_edges("extract_claims", route_claims, [*CHECKERS, END])
    for node_name in CHECKERS:
        builder.add_edge(node_name, "final_judge")
    builder.add_edge("final_judge", END)
    return builder.compile()


review_graph = build_review_graph()


def run_review(hotel_name: str, hotel_review: str, max_concurrency: int = config.MAX_CONCURRENCY) -> Dict[str, Any]:
    """Run the review graph and return the final results (same structure as before the migration)."""
    final_state = review_graph.invoke(
        {"hotel_name": hotel_name, "hotel_review": hotel_review},
        config={"max_concurrency": max_concurrency},
    )
    # Error messages can quote request URLs; keep API keys out of the saved results
    return redact_value(final_state["final_results"])


def write_graph_docs(path: Path = config.PROJECT_ROOT / "docs" / "graph.md") -> None:
    """Write Mermaid diagrams of the review graph and of each checker's graph."""
    sections: List[str] = [
        "# Pipeline graphs",
        "",
        "Generated from the compiled LangGraph graphs with `python -m fake_review_detection.graph`.",
        "",
    ]
    for title, graph in [
        ("Review graph", review_graph),
        ("Location checker (`location_check`)", location_graph),
        ("Web search checker (`web_check`)", web_search_graph),
        ("Opinion checker (`opinion_check`)", opinion_graph),
    ]:
        sections += [f"## {title}", "", "```mermaid", graph.get_graph().draw_mermaid().strip(), "```", ""]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    write_graph_docs()
    print("Wrote docs/graph.md")
