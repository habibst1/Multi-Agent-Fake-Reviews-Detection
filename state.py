# state.py

import operator
from typing import Annotated, Any, Dict, List, TypedDict


class ClaimTask(TypedDict):
    """One claim, sent to its checker by the router."""
    index: int
    claim_id: str
    claim_text: str
    hotel_name: str
    claim_type: str
    claim_nature: str
    route_message: str


class ReviewState(TypedDict, total=False):
    """State shared by the nodes of the review graph."""
    hotel_name: str
    hotel_review: str
    claims: List[Dict[str, Any]]
    # Every checker appends its result; the reducer merges the results of parallel checks
    checker_results: Annotated[List[Dict[str, Any]], operator.add]
    final_results: Dict[str, Any]
