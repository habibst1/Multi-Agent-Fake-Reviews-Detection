# fake_review_detection/agents/web_search_checker.py

import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import clip
import torch
from langgraph.graph import END, START, StateGraph
from PIL import Image
from pydantic import ValidationError
from sentence_transformers import util

from fake_review_detection import config, models
from fake_review_detection.clients import http_client, llm
from fake_review_detection.run_log import note_failure, with_reasons

logger = logging.getLogger(__name__)


# --- Answer formats (Ollama constrains the output to these) ---
class RankedEvidence(TypedDict):
    index: int
    score: Union[int, float]
    reasoning: str


class SufficiencyCheck(TypedDict):
    is_sufficient: bool
    reasoning: str
    additional_info_needed: str


class ClaimVerification(TypedDict):
    verdict: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    confidence_score: float
    reasoning: str
    supporting_text_evidence: List[str]
    supporting_visual_evidence: List[str]
    contradicting_text_evidence: List[str]
    contradicting_visual_evidence: List[str]
    red_flags: List[str]
    recommendation: str


# --- Core Web Search Checking Functions ---

def split_text_into_chunks(text: str, max_words: int = 250) -> List[str]:
    """Split text into manageable chunks for processing"""
    paragraphs = re.split(r'\n{2,}', text.strip())
    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para.strip())
        else:
            # Split long paragraphs into smaller chunks
            for i in range(0, len(words), max_words):
                chunk = ' '.join(words[i:i+max_words])
                chunks.append(chunk.strip())
    # Filter out very short chunks (noise)
    return [c for c in chunks if len(c.split()) > 5]


def google_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """Search Google and return results with metadata"""
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": config.GOOGLE_API_KEY,
        "cx": config.SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results
    }
    try:
        response = http_client.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        search_results = []
        if "items" in results:
            for item in results["items"]:
                snippet = item.get("snippet", "")
                title = item.get("title", "")
                url = item.get("link", "")
                if snippet:
                    search_results.append({
                        'snippet': snippet,
                        'title': title,
                        'url': url
                    })
        return search_results
    except Exception as e:
        note_failure(f"Error in Google Search: {e}")
        return []


def google_image_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """Search Google Images and return results with metadata"""
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": config.GOOGLE_API_KEY,
        "cx": config.SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results,
        "searchType": "image"
    }
    try:
        response = http_client.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        image_results = []
        if "items" in results:
            for item in results["items"]:
                image_url = item.get("link", "")
                title = item.get("title", "")
                context_url = item.get("image", {}).get("contextLink", "")
                snippet = item.get("snippet", "")
                if image_url:
                    image_results.append({
                        'image_url': image_url,
                        'title': title,
                        'context_url': context_url,
                        'snippet': snippet
                    })
        return image_results
    except Exception as e:
        note_failure(f"Error in Google Image Search: {e}")
        return []


def download_and_process_image(image_url: str, timeout: int = 10) -> Tuple[Optional[torch.Tensor], Optional[Image.Image]]:
    """Download image and preprocess it for CLIP"""
    _, clip_preprocess = models.get_clip()
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = http_client.get(image_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        # Open image and convert to RGB
        image = Image.open(BytesIO(response.content)).convert('RGB')
        # Preprocess for CLIP
        image_tensor = clip_preprocess(image).unsqueeze(0).to(models.device)
        return image_tensor, image
    except Exception as e:
        logger.warning(f"Error processing image {image_url}: {e}")
        return None, None


def find_most_similar_images(claim_text: str, image_results: List[Dict], top_k: int = 3) -> List[Dict]:
    """Find most semantically similar images using CLIP"""
    clip_model, _ = models.get_clip()
    if not image_results or not clip_model:
        return []
    try:
        # Encode the claim text with CLIP
        text_tokens = clip.tokenize([claim_text]).to(models.device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    except Exception as e:
        note_failure(f"Error encoding text with CLIP: {e}")
        return []

    similar_images = []
    logger.info(f"🖼️  Processing {len(image_results)} images...")
    for i, img_result in enumerate(image_results):
        logger.info(f"  Processing image {i+1}/{len(image_results)}: {img_result['title'][:50]}...")
        image_tensor, pil_image = download_and_process_image(img_result['image_url'])
        if image_tensor is not None:
            try:
                with torch.no_grad():
                    # Encode image with CLIP
                    image_features = clip_model.encode_image(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    # Calculate cosine similarity
                    similarity = torch.cosine_similarity(text_features, image_features).item()
                    similar_images.append({
                        'image_url': img_result['image_url'],
                        'title': img_result['title'],
                        'context_url': img_result['context_url'],
                        'snippet': img_result['snippet'],
                        'similarity_score': similarity,
                        'pil_image': pil_image  # Store for potential further processing
                    })
            except Exception as e:
                logger.warning(f"    Error encoding image: {e}")
                continue
    # Sort by similarity and return top-k
    similar_images.sort(key=lambda x: x['similarity_score'], reverse=True)
    # Remove PIL images from final results to save memory
    for img in similar_images[:top_k]:
        if 'pil_image' in img:
            del img['pil_image']
    return similar_images[:top_k]


def find_most_similar_evidence(claim_text: str, search_results: List[Dict], top_k: int = 3) -> List[Dict]:
    """Find most semantically similar evidence using SBERT"""
    sbert_model = models.get_sbert()
    if not search_results or not sbert_model:
        return []
    try:
        claim_embedding = sbert_model.encode(claim_text, convert_to_tensor=True)
    except Exception as e:
        note_failure(f"Error encoding claim with SBERT: {e}")
        return []

    # Process all chunks with source tracking
    all_chunks_with_sources = []
    for result in search_results:
        chunks = split_text_into_chunks(result['snippet'])
        for chunk in chunks:
            all_chunks_with_sources.append({
                'text': chunk,
                'title': result['title'],
                'url': result['url']
            })
    if not all_chunks_with_sources:
        return []

    try:
        # Calculate semantic similarity
        chunk_texts = [item['text'] for item in all_chunks_with_sources]
        chunk_embeddings = sbert_model.encode(chunk_texts, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(claim_embedding, chunk_embeddings)[0]
        # Get top-k most similar chunks
        top_indices = similarities.argsort(descending=True)[:top_k]
        top_evidence = []
        for i in top_indices:
            top_evidence.append({
                'text': all_chunks_with_sources[i]['text'],
                'title': all_chunks_with_sources[i]['title'],
                'url': all_chunks_with_sources[i]['url'],
                'similarity_score': float(similarities[i])
            })
        return top_evidence
    except Exception as e:
        note_failure(f"Error finding similar evidence with SBERT: {e}")
        return []


def verify_claim_with_llm(
    claim_text: str,
    text_evidence_list: List[Dict],
    visual_evidence_list: List[Dict],
    hotel_name: str
) -> Dict[str, Any]:
    """Use the local LLM to verify claim against both text and visual evidence"""
    text_evidence_section = ""
    for i, evidence in enumerate(text_evidence_list, 1):
        text_evidence_section += f"""
Text Evidence {i}:
Source: {evidence['title']} ({evidence['url']})
Content: {evidence['text']}
Similarity Score: {evidence['similarity_score']:.3f}
---
"""
    visual_evidence_section = ""
    for i, evidence in enumerate(visual_evidence_list, 1):
        visual_evidence_section += f"""
Visual Evidence {i}:
Image Title: {evidence['title']}
Image URL: {evidence['image_url']}
Context URL: {evidence['context_url']}
Description: {evidence['snippet']}
Similarity Score: {evidence['similarity_score']:.3f}
---
"""
    prompt = f"""
You are an expert fact-checker specializing in hotel and hospitality claims. Analyze the following hotel claim against the provided text and visual evidence.
HOTEL NAME: {hotel_name}
HOTEL CLAIM TO VERIFY:
"{claim_text}"
TEXT EVIDENCE FOUND:
{text_evidence_section}
VISUAL EVIDENCE FOUND:
{visual_evidence_section}
ANALYSIS INSTRUCTIONS:
1. Evaluate the claim against each piece of text and visual evidence
2. Consider source credibility (official hotel sites, booking platforms, verified reviews are more reliable)
3. Look for specific details that support or contradict the claim
4. Identify any exaggerated marketing language or unrealistic features
5. Apply strict skepticism: do NOT accept the claim unless there is clear evidence supporting it.
Return ONLY a valid JSON (no extra text):
{{
    "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
    "confidence_score": 0.85,
    "reasoning": "Detailed explanation of your analysis and decision, including how text and visual evidence complement each other",
    "supporting_text_evidence": ["Specific text evidence that supports the claim"],
    "supporting_visual_evidence": ["Specific visual evidence that supports the claim"],
    "contradicting_text_evidence": ["Specific text evidence that contradicts the claim"],
    "contradicting_visual_evidence": ["Specific visual evidence that contradicts the claim"],
    "red_flags": ["Any suspicious patterns, inconsistencies, or warning signs"],
    "recommendation": "Your final recommendation regarding this claim"
}}
"""
    try:
        return llm.ask_json(prompt, ClaimVerification)
    except ValidationError as e:
        return {
            "verdict": "ERROR",
            "confidence_score": 0.0,
            "reasoning": f"Failed to parse LLM response: {str(e)}",
            "supporting_text_evidence": [],
            "supporting_visual_evidence": [],
            "contradicting_text_evidence": [],
            "contradicting_visual_evidence": [],
            "red_flags": ["JSON parsing error"],
            "recommendation": "Manual review required due to parsing error"
        }
    except Exception as e:
        return {
            "verdict": "ERROR",
            "confidence_score": 0.0,
            "reasoning": f"LLM error: {str(e)}",
            "supporting_text_evidence": [],
            "supporting_visual_evidence": [],
            "contradicting_text_evidence": [],
            "contradicting_visual_evidence": [],
            "red_flags": ["API Error"],
            "recommendation": "Manual review required due to API error"
        }


def rank_evidence_with_llm(
    claim_text: str,
    evidence_list: List[Dict],
    hotel_name: str,
    evidence_type: str = "text"
) -> List[Dict]:
    """Rank textual or visual evidence with the local LLM based on usefulness, credibility, and relevance"""
    if not evidence_list:
        return []
    # Prepare evidence for ranking
    evidence_section = ""
    for i, evidence in enumerate(evidence_list, 1):
        if evidence_type == "text":
            evidence_section += f"""
Evidence {i}:
Source: {evidence['title']} ({evidence['url']})
Content: {evidence['text']}
Similarity Score: {evidence['similarity_score']:.3f}
---
"""
        else:  # visual
            evidence_section += f"""
Evidence {i}:
Image Title: {evidence['title']}
Image URL: {evidence['image_url']}
Context URL: {evidence['context_url']}
Description: {evidence['snippet']}
Similarity Score: {evidence['similarity_score']:.3f}
---
"""
    prompt = f"""
You are an expert fact-checker specializing in hotel and hospitality claims. Your task is to rank the following {evidence_type} evidence based on usefulness, credibility, and relevance to the claim.
HOTEL NAME:
{hotel_name}
HOTEL CLAIM:
"{claim_text}"
EVIDENCE TO RANK:
{evidence_section}
RANKING INSTRUCTIONS:
1. Evaluate each piece of evidence based on:
   - Usefulness: Does it provide specific, actionable information relevant to the claim?
   - Credibility: Is the source reliable (e.g., official hotel sites, reputable booking platforms, verified reviews)?
   - Relevance: Does it directly address the claim's specifics?
2. Assign a score (0-100) to each evidence based on these criteria.
3. Return the ranked evidence in descending order of score.
Provide the output in this JSON format (no extra text):
[
    {{
        "index": 1,
        "score": 95,
        "reasoning": "Explanation of why this evidence is ranked as such"
    }},
    ...
]
"""
    try:
        ranked_results = llm.ask_json(prompt, List[RankedEvidence])
        # Sort evidence based on the LLM's ranking
        ranked_evidence = []
        for rank in sorted(ranked_results, key=lambda x: x['score'], reverse=True):
            idx = rank['index'] - 1  # Convert to 0-based index
            if 0 <= idx < len(evidence_list):
                evidence = evidence_list[idx].copy()
                evidence['ranking_score'] = rank['score']
                evidence['ranking_reasoning'] = rank['reasoning']
                ranked_evidence.append(evidence)
        return ranked_evidence
    except Exception as e:
        note_failure(f"Error ranking {evidence_type} evidence: {e}")
        # Fallback: return original evidence with default ranking values
        fallback_evidence = []
        for evidence in evidence_list:
            evidence_copy = evidence.copy()
            evidence_copy['ranking_score'] = 0
            evidence_copy['ranking_reasoning'] = f"Ranking failed due to error: {str(e)}"
            fallback_evidence.append(evidence_copy)
        return fallback_evidence


def check_evidence_sufficiency(
    claim_text: str,
    text_evidence_list: List[Dict],
    visual_evidence_list: List[Dict],
    hotel_name: str
) -> Dict[str, Any]:
    """Check if the provided evidence is sufficient to verify the claim"""
    text_evidence_section = ""
    for i, evidence in enumerate(text_evidence_list, 1):
        text_evidence_section += f"""
Text Evidence {i}:
Source: {evidence['title']} ({evidence['url']})
Content: {evidence['text']}
Similarity Score: {evidence['similarity_score']:.3f}
Ranking Score: {evidence.get('ranking_score', 0):.1f}
---
"""
    visual_evidence_section = ""
    for i, evidence in enumerate(visual_evidence_list, 1):
        visual_evidence_section += f"""
Visual Evidence {i}:
Image Title: {evidence['title']}
Image URL: {evidence['image_url']}
Context URL: {evidence['context_url']}
Description: {evidence['snippet']}
Similarity Score: {evidence['similarity_score']:.3f}
Ranking Score: {evidence.get('ranking_score', 0):.1f}
---
"""
    prompt = f"""
You are an expert fact-checker specializing in hotel and hospitality claims. Your task is to determine if the provided text and visual evidence is sufficient to verify the following claim: {hotel_name}: "{claim_text}"
TEXT EVIDENCE:
{text_evidence_section}
VISUAL EVIDENCE:
{visual_evidence_section}
SUFFICIENCY INSTRUCTIONS:
1. Evaluate if the provided evidence is enough to conclusively verify or refute the claim.
2. Consider the specificity, credibility, and completeness of the evidence.
3. If the evidence is insufficient, specify what additional information is needed.
Provide the output in this JSON format (no extra text):
{{
    "is_sufficient": true | false,
    "reasoning": "Explanation of why the evidence is or is not sufficient",
    "additional_info_needed": "Specific information needed if evidence is insufficient"
}}
"""
    try:
        return llm.ask_json(prompt, SufficiencyCheck)
    except Exception as e:
        note_failure(f"Error checking evidence sufficiency: {e}")
        return {
            "is_sufficient": False,
            "reasoning": f"Error in LLM call: {str(e)}",
            "additional_info_needed": "Unable to assess due to API error"
        }


# --- Web search checker graph: search, select, rank, check sufficiency (loop), verify ---

MAX_EVIDENCE_PER_KIND = 3  # the sufficiency check escalates from top 1 to top 3


class WebState(TypedDict, total=False):
    claim_id: str
    claim_text: str
    hotel_name: str
    claim_type: str
    search_query: str
    search_results: List[Dict[str, Any]]
    image_results: List[Dict[str, Any]]
    text_evidence: List[Dict[str, Any]]
    visual_evidence: List[Dict[str, Any]]
    ranked_text: List[Dict[str, Any]]
    ranked_visual: List[Dict[str, Any]]
    top_k: int
    selected_text: List[Dict[str, Any]]
    selected_visual: List[Dict[str, Any]]
    sufficiency: Dict[str, Any]
    result: Dict[str, Any]


def _no_evidence_result(claim_id, error, search_query, sources_found):
    return {
        "claim_id": claim_id,
        "verdict": "unable_to_verify",
        "confidence_score": 0.0,
        "evidence": {
            "error": error,
            "search_queries": [search_query],
            "sources_found": sources_found,
            "supporting_sources": 0,
            "contradicting_sources": 0,
            "neutral_sources": sources_found,
            "key_evidence": []
        }
    }


def search(state: WebState) -> Dict[str, Any]:
    """Search Google for text and images about the claim."""
    hotel_name = state.get("hotel_name")
    claim_text = state.get("claim_text")
    claim_id = state.get("claim_id")

    if not all([hotel_name, claim_text, claim_id]):
        return {"result": {
            "claim_id": claim_id,
            "verdict": "unable_to_verify",
            "confidence_score": 0.0,
            "evidence": {
                "error": "Missing required input parameters"
            }
        }}

    search_query = f'"{hotel_name}" {claim_text}'
    logger.info(f"🔍 Verifying: {claim_text}")
    logger.info(f"🔎 Search Query: {search_query}")
    logger.info("-" * 80)
    logger.info("📚 Searching for text evidence...")
    search_results = google_search(search_query, num_results=10)
    logger.info("🖼️  Searching for visual evidence...")
    image_results = google_image_search(search_query, num_results=10)

    if not search_results and not image_results:
        logger.info("❌ No search results found")
        return {"result": _no_evidence_result(claim_id, with_reasons("No search results found"), search_query, 0)}

    logger.info(f"📚 Found {len(search_results)} text results")
    logger.info(f"🖼️  Found {len(image_results)} image results")
    return {"search_query": search_query, "search_results": search_results, "image_results": image_results}


def select_evidence(state: WebState) -> Dict[str, Any]:
    """Keep the 3 most similar text snippets (SBERT) and the 3 most similar images (CLIP)."""
    claim_text = state["claim_text"]
    search_results = state["search_results"]
    image_results = state["image_results"]

    text_evidence_list = find_most_similar_evidence(claim_text, search_results, top_k=3)
    visual_evidence_list = find_most_similar_images(claim_text, image_results, top_k=3)

    if not text_evidence_list and not visual_evidence_list:
        logger.info("❌ No relevant evidence found")
        sources_found = len(search_results) + len(image_results)
        return {"result": _no_evidence_result(state["claim_id"], with_reasons("No relevant evidence found"), state["search_query"], sources_found)}

    logger.info(f"🎯 Found {len(text_evidence_list)} relevant text evidence pieces")
    # Print textual evidence details
    if text_evidence_list:
        logger.info("\n📝 Textual Evidence Details:")
        for i, evidence in enumerate(text_evidence_list, 1):
            logger.info(f"Text Evidence {i}:")
            logger.info(f"  Title: {evidence['title']}")
            logger.info(f"  URL: {evidence['url']}")
            logger.info(f"  Content: {evidence['text']}")
            logger.info(f"  Similarity Score: {evidence['similarity_score']:.3f}")
            logger.info("-" * 40)

    logger.info(f"🎯 Found {len(visual_evidence_list)} relevant visual evidence pieces")
    # Print visual evidence details
    if visual_evidence_list:
        logger.info("\n🖼️ Visual Evidence Details:")
        for i, evidence in enumerate(visual_evidence_list, 1):
            logger.info(f"Visual Evidence {i}:")
            logger.info(f"  Image Title: {evidence['title']}")
            logger.info(f"  Image URL: {evidence['image_url']}")
            logger.info(f"  Context URL: {evidence['context_url']}")
            logger.info(f"  Description: {evidence['snippet']}")
            logger.info(f"  Similarity Score: {evidence['similarity_score']:.3f}")
            logger.info("-" * 40)

    return {"text_evidence": text_evidence_list, "visual_evidence": visual_evidence_list}


def rank_evidence(state: WebState) -> Dict[str, Any]:
    """Let the LLM rank the selected text and visual evidence."""
    claim_text = state["claim_text"]
    hotel_name = state["hotel_name"]

    logger.info("\n🤖 Ranking text evidence...")
    ranked_text_evidence = rank_evidence_with_llm(claim_text, state["text_evidence"], hotel_name, evidence_type="text")
    # Print ranked textual evidence details
    if ranked_text_evidence:
        logger.info("\n📝 Ranked Textual Evidence Details:")
        for i, evidence in enumerate(ranked_text_evidence, 1):
            logger.info(f"Ranked Text Evidence {i}:")
            logger.info(f"  Title: {evidence['title']}")
            logger.info(f"  URL: {evidence['url']}")
            logger.info(f"  Content: {evidence['text']}")
            logger.info(f"  Similarity Score: {evidence['similarity_score']:.3f}")
            logger.info(f"  Ranking Score: {evidence['ranking_score']:.1f}")
            logger.info(f"  Ranking Reasoning: {evidence['ranking_reasoning']}")
            logger.info("-" * 40)

    logger.info("\n🤖 Ranking visual evidence...")
    ranked_visual_evidence = rank_evidence_with_llm(claim_text, state["visual_evidence"], hotel_name, evidence_type="visual")
    # Print ranked visual evidence details
    if ranked_visual_evidence:
        logger.info("\n🖼️ Ranked Visual Evidence Details:")
        for i, evidence in enumerate(ranked_visual_evidence, 1):
            logger.info(f"Ranked Visual Evidence {i}:")
            logger.info(f"  Image Title: {evidence['title']}")
            logger.info(f"  Image URL: {evidence['image_url']}")
            logger.info(f"  Context URL: {evidence['context_url']}")
            logger.info(f"  Description: {evidence['snippet']}")
            logger.info(f"  Similarity Score: {evidence['similarity_score']:.3f}")
            logger.info(f"  Ranking Score: {evidence['ranking_score']:.1f}")
            logger.info(f"  Ranking Reasoning: {evidence['ranking_reasoning']}")
            logger.info("-" * 40)

    return {"ranked_text": ranked_text_evidence, "ranked_visual": ranked_visual_evidence}


def check_sufficiency(state: WebState) -> Dict[str, Any]:
    """Ask the LLM whether the top-k evidence is enough; each pass adds one more piece of each kind."""
    top_k = state.get("top_k", 0) + 1
    if top_k == 1:
        logger.info("\n🤖 Checking evidence sufficiency with top 1 text and visual evidence...")
    else:
        still = "still " if top_k > 2 else ""
        logger.info(f"\n⚠️  Evidence {still}insufficient: {state['sufficiency']['reasoning']}")
        logger.info(f"\n🤖 Escalating to top {top_k} text and visual evidence...")
    selected_text = state["ranked_text"][:top_k]
    selected_visual = state["ranked_visual"][:top_k]
    sufficiency_result = check_evidence_sufficiency(state["claim_text"], selected_text, selected_visual, state["hotel_name"])
    return {
        "top_k": top_k,
        "selected_text": selected_text,
        "selected_visual": selected_visual,
        "sufficiency": sufficiency_result
    }


def _after_sufficiency(state: WebState):
    if not state["sufficiency"]["is_sufficient"] and state["top_k"] < MAX_EVIDENCE_PER_KIND:
        return "check_sufficiency"
    return "verify"


def verify(state: WebState) -> Dict[str, Any]:
    """Let the LLM give the verdict on the selected evidence."""
    claim_text = state["claim_text"]
    selected_text = state["selected_text"]
    selected_visual = state["selected_visual"]
    sufficiency_result = state["sufficiency"]

    logger.info(f"\n🎯 Sufficiency Result: {'Sufficient' if sufficiency_result['is_sufficient'] else 'Insufficient'}")
    logger.info(f"💭 Reasoning: {sufficiency_result['reasoning']}")

    logger.info("\n🤖 Analyzing with the LLM (selected evidence)...")
    llm_analysis = verify_claim_with_llm(claim_text, selected_text, selected_visual, state["hotel_name"])

    evidence = {
        "search_queries": [state["search_query"]],
        "supporting_text_evidence": llm_analysis["supporting_text_evidence"],
        "supporting_visual_evidence": llm_analysis["supporting_visual_evidence"],
        "contradicting_text_evidence": llm_analysis["contradicting_text_evidence"],
        "contradicting_visual_evidence": llm_analysis["contradicting_visual_evidence"],
        "red_flags": llm_analysis["red_flags"],
        "recommendation": llm_analysis["recommendation"],
        "reasoning": llm_analysis["reasoning"],
        "text_evidence_count": len(selected_text),
        "visual_evidence_count": len(selected_visual),
        "sufficiency_check": sufficiency_result
    }

    if selected_text:
        evidence["text_sources"] = [e['url'] for e in selected_text]
    if selected_visual:
        evidence["visual_sources"] = [e['image_url'] for e in selected_visual]

    # Map the LLM verdict to standardized format
    verdict_mapping = {
        "SUPPORTED": "true",
        "CONTRADICTED": "false",
        "INSUFFICIENT_EVIDENCE": "unable_to_verify",
        "ERROR": "unable_to_verify"
    }
    standardized_verdict = verdict_mapping.get(llm_analysis["verdict"], "unable_to_verify")

    result = {
        "claim_id": state["claim_id"],
        "verdict": standardized_verdict,  # Use standardized verdict
        "confidence_score": llm_analysis["confidence_score"],
        "evidence": evidence
    }

    # Update emoji display to use standardized verdict
    if standardized_verdict == "true":
        verdict_emoji = "✅"
    elif standardized_verdict == "false":
        verdict_emoji = "❌"
    else:  # "unable_to_verify"
        verdict_emoji = "❓"

    logger.info(f"\n{verdict_emoji} VERDICT: {standardized_verdict.upper()}")
    logger.info(f"🎯 Confidence: {llm_analysis['confidence_score']:.2f}")
    logger.info(f"💭 Reasoning: {llm_analysis['reasoning']}")
    logger.info("=" * 80)

    return {"result": result}


def _stop_when_decided(next_node):
    def route(state: WebState):
        return END if state.get("result") is not None else next_node
    return route


def build_web_search_graph():
    builder = StateGraph(WebState)
    builder.add_node("search", search)
    builder.add_node("select_evidence", select_evidence)
    builder.add_node("rank_evidence", rank_evidence)
    builder.add_node("check_sufficiency", check_sufficiency)
    builder.add_node("verify", verify)
    builder.add_edge(START, "search")
    builder.add_conditional_edges("search", _stop_when_decided("select_evidence"), ["select_evidence", END])
    builder.add_conditional_edges("select_evidence", _stop_when_decided("rank_evidence"), ["rank_evidence", END])
    builder.add_edge("rank_evidence", "check_sufficiency")
    builder.add_conditional_edges("check_sufficiency", _after_sufficiency, ["check_sufficiency", "verify"])
    builder.add_edge("verify", END)
    return builder.compile()


web_search_graph = build_web_search_graph()


def check_web_search_claim(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to process web search claim.

    Expected input format:
    {
        "claim_id": "claim_1",
        "claim_text": "The hotel has a rooftop infinity pool.",
        "hotel_name": "Grand Plaza Hotel"
    }

    Returns:
    {
        "claim_id": "claim_1",
        "verdict": "true|false|unable_to_verify",
        "confidence_score": 0.85,
        "evidence": {...}
    }
    """
    try:
        return web_search_graph.invoke(input_data)["result"]
    except Exception as e:
        note_failure(f"Error in check_web_search_claim: {e}")
        return {
            "claim_id": input_data.get("claim_id", "unknown"),
            "verdict": "unable_to_verify",
            "confidence_score": 0.0,
            "evidence": {
                "error": f"Processing error: {str(e)}",
                "search_queries": [],
                "sources_found": 0,
                "supporting_sources": 0,
                "contradicting_sources": 0,
                "neutral_sources": 0,
                "key_evidence": []
            }
        }
