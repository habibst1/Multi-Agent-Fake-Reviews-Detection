# agents/web_search_checker.py

import os
import json
import requests
import re
import time
from io import BytesIO
from typing import Dict, List, Any, Tuple, Optional

from groq import Groq
from sentence_transformers import SentenceTransformer, util
import torch
from PIL import Image
import clip

# --- Configuration and Client Initialization ---

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
if not all([GOOGLE_API_KEY, SEARCH_ENGINE_ID]):
    raise ValueError("Web search checker requires GOOGLE_API_KEY and SEARCH_ENGINE_ID. Please check your .env file.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please check your .env file.")
client = Groq(api_key=GROQ_API_KEY)

# --- Model Initialization ---
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    print("🔧 Loading SentenceTransformer model for web search checker...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer model loaded.")
except Exception as e:
    print(f"❌ Failed to load SentenceTransformer model: {e}")
    sbert_model = None

try:
    print("🔧 Loading CLIP model for web search checker...")
    clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
    print("✅ CLIP model loaded.")
except Exception as e:
    print(f"❌ Failed to load CLIP model: {e}")
    clip_model, clip_preprocess = None, None

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
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results
    }
    try:
        response = requests.get(search_url, params=params, timeout=10) # Added timeout
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
        print(f"Error in Google Search: {e}")
        return []


def google_image_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """Search Google Images and return results with metadata"""
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results,
        "searchType": "image"
    }
    try:
        response = requests.get(search_url, params=params, timeout=10) # Added timeout
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
        print(f"Error in Google Image Search: {e}")
        return []


def download_and_process_image(image_url: str, timeout: int = 10) -> Tuple[Optional[torch.Tensor], Optional[Image.Image]]:
    """Download image and preprocess it for CLIP"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        # Open image and convert to RGB
        image = Image.open(BytesIO(response.content)).convert('RGB')
        # Preprocess for CLIP
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        return image_tensor, image
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None, None


def find_most_similar_images(claim_text: str, image_results: List[Dict], top_k: int = 3) -> List[Dict]:
    """Find most semantically similar images using CLIP"""
    if not image_results or not clip_model:
        return []
    try:
        # Encode the claim text with CLIP
        text_tokens = clip.tokenize([claim_text]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    except Exception as e:
        print(f"Error encoding text with CLIP: {e}")
        return []

    similar_images = []
    print(f"🖼️  Processing {len(image_results)} images...")
    for i, img_result in enumerate(image_results):
        print(f"  Processing image {i+1}/{len(image_results)}: {img_result['title'][:50]}...")
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
                print(f"    Error encoding image: {e}")
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
    if not search_results or not sbert_model:
        return []
    try:
        claim_embedding = sbert_model.encode(claim_text, convert_to_tensor=True)
    except Exception as e:
        print(f"Error encoding claim with SBERT: {e}")
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
        print(f"Error finding similar evidence with SBERT: {e}")
        return []


def generate_with_groq(prompt: str, model: str) -> Optional[str]:
    """Generate text using Groq API"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error in Groq API (generate_with_groq): {e}")
        return None


def verify_claim_with_groq(
    claim_text: str,
    text_evidence_list: List[Dict],
    visual_evidence_list: List[Dict],
    hotel_name: str
) -> Dict[str, Any]:
    """Use Groq API to verify claim against both text and visual evidence"""
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
        response_text = generate_with_groq(prompt , "llama-3.3-70b-versatile")
        if not response_text:
            raise ValueError("Empty response from Groq API")
        # Extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text.strip()
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        return {
            "verdict": "ERROR",
            "confidence_score": 0.0,
            "reasoning": f"Failed to parse Groq response: {str(e)}",
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
            "reasoning": f"Groq API error: {str(e)}",
            "supporting_text_evidence": [],
            "supporting_visual_evidence": [],
            "contradicting_text_evidence": [],
            "contradicting_visual_evidence": [],
            "red_flags": ["API Error"],
            "recommendation": "Manual review required due to API error"
        }


def rank_evidence_with_groq(
    claim_text: str,
    evidence_list: List[Dict],
    hotel_name: str,
    evidence_type: str = "text"
) -> List[Dict]:
    """Rank textual or visual evidence using Groq based on usefulness, credibility, and relevance"""
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
        response_text = generate_with_groq(prompt, "llama-3.3-70b-versatile")
        if not response_text:
            raise ValueError("Empty response from Groq API")
        # Extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text.strip()
        ranked_results = json.loads(json_text)
        # Sort evidence based on Groq's ranking
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
        print(f"Error ranking {evidence_type} evidence: {e}")
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
        response_text = generate_with_groq(prompt , "llama-3.3-70b-versatile")
        if not response_text:
            raise ValueError("Empty response from Groq API")
        # Extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text.strip()
        return json.loads(json_text)
    except Exception as e:
        print(f"Error checking evidence sufficiency: {e}")
        return {
            "is_sufficient": False,
            "reasoning": f"Error in Groq API: {str(e)}",
            "additional_info_needed": "Unable to assess due to API error"
        }


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
        hotel_name = input_data.get("hotel_name")
        claim_text = input_data.get("claim_text")
        claim_id = input_data.get("claim_id")
        claim_type = input_data.get("claim_type", "factual_claim")

        if not all([hotel_name, claim_text, claim_id]):
            return {
                "claim_id": claim_id,
                "verdict": "unable_to_verify",
                "confidence_score": 0.0,
                "evidence": {
                    "error": "Missing required input parameters"
                }
            }

        search_query = f'"{hotel_name}" {claim_text}'
        print(f"🔍 Verifying: {claim_text}")
        print(f"🔎 Search Query: {search_query}")
        print("-" * 80)
        print("📚 Searching for text evidence...")
        search_results = google_search(search_query, num_results=10)
        print("🖼️  Searching for visual evidence...")
        image_results = google_image_search(search_query, num_results=10)

        if not search_results and not image_results:
            print("❌ No search results found")
            return {
                "claim_id": claim_id,
                "verdict": "unable_to_verify",
                "confidence_score": 0.0,
                "evidence": {
                    "error": "No search results found",
                    "search_queries": [search_query],
                    "sources_found": 0,
                    "supporting_sources": 0,
                    "contradicting_sources": 0,
                    "neutral_sources": 0,
                    "key_evidence": []
                }
            }

        print(f"📚 Found {len(search_results)} text results")
        print(f"🖼️  Found {len(image_results)} image results")

        text_evidence_list = find_most_similar_evidence(claim_text, search_results, top_k=3)
        visual_evidence_list = find_most_similar_images(claim_text, image_results, top_k=3)

        if not text_evidence_list and not visual_evidence_list:
            print("❌ No relevant evidence found")
            return {
                "claim_id": claim_id,
                "verdict": "unable_to_verify",
                "confidence_score": 0.0,
                "evidence": {
                    "error": "No relevant evidence found",
                    "search_queries": [search_query],
                    "sources_found": len(search_results) + len(image_results),
                    "supporting_sources": 0,
                    "contradicting_sources": 0,
                    "neutral_sources": len(search_results) + len(image_results),
                    "key_evidence": []
                }
            }

        print(f"🎯 Found {len(text_evidence_list)} relevant text evidence pieces")
        # Print textual evidence details
        if text_evidence_list:
            print("\n📝 Textual Evidence Details:")
            for i, evidence in enumerate(text_evidence_list, 1):
                print(f"Text Evidence {i}:")
                print(f"  Title: {evidence['title']}")
                print(f"  URL: {evidence['url']}")
                print(f"  Content: {evidence['text']}")
                print(f"  Similarity Score: {evidence['similarity_score']:.3f}")
                print("-" * 40)

        print(f"🎯 Found {len(visual_evidence_list)} relevant visual evidence pieces")
        # Print visual evidence details
        if visual_evidence_list:
            print("\n🖼️ Visual Evidence Details:")
            for i, evidence in enumerate(visual_evidence_list, 1):
                print(f"Visual Evidence {i}:")
                print(f"  Image Title: {evidence['title']}")
                print(f"  Image URL: {evidence['image_url']}")
                print(f"  Context URL: {evidence['context_url']}")
                print(f"  Description: {evidence['snippet']}")
                print(f"  Similarity Score: {evidence['similarity_score']:.3f}")
                print("-" * 40)

        print("\n🤖 Ranking text evidence...")
        ranked_text_evidence = rank_evidence_with_groq(claim_text, text_evidence_list, hotel_name, evidence_type="text")
        # Print ranked textual evidence details
        if ranked_text_evidence:
            print("\n📝 Ranked Textual Evidence Details:")
            for i, evidence in enumerate(ranked_text_evidence, 1):
                print(f"Ranked Text Evidence {i}:")
                print(f"  Title: {evidence['title']}")
                print(f"  URL: {evidence['url']}")
                print(f"  Content: {evidence['text']}")
                print(f"  Similarity Score: {evidence['similarity_score']:.3f}")
                print(f"  Ranking Score: {evidence['ranking_score']:.1f}")
                print(f"  Ranking Reasoning: {evidence['ranking_reasoning']}")
                print("-" * 40)

        print("\n🤖 Ranking visual evidence...")
        ranked_visual_evidence = rank_evidence_with_groq(claim_text, visual_evidence_list, hotel_name, evidence_type="visual")
        # Print ranked visual evidence details
        if ranked_visual_evidence:
            print("\n🖼️ Ranked Visual Evidence Details:")
            for i, evidence in enumerate(ranked_visual_evidence, 1):
                print(f"Ranked Visual Evidence {i}:")
                print(f"  Image Title: {evidence['title']}")
                print(f"  Image URL: {evidence['image_url']}")
                print(f"  Context URL: {evidence['context_url']}")
                print(f"  Description: {evidence['snippet']}")
                print(f"  Similarity Score: {evidence['similarity_score']:.3f}")
                print(f"  Ranking Score: {evidence['ranking_score']:.1f}")
                print(f"  Ranking Reasoning: {evidence['ranking_reasoning']}")
                print("-" * 40)

        selected_text = ranked_text_evidence[:1] if ranked_text_evidence else []
        selected_visual = ranked_visual_evidence[:1] if ranked_visual_evidence else []

        print("\n🤖 Checking evidence sufficiency with top 1 text and visual evidence...")
        sufficiency_result = check_evidence_sufficiency(claim_text, selected_text, selected_visual, hotel_name)

        if not sufficiency_result['is_sufficient']:
            print(f"\n⚠️  Evidence insufficient: {sufficiency_result['reasoning']}")
            print("\n🤖 Escalating to top 2 text and visual evidence...")
            selected_text = ranked_text_evidence[:2] if len(ranked_text_evidence) >= 2 else ranked_text_evidence
            selected_visual = ranked_visual_evidence[:2] if len(ranked_visual_evidence) >= 2 else ranked_visual_evidence
            sufficiency_result = check_evidence_sufficiency(claim_text, selected_text, selected_visual, hotel_name)

            if not sufficiency_result['is_sufficient']:
                print(f"\n⚠️  Evidence still insufficient: {sufficiency_result['reasoning']}")
                print("\n🤖 Escalating to top 3 text and visual evidence...")
                selected_text = ranked_text_evidence[:3] if len(ranked_text_evidence) >= 3 else ranked_text_evidence
                selected_visual = ranked_visual_evidence[:3] if len(ranked_visual_evidence) >= 3 else ranked_visual_evidence
                sufficiency_result = check_evidence_sufficiency(claim_text, selected_text, selected_visual, hotel_name)

        print(f"\n🎯 Sufficiency Result: {'Sufficient' if sufficiency_result['is_sufficient'] else 'Insufficient'}")
        print(f"💭 Reasoning: {sufficiency_result['reasoning']}")

        print("\n🤖 Analyzing with Groq LLM (selected evidence)...")
        groq_analysis = verify_claim_with_groq(claim_text, selected_text, selected_visual, hotel_name)

        evidence = {
            "search_queries": [search_query],
            "supporting_text_evidence": groq_analysis["supporting_text_evidence"],
            "supporting_visual_evidence": groq_analysis["supporting_visual_evidence"],
            "contradicting_text_evidence": groq_analysis["contradicting_text_evidence"],
            "contradicting_visual_evidence": groq_analysis["contradicting_visual_evidence"],
            "red_flags": groq_analysis["red_flags"],
            "recommendation": groq_analysis["recommendation"],
            "reasoning": groq_analysis["reasoning"],
            "text_evidence_count": len(selected_text),
            "visual_evidence_count": len(selected_visual),
            "sufficiency_check": sufficiency_result
        }

        if selected_text:
            evidence["text_sources"] = [e['url'] for e in selected_text]
        if selected_visual:
            evidence["visual_sources"] = [e['image_url'] for e in selected_visual]

        # Map Groq analysis verdict to standardized format
        verdict_mapping = {
            "SUPPORTED": "true",
            "CONTRADICTED": "false",
            "INSUFFICIENT_EVIDENCE": "unable_to_verify",
            "ERROR": "unable_to_verify"
        }
        groq_verdict = groq_analysis["verdict"]
        standardized_verdict = verdict_mapping.get(groq_verdict, "unable_to_verify")

        result = {
            "claim_id": claim_id,
            "verdict": standardized_verdict,  # Use standardized verdict
            "confidence_score": groq_analysis["confidence_score"],
            "evidence": evidence
        }

        # Update emoji display to use standardized verdict
        if standardized_verdict == "true":
            verdict_emoji = "✅"
        elif standardized_verdict == "false":
            verdict_emoji = "❌"
        else:  # "unable_to_verify"
            verdict_emoji = "❓"

        print(f"\n{verdict_emoji} VERDICT: {standardized_verdict.upper()}")
        print(f"🎯 Confidence: {groq_analysis['confidence_score']:.2f}")
        print(f"💭 Reasoning: {groq_analysis['reasoning']}")
        print("=" * 80)

        return result

    except Exception as e:
        print(f"Error in check_web_search_claim: {e}")
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

# Example usage (if run directly for testing):
# if __name__ == "__main__":
#     # Ensure .env is loaded if running this script directly
#     # from dotenv import load_dotenv # pip install python-dotenv
#     # load_dotenv()
#     # test_data = {
#     #     "claim_id": "test_claim_1",
#     #     "claim_text": "The hotel has a rooftop infinity pool.",
#     #     "hotel_name": "Grand Plaza Hotel"
#     # }
#     # result = check_web_search_claim(test_data)
#     # print(json.dumps(result, indent=2))
#     pass
