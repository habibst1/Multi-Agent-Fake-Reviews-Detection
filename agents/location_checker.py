# agents/location_checker.py

import logging
import math
from difflib import SequenceMatcher
from typing import Any, Dict, Literal, Optional, Tuple, TypedDict

import requests
from langgraph.graph import END, START, StateGraph

import config
import http_client
import llm
from run_log import note_failure, with_reasons

logger = logging.getLogger(__name__)

# --- Configuration ---
GEOAPIFY_PLACES_URL = "https://api.geoapify.com/v2/places"

# --- Place Type Mappings ---
PLACE_TYPE_MAPPINGS = {
    'attraction': 'tourism.attraction',
    'beach': 'beach',
    'restaurant': 'catering.restaurant',
    'cafe': 'catering.cafe',
    'hospital': 'healthcare.hospital',
    'pharmacy': 'healthcare.pharmacy',
    'bank': 'commercial.bank',
    'gas station': 'commercial.gas',
    'fuel station': 'commercial.gas',
    'petrol station': 'commercial.gas',
    'supermarket': 'commercial.supermarket',
    'grocery store': 'commercial.supermarket',
    'park': 'leisure.park',
    'museum': 'entertainment.museum',
    'hotel': 'accommodation.hotel',
    'church': 'religion',
    'school': 'education.school',
    'university': 'education.university',
    'library': 'education.library',
    'entertainment': 'entertainment',
    'cinema': 'entertainment.cinema',
    'theater': 'entertainment.theatre',
    'shopping mall': 'commercial.shopping_mall',
    'mall': 'commercial.shopping_mall',
    'airport': 'airport',
    'train station': 'public_transport.train',
    'bus station': 'public_transport.bus',
    'subway station': 'public_transport.subway',
    'ferry terminal': 'public_transport.ferry',
    'gym': 'sport.fitness',
    'fitness center': 'sport.fitness',
    'swimming pool': 'sport.swimming',
    'atm': 'commercial.bank',
    'post office': 'service.post',
    'bakery': 'catering.bakery',
    'nightclub': 'adult.nightclub',
    'bar': 'catering.bar',
    'pub': 'catering.pub',
    'spa': 'healthcare.spa',
    'dentist': 'healthcare.dentist',
    'veterinary': 'healthcare.veterinary',
    'bookstore': 'commercial.bookstore',
    'clothing store': 'commercial.clothing',
    'electronics store': 'commercial.electronics',
    'tourist information': 'tourism.information',
    'police station': 'service.police',
    'fire station': 'service.fire_station',
    'casino': 'adult.casino',
    'forest': 'natural.forest',
}


# --- Answer formats (Ollama constrains the output to these) ---
class ExtractedDestination(TypedDict):
    destination: Optional[str]
    type: Literal["specific", "category"]
    category: Optional[str]


class LocationJudgment(TypedDict):
    verdict: Literal["true", "false", "inconclusive"]
    reason: str
    confidence_score: float


# --- Core Location Checking Functions ---

def llm_extract_destination(claim_text, hotel_name):
    """Extracts destination, type, and category from a location claim using LLM."""
    prompt = f"""
**Task:** Extract the destination location from this travel claim, knowing the origin is {hotel_name}.
**Claim:** "{claim_text}"
**Instructions:**
1. Identify the destination location mentioned in relation to {hotel_name}
2. Determine if it's a SPECIFIC named location or a CATEGORY of places
3. Return a VALID JSON object with this exact format:
   {{
     "destination": "extracted_location",
     "type": "specific|category",
     "category": "place_type_if_applicable"
   }}
**Examples:**
- "The hotel was close to Central Park" → {{"destination": "Central Park", "type": "specific", "category": null}}
- "The hotel was near the nearest beach" → {{"destination": "beach", "type": "category", "category": "beach"}}
- "We could walk to a restaurant" → {{"destination": "restaurant", "type": "category", "category": "restaurant"}}
- "The hotel was 5 minutes from the closest gas station" → {{"destination": "gas station", "type": "category", "category": "gas station"}}
**Common category keywords:** nearest, closest, a, an, any, some
Your response must contain ONLY the JSON object, nothing else.
"""
    try:
        return llm.ask_json(prompt, ExtractedDestination)
    except Exception as e:
        note_failure(f"Error in LLM call (llm_extract_destination): {e}")
        return {"destination": None, "type": "specific", "category": None}


def find_best_matching_category(extracted_category):
    """
    Find the best matching category from PLACE_TYPE_MAPPINGS using fuzzy matching.
    Returns the best match or None if no good match is found.
    """
    if not extracted_category:
        return None
    # Direct match first
    if extracted_category in PLACE_TYPE_MAPPINGS:
        return extracted_category
    # Create expanded keywords for better matching
    category_keywords = {
        'shopping areas': 'shopping mall',
        'shopping centers': 'shopping mall',
        'shopping': 'shopping mall',
        'malls': 'shopping mall',
        'stores': 'supermarket',
        'food places': 'restaurant',
        'dining': 'restaurant',
        'eateries': 'restaurant',
        'medical facilities': 'hospital',
        'healthcare': 'hospital',
        'transit': 'train station',
        'transportation': 'train station',
        'worship places': 'church',
        'religious sites': 'church',
        'entertainment': 'cinema',
        'sports facilities': 'gym',
        'fitness': 'gym',
        'education': 'school',
        'learning': 'school',
        'finance': 'bank',
        'fuel': 'gas station',
        'petrol': 'gas station',
        'tube': 'train station',
    }
    # Check keyword mappings first
    extracted_lower = extracted_category.lower()
    for keyword, mapped_type in category_keywords.items():
        if keyword in extracted_lower or extracted_lower in keyword:
            logger.info(f"Keyword match: '{extracted_category}' -> '{mapped_type}'")
            return mapped_type
    # Fuzzy matching using similarity scores
    best_match = None
    best_score = 0.0
    min_score = 0.6  # Minimum similarity threshold
    for category in PLACE_TYPE_MAPPINGS.keys():
        # Calculate similarity score
        similarity = SequenceMatcher(None, extracted_lower, category.lower()).ratio()
        # Also check if one contains the other (partial matches)
        if extracted_lower in category.lower() or category.lower() in extracted_lower:
            similarity = max(similarity, 0.8)  # Boost partial matches
        if similarity > best_score and similarity >= min_score:
            best_score = similarity
            best_match = category
    if best_match:
        logger.info(f"Fuzzy match: '{extracted_category}' -> '{best_match}' (score: {best_score:.2f})")
        return best_match
    # Fallback: try to find any category that contains a word from the extracted category
    extracted_words = extracted_lower.split()
    for word in extracted_words:
        if len(word) > 2:  # Skip very short words
            for category in PLACE_TYPE_MAPPINGS.keys():
                if word in category.lower():
                    logger.info(f"Word match: '{extracted_category}' -> '{category}' (matched word: '{word}')")
                    return category
    logger.info(f"No suitable match found for: '{extracted_category}'")
    return None


def calculate_bounding_box(center_lat, center_lon, radius_km):
    """
    Calculate bounding box coordinates for a given center point and radius.
    Returns (south, west, north, east) coordinates.
    """
    # Earth's radius in kilometers
    earth_radius = 6371.0
    # Convert radius to degrees
    lat_change = radius_km / earth_radius * (180 / math.pi)
    lon_change = radius_km / earth_radius * (180 / math.pi) / math.cos(center_lat * math.pi / 180)
    south = center_lat - lat_change
    north = center_lat + lat_change
    west = center_lon - lon_change
    east = center_lon + lon_change
    return south, west, north, east


def geocode_location_bounded(location_name, center_coords=None, radii=[5, 15, 30, 50]):
    """
    Geocode a location with bounded box search using progressively larger radii.
    For specific destinations when hotel coordinates are available.
    """
    if not location_name:
        return None
    # If no center coordinates provided, fall back to regular geocoding
    if not center_coords:
        return geocode_location(location_name)
    center_lat, center_lon = center_coords
    for radius in radii:
        logger.info(f"Trying geocoding with {radius}km radius around hotel...")
        # Calculate bounding box
        south, west, north, east = calculate_bounding_box(center_lat, center_lon, radius)
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": location_name,
                "format": "json",
                "limit": 5,  # Get more results to find the best match
                "addressdetails": 1,
                "bounded": 1,  # Restrict to bounding box
                "viewbox": f"{west},{north},{east},{south}"
            }
            headers = {
                "User-Agent": "LocationFactChecker/1.0 (contact@example.com)"
            }
            response = http_client.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            results = response.json()
            if results:
                # Find the closest result to the center
                best_result = None
                min_distance = float('inf')
                for result in results:
                    result_lat = float(result["lat"])
                    result_lon = float(result["lon"])
                    # Calculate distance to center (placeholder, actual calc done later)
                    distance = calculate_haversine_distance(center_lat, center_lon, result_lat, result_lon)
                    if distance < min_distance:
                        min_distance = distance
                        best_result = result
                if best_result:
                    logger.info(f"Found '{location_name}' at {min_distance:.2f}km from hotel (radius: {radius}km)")
                    return (float(best_result["lat"]), float(best_result["lon"]))
        except Exception as e:
            note_failure(f"Geocoding error for '{location_name}' with {radius}km radius: {e}")
            continue
    # If bounded search fails, try unbounded search as fallback
    logger.info(f"Bounded search failed, trying unbounded geocoding for '{location_name}'...")
    return geocode_location(location_name)


def geocode_location(location_name):
    """Original geocoding function for fallback."""
    if not location_name:
        return None
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": location_name,
            "format": "json",
            "limit": 1,
            "addressdetails": 1
        }
        headers = {
            "User-Agent": "LocationFactChecker/1.0 (contact@example.com)"
        }
        response = http_client.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json()
        if results:
            return (float(results[0]["lat"]), float(results[0]["lon"]))
        return None
    except Exception as e:
        note_failure(f"Geocoding error for '{location_name}': {e}")
        return None


def find_nearest_places(origin_coords, place_type, limit=3):
    """
    Find the top N nearest places of a specific type using Geoapify Places API.
    Returns a list of places sorted by distance (closest first).
    No radius restriction - finds the nearest places wherever they are.
    """
    if not origin_coords or place_type not in PLACE_TYPE_MAPPINGS:
        logger.info(f"Invalid coordinates or unsupported place type: {place_type}")
        return []
    lat, lon = origin_coords
    category = PLACE_TYPE_MAPPINGS[place_type]
    # Geoapify Places API parameters - no radius filter, just bias towards origin
    params = {
        'categories': category,
        'bias': f'proximity:{lon},{lat}',
        'limit': limit,  # Use the same limit parameter for API call
        'apiKey': config.GEOAPIFY_API_KEY
    }
    try:
        response = http_client.get(GEOAPIFY_PLACES_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data.get('features') or len(data['features']) == 0:
            logger.info(f"No {place_type} found")
            return []
        # Calculate distances for all places
        places_with_distance = []
        for feature in data['features']:
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            if geometry.get('type') == 'Point':
                coords = geometry['coordinates']  # [longitude, latitude]
                place_lon, place_lat = coords[0], coords[1]
            else:
                continue
            # Calculate distance using Haversine formula for better accuracy
            distance = calculate_haversine_distance(lat, lon, place_lat, place_lon)
            place_info = {
                'name': properties.get('name', f'Unnamed {place_type}'),
                'lat': place_lat,
                'lon': place_lon,
                'address': properties.get('address_line2', ''),
                'category': properties.get('categories', []),
                'distance_km': round(distance, 3),  # Keep in kilometers for sorting
                'distance_m': round(distance * 1000, 2)  # Convert to meters for display
            }
            places_with_distance.append(place_info)
        # Sort by distance and return all results (already limited by API call)
        places_with_distance.sort(key=lambda x: x['distance_km'])
        if places_with_distance:
            logger.info(f"Found {len(places_with_distance)} nearest {place_type}(s):")
            for i, place in enumerate(places_with_distance, 1):
                logger.info(f"  {i}. {place['name']} ({place['distance_m']}m away)")
        return places_with_distance
    except requests.exceptions.RequestException as e:
        note_failure(f"Error with Geoapify API request: {e}")
        return []
    except Exception as e:
        note_failure(f"Error finding nearest {place_type}: {e}")
        return []


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    # Radius of earth in kilometers
    r = 6371
    return c * r


def get_route(origin_coords, destination_coords, mode="foot-walking"):
    """Calculate route using OpenRouteService."""
    if not origin_coords or not destination_coords:
        return None
    try:
        url = f"https://api.openrouteservice.org/v2/directions/{mode}"
        headers = {
            "Authorization": config.ORS_API_KEY,
            "Accept": "application/json, application/geo+json"
        }
        body = {
            "coordinates": [
                [origin_coords[1], origin_coords[0]],  # ORS expects [lon, lat]
                [destination_coords[1], destination_coords[0]]
            ]
        }
        response = http_client.post(url, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get("routes") or len(data["routes"]) == 0:
            return None
        if not data["routes"][0].get("summary"):
            return None
        summary = data["routes"][0]["summary"]
        return {
            "duration_min": round(summary["duration"] / 60, 1),
            "distance_km": round(summary["distance"] / 1000, 2)
        }
    except Exception as e:
        note_failure(f"Route calculation error: {e}")
        return None


def llm_final_judgment(claim_text, origin, destinations_info, destination_type="specific"):
    """
    Enhanced judgment function that considers multiple nearby places for better context.
    """
    if destination_type == "category" and isinstance(destinations_info, list) and len(destinations_info) > 0:
        # Multiple destinations case
        context_info = f"""
**Claim:** "{claim_text}"
**Context:**
- Origin: {origin}
- Destination Type: {destination_type}
- Closest {len(destinations_info)} {destinations_info[0].get('name', '').split()[-1] if destinations_info else 'places'} found:
"""
        for i, dest_info in enumerate(destinations_info, 1):
            route_info = dest_info.get('route_info', {})
            context_info += f"""
  {i}. {dest_info['name']}:
     - Distance: {route_info.get('distance_km', 'N/A')} km
     - Walking Time: {route_info.get('duration_min', 'N/A')} minutes
"""
    else:
        # Single destination case (backward compatibility)
        route_info = destinations_info if isinstance(destinations_info, dict) and 'duration_min' in destinations_info else {}
        context_info = f"""
**Claim:** "{claim_text}"
**Context:**
- Origin: {origin}
- Destination: {destinations_info.get('name', 'Unknown') if isinstance(destinations_info, dict) else destinations_info}
- Calculated Walking Duration: {route_info.get('duration_min', 'N/A')} minutes
- Calculated Distance: {route_info.get('distance_km', 'N/A')} km
"""
    prompt = f"""
**Fact-Checking Task:** Evaluate this travel claim's accuracy based on route data.
{context_info}
**Evaluation Rules:**
1. True if claim is reasonably accurate:
   - For claims ≤ 5 minutes: Allow ±2 minutes margin
   - For claims 5-15 minutes: Allow ±3 minutes  or 30% margin (whichever is larger)
   - For claims > 15 minutes: Allow 25% margin
2. False only if claim is significantly inaccurate:
   - Difference exceeds the above margins substantially
3. Inconclusive: If insufficient data exists
**Additional Context for Multiple Locations:**
- Allow reasonable margin for subjective terms like "close", "near", "far"
- "Walking distance" typically means under 20-30 minutes
- "Close" or "near" typically means under 10-15 minutes walk
- "Far" typically means over 20-30 minutes walk
- Note: These are the closest locations found, there may be others farther away
**Response Format (JSON ONLY):**
{{
  "verdict": "true|false|inconclusive",
  "reason": "brief explanation referencing the most relevant location(s)",
  "confidence_score": 0.0-1.0
}}
**Your Evaluation:**
"""
    try:
        result = llm.ask_json(prompt, LocationJudgment)
        return {
            "verdict": result['verdict'].lower(),
            "reason": result['reason'],
            "confidence_score": result['confidence_score']
        }
    except Exception as e:
        note_failure(f"LLM judgment error (llm_final_judgment): {e}")
        return {
            "verdict": "inconclusive",
            "reason": f"evaluation error [reason: {e}]",
            "confidence_score": 0.0 # Lowest confidence for errors
        }


# --- Location checker graph: one node per step of the original check ---

class LocationState(TypedDict, total=False):
    claim_id: str
    claim_text: str
    hotel_name: str
    claim_type: str
    extracted: Dict[str, Any]
    origin_coords: Optional[Tuple[float, float]]
    evidence: Dict[str, Any]
    destinations_for_judgment: Any
    result: Dict[str, Any]


def _unable_to_verify(claim_id, evidence):
    return {
        "claim_id": claim_id,
        "verdict": "unable_to_verify",
        "confidence_score": 0.0,
        "evidence": evidence
    }


def parse_claim(state: LocationState) -> Dict[str, Any]:
    """Step 1: extract the destination and whether it is a specific place or a category."""
    hotel_name = state.get("hotel_name")
    claim_text = state.get("claim_text")
    claim_id = state.get("claim_id")
    if not all([hotel_name, claim_text, claim_id]):
        return {"result": _unable_to_verify(claim_id, {"error": "Missing required input parameters"})}
    logger.info(f"\nProcessing claim: '{claim_text}'")
    logger.info(f"Origin hotel: {hotel_name}")
    extracted = llm_extract_destination(claim_text, hotel_name)
    logger.info(f"Extracted destination: {extracted.get('destination')}")
    logger.info(f"Destination type: {extracted.get('type', 'specific')}")
    logger.info(f"Category: {extracted.get('category')}")
    return {"extracted": extracted}


def geocode_hotel(state: LocationState) -> Dict[str, Any]:
    """Step 2: geocode the hotel, and stop when there is no destination or no hotel location."""
    claim_id = state["claim_id"]
    hotel_name = state["hotel_name"]
    extracted = state["extracted"]
    destination = extracted.get("destination")
    origin_coords = geocode_location(hotel_name)
    logger.info(f"Hotel coordinates: {origin_coords}")
    if not destination:
        return {"result": _unable_to_verify(claim_id, {
            "error": with_reasons("No destination found in claim"),
            "extracted_info": extracted
        })}
    if not origin_coords:
        return {"result": _unable_to_verify(claim_id, {
            "error": with_reasons("Could not geocode hotel location"),
            "hotel_name": hotel_name,
            "destination": destination
        })}
    evidence = {
        "hotel_coordinates": {"lat": origin_coords[0], "lng": origin_coords[1]},
        "extracted_destination": destination,
        "destination_type": extracted.get("type", "specific")
    }
    return {"origin_coords": origin_coords, "evidence": evidence}


def find_category_places(state: LocationState) -> Dict[str, Any]:
    """Step 3, category destination: find the nearest places of that kind and route to each."""
    claim_id = state["claim_id"]
    origin_coords = state["origin_coords"]
    evidence = dict(state["evidence"])
    category = state["extracted"].get("category")
    # Find the best matching category if exact match doesn't exist
    matched_category = find_best_matching_category(category)
    if not matched_category:
        return {"result": _unable_to_verify(claim_id, {
            **evidence,
            "error": f"Could not match '{category}' to any known place type"
        })}
    if matched_category != category:
        logger.info(f"Using matched category: {matched_category} (original: {category})")
        category = matched_category
    evidence["category"] = category
    # Find nearest places of this category (top 3, no radius restriction)
    nearest_places = find_nearest_places(origin_coords, category, limit=3)
    if not nearest_places:
        return {"result": _unable_to_verify(claim_id, {
            **evidence,
            "error": with_reasons(f"Could not find any {category}")
        })}
    # Calculate routes for all found places
    places_with_routes = []
    for place in nearest_places:
        place_coords = (place['lat'], place['lon'])
        route_info = get_route(origin_coords, place_coords)
        place_data = place.copy()
        place_data['route_info'] = route_info if route_info else {"duration_min": None, "distance_km": None}
        places_with_routes.append(place_data)
    # Filter out places without route info for the judgment
    valid_places = [p for p in places_with_routes if p['route_info']['duration_min'] is not None]
    if not valid_places:
        return {"result": _unable_to_verify(claim_id, {
            **evidence,
            "error": with_reasons(f"Could not calculate routes to nearby {category}"),
            "found_places": places_with_routes
        })}
    logger.info(f"Calculated routes to {len(valid_places)} {category}(s)")
    for place in valid_places:
        route = place['route_info']
        logger.info(f"  - {place['name']}: {route['duration_min']}min, {route['distance_km']}km")
    evidence["found_places"] = places_with_routes
    evidence["places_count"] = len(valid_places)
    return {"evidence": evidence, "destinations_for_judgment": valid_places}


def geocode_destination(state: LocationState) -> Dict[str, Any]:
    """Step 3, specific destination: geocode it with growing search boxes and route to it."""
    claim_id = state["claim_id"]
    origin_coords = state["origin_coords"]
    evidence = dict(state["evidence"])
    destination = state["extracted"].get("destination")
    # SPECIFIC named location - Use bounded box geocoding with progressive radii
    logger.info(f"Using bounded box geocoding for specific destination: {destination}")
    destination_coords = geocode_location_bounded(destination, origin_coords, radii=[5, 15, 30, 50])
    logger.info(f"Destination coordinates: {destination_coords}")
    if not destination_coords:
        return {"result": _unable_to_verify(claim_id, {
            **evidence,
            "error": with_reasons("Could not geocode destination even with bounded search")
        })}
    evidence["destination_coordinates"] = {"lat": destination_coords[0], "lng": destination_coords[1]}
    route_info = get_route(origin_coords, destination_coords)
    logger.info(f"Route info: {route_info}")
    if not route_info:
        return {"result": _unable_to_verify(claim_id, {
            **evidence,
            "error": with_reasons("Could not calculate route")
        })}
    evidence["distance_km"] = route_info["distance_km"]
    evidence["walking_duration_min"] = route_info["duration_min"]
    return {"evidence": evidence, "destinations_for_judgment": route_info}


def judge(state: LocationState) -> Dict[str, Any]:
    """Step 4: let the LLM judge the claim against the route data."""
    evidence = dict(state["evidence"])
    dest_type = state["extracted"].get("type", "specific")
    judgment = llm_final_judgment(state["claim_text"], state["hotel_name"], state["destinations_for_judgment"], dest_type)
    logger.info(f"   💭 Reasoning: {judgment['reason']}")
    # Convert judgment verdict to standardized format
    verdict = judgment["verdict"]
    if verdict == "inconclusive":
        verdict = "unable_to_verify"
    evidence["reasoning"] = judgment["reason"]
    logger.info(f"JUDGMENT {judgment}")
    return {"result": {
        "claim_id": state["claim_id"],
        "verdict": verdict,
        "confidence_score": judgment["confidence_score"],
        "evidence": evidence
    }}


def _stop_when_decided(next_node):
    def route(state: LocationState):
        return END if state.get("result") is not None else next_node
    return route


def _route_by_destination_type(state: LocationState):
    if state.get("result") is not None:
        return END
    if state["extracted"].get("type", "specific") == "category":
        return "find_category_places"
    return "geocode_destination"


def build_location_graph():
    builder = StateGraph(LocationState)
    builder.add_node("parse_claim", parse_claim)
    builder.add_node("geocode_hotel", geocode_hotel)
    builder.add_node("find_category_places", find_category_places)
    builder.add_node("geocode_destination", geocode_destination)
    builder.add_node("judge", judge)
    builder.add_edge(START, "parse_claim")
    builder.add_conditional_edges("parse_claim", _stop_when_decided("geocode_hotel"), ["geocode_hotel", END])
    builder.add_conditional_edges("geocode_hotel", _route_by_destination_type, ["find_category_places", "geocode_destination", END])
    builder.add_conditional_edges("find_category_places", _stop_when_decided("judge"), ["judge", END])
    builder.add_conditional_edges("geocode_destination", _stop_when_decided("judge"), ["judge", END])
    builder.add_edge("judge", END)
    return builder.compile()


location_graph = build_location_graph()


def check_location_claim(input_data):
    """
    Main function to process location claim.
    Expected input format:
    {
        "hotel_name": "Grand Hotel Paris",
        "claim_text": "located in the heart of Paris near the Eiffel Tower",
        "claim_id": "claim_1"
    }
    Returns:
    {
        "claim_id": "claim_1",
        "verdict": true/false/"unable_to_verify",
        "confidence_score": 0.85,
        "evidence": {...}
    }
    """
    try:
        return location_graph.invoke(input_data)["result"]
    except Exception as e:
        note_failure(f"Error in check_location_claim: {e}")
        return {
            "claim_id": input_data.get("claim_id", "unknown"),
            "verdict": "unable_to_verify",
            "confidence_score": 0.0,
            "evidence": {
                "error": f"Processing error: {str(e)}"
            }
        }
