# agents/location_checker.py

import os
import json
import requests
import re
import math
from difflib import SequenceMatcher
from groq import Groq

# --- Configuration and Client Initialization ---
ORS_API_KEY = os.getenv("ORS_API_KEY")
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")
GEOAPIFY_PLACES_URL = "https://api.geoapify.com/v2/places"

if not all([ORS_API_KEY, GEOAPIFY_API_KEY]):
    raise ValueError("Location checker requires ORS_API_KEY and GEOAPIFY_API_KEY. Please check your .env file.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please check your .env file.")
client = Groq(api_key=GROQ_API_KEY)

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
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt, # Note: Original code appended claim_text here, but it's already in the prompt.
                }
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.1  # Lower temperature for more deterministic responses
        )
        content = chat_completion.choices[0].message.content.strip()
        extracted = json.loads(content)
        # Ensure all required fields exist
        if 'type' not in extracted:
            extracted['type'] = 'specific'
        if 'category' not in extracted:
            extracted['category'] = None
        return extracted
    except json.JSONDecodeError:
        print("Warning: Failed to parse JSON response, using fallback extraction")
        # Fallback logic (simplified)
        text_lower = claim_text.lower()
        category_keywords = ['nearest', 'closest', 'any', 'a ', 'an ']
        for keyword in category_keywords:
            if keyword in text_lower:
                for place_type in PLACE_TYPE_MAPPINGS.keys():
                    if place_type in text_lower:
                        return {
                            "destination": place_type,
                            "type": "category",
                            "category": place_type
                        }
        return {"destination": None, "type": "specific", "category": None}
    except Exception as e:
        print("Error in Groq API call (llm_extract_destination):", e)
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
            print(f"Keyword match: '{extracted_category}' -> '{mapped_type}'")
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
        print(f"Fuzzy match: '{extracted_category}' -> '{best_match}' (score: {best_score:.2f})")
        return best_match
    # Fallback: try to find any category that contains a word from the extracted category
    extracted_words = extracted_lower.split()
    for word in extracted_words:
        if len(word) > 2:  # Skip very short words
            for category in PLACE_TYPE_MAPPINGS.keys():
                if word in category.lower():
                    print(f"Word match: '{extracted_category}' -> '{category}' (matched word: '{word}')")
                    return category
    print(f"No suitable match found for: '{extracted_category}'")
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
        print(f"Trying geocoding with {radius}km radius around hotel...")
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
            response = requests.get(url, params=params, headers=headers, timeout=10)
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
                    print(f"Found '{location_name}' at {min_distance:.2f}km from hotel (radius: {radius}km)")
                    return (float(best_result["lat"]), float(best_result["lon"]))
        except Exception as e:
            print(f"Geocoding error for '{location_name}' with {radius}km radius: {e}")
            continue
    # If bounded search fails, try unbounded search as fallback
    print(f"Bounded search failed, trying unbounded geocoding for '{location_name}'...")
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
        response = requests.get(url, params=params, headers=headers, timeout=10) # Added timeout
        response.raise_for_status()
        results = response.json()
        if results:
            return (float(results[0]["lat"]), float(results[0]["lon"]))
        return None
    except Exception as e:
        print(f"Geocoding error for '{location_name}':", e)
        return None


def find_nearest_places(origin_coords, place_type, limit=3):
    """
    Find the top N nearest places of a specific type using Geoapify Places API.
    Returns a list of places sorted by distance (closest first).
    No radius restriction - finds the nearest places wherever they are.
    """
    if not origin_coords or place_type not in PLACE_TYPE_MAPPINGS:
        print(f"Invalid coordinates or unsupported place type: {place_type}")
        return []
    lat, lon = origin_coords
    category = PLACE_TYPE_MAPPINGS[place_type]
    # Geoapify Places API parameters - no radius filter, just bias towards origin
    params = {
        'categories': category,
        'bias': f'proximity:{lon},{lat}',
        'limit': limit,  # Use the same limit parameter for API call
        'apiKey': GEOAPIFY_API_KEY
    }
    try:
        response = requests.get(GEOAPIFY_PLACES_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data.get('features') or len(data['features']) == 0:
            print(f"No {place_type} found")
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
            print(f"Found {len(places_with_distance)} nearest {place_type}(s):")
            for i, place in enumerate(places_with_distance, 1):
                print(f"  {i}. {place['name']} ({place['distance_m']}m away)")
        return places_with_distance
    except requests.exceptions.RequestException as e:
        print(f"Error with Geoapify API request: {e}")
        return []
    except Exception as e:
        print(f"Error finding nearest {place_type}: {e}")
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
            "Authorization": ORS_API_KEY,
            "Accept": "application/json, application/geo+json"
        }
        body = {
            "coordinates": [
                [origin_coords[1], origin_coords[0]],  # ORS expects [lon, lat]
                [destination_coords[1], destination_coords[0]]
            ]
        }
        response = requests.post(url, json=body, headers=headers, timeout=10) # Added timeout
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
        print(f"Route calculation error: {e}")
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
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            # Note: Original code used "qwen/qwen3-32b". Using a Groq model.
            # You might want to make this configurable or use a specific Groq model.
            model="llama-3.3-70b-versatile", # Using a Groq model
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = chat_completion.choices[0].message.content.strip()
        result = json.loads(content)
        if not all(key in result for key in ['verdict', 'reason' , 'confidence_score']):
            raise ValueError("Missing required fields in response")
        return {
            "verdict": result['verdict'].lower(),
            "reason": result['reason'],
            "confidence_score": result['confidence_score']
        }
    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"Groq judgment error (llm_final_judgment): {e}")
        return {
            "verdict": "inconclusive",
            "reason": "evaluation error",
            "confidence_score": 0.0 # Lowest confidence for errors
        }


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
        # Extract input parameters
        hotel_name = input_data.get("hotel_name")
        claim_text = input_data.get("claim_text")
        claim_id = input_data.get("claim_id")
        if not all([hotel_name, claim_text, claim_id]):
            return {
                "claim_id": claim_id,
                "verdict": "unable_to_verify",
                "confidence_score": 0.0,
                "evidence": {
                    "error": "Missing required input parameters"
                }
            }
        print(f"\nProcessing claim: '{claim_text}'")
        print(f"Origin hotel: {hotel_name}")
        # Step 1: Extract destination and determine type
        extracted = llm_extract_destination(claim_text, hotel_name)
        destination = extracted.get("destination")
        dest_type = extracted.get("type", "specific")
        category = extracted.get("category")
        print(f"Extracted destination: {destination}")
        print(f"Destination type: {dest_type}")
        print(f"Category: {category}")
        # Step 2: Geocode origin
        origin_coords = geocode_location(hotel_name)
        print(f"Hotel coordinates: {origin_coords}")
        if not destination:
            return {
                "claim_id": claim_id,
                "verdict": "unable_to_verify",
                "confidence_score": 0.0,
                "evidence": {
                    "error": "No destination found in claim",
                    "extracted_info": extracted
                }
            }
        if not origin_coords:
            return {
                "claim_id": claim_id,
                "verdict": "unable_to_verify",
                "confidence_score": 0.0,
                "evidence": {
                    "error": "Could not geocode hotel location",
                    "hotel_name": hotel_name,
                    "destination": destination
                }
            }
        evidence = {
            "hotel_coordinates": {"lat": origin_coords[0], "lng": origin_coords[1]},
            "extracted_destination": destination,
            "destination_type": dest_type
        }
        # Step 3: Get destination coordinates based on type
        if dest_type == "category":
            # Find the best matching category if exact match doesn't exist
            matched_category = find_best_matching_category(category)
            if not matched_category:
                return {
                    "claim_id": claim_id,
                    "verdict": "unable_to_verify",
                    "confidence_score": 0.0,
                    "evidence": {
                        **evidence,
                        "error": f"Could not match '{category}' to any known place type"
                    }
                }
            if matched_category != category:
                print(f"Using matched category: {matched_category} (original: {category})")
                category = matched_category
            evidence["category"] = category
            # Find nearest places of this category (top 3, no radius restriction)
            nearest_places = find_nearest_places(origin_coords, category, limit=3)
            if not nearest_places:
                return {
                    "claim_id": claim_id,
                    "verdict": "unable_to_verify",
                    "confidence_score": 0.0,
                    "evidence": {
                        **evidence,
                        "error": f"Could not find any {category}"
                    }
                }
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
                return {
                    "claim_id": claim_id,
                    "verdict": "unable_to_verify",
                    "confidence_score": 0.0,
                    "evidence": {
                        **evidence,
                        "error": f"Could not calculate routes to nearby {category}",
                        "found_places": places_with_routes
                    }
                }
            print(f"Calculated routes to {len(valid_places)} {category}(s)")
            for place in valid_places:
                route = place['route_info']
                print(f"  - {place['name']}: {route['duration_min']}min, {route['distance_km']}km")
            destinations_for_judgment = valid_places
            evidence["found_places"] = places_with_routes
            evidence["places_count"] = len(valid_places)
        else:
            # SPECIFIC named location - Use bounded box geocoding with progressive radii
            print(f"Using bounded box geocoding for specific destination: {destination}")
            destination_coords = geocode_location_bounded(destination, origin_coords, radii=[5, 15, 30, 50])
            print(f"Destination coordinates: {destination_coords}")
            if not destination_coords:
                return {
                    "claim_id": claim_id,
                    "verdict": "unable_to_verify",
                    "confidence_score": 0.0,
                    "evidence": {
                        **evidence,
                        "error": "Could not geocode destination even with bounded search"
                    }
                }
            evidence["destination_coordinates"] = {"lat": destination_coords[0], "lng": destination_coords[1]}
            # Step 4: Calculate route
            route_info = get_route(origin_coords, destination_coords)
            print(f"Route info: {route_info}")
            if not route_info:
                return {
                    "claim_id": claim_id,
                    "verdict": "unable_to_verify",
                    "confidence_score": 0.0,
                    "evidence": {
                        **evidence,
                        "error": "Could not calculate route"
                    }
                }
            evidence["distance_km"] = route_info["distance_km"]
            evidence["walking_duration_min"] = route_info["duration_min"]
            destinations_for_judgment = route_info
        # Step 5: Final judgment with enhanced context
        judgment = llm_final_judgment(claim_text, hotel_name, destinations_for_judgment, dest_type)
        # Log the reasoning from llm_final_judgment
        print(f"   💭 Reasoning: {judgment['reason']}")
        # Convert judgment verdict to standardized format
        verdict = judgment["verdict"]
        if verdict == "inconclusive":
            verdict = "unable_to_verify"
        evidence["reasoning"] = judgment["reason"]
        print("JUDGMENT" , judgment)
        result = {
            "claim_id": claim_id,
            "verdict": verdict,
            "confidence_score": judgment["confidence_score"],
            "evidence": evidence
        }
        return result
    except Exception as e:
        print(f"Error in check_location_claim: {e}")
        return {
            "claim_id": input_data.get("claim_id", "unknown"),
            "verdict": "unable_to_verify",
            "confidence_score": 0.0,
            "evidence": {
                "error": f"Processing error: {str(e)}"
            }
        }

# Example usage (if run directly for testing):
# if __name__ == "__main__":
#     # Ensure .env is loaded if running this script directly
#     # from dotenv import load_dotenv # pip install python-dotenv
#     # load_dotenv()
#     # test_data = {
#     #     "hotel_name": "Hotel Example",
#     #     "claim_text": "The hotel is 5 minutes walk from the nearest beach.",
#     #     "claim_id": "test_claim_1"
#     # }
#     # result = check_location_claim(test_data)
#     # print(json.dumps(result, indent=2))
#     pass
