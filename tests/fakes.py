"""A deterministic fake world for running the original and the migrated pipeline side by side.

Both implementations get the same answers from the same fake LLM, web services and
embedding models, so any difference between their outputs comes from the code.
"""

import hashlib
import io
import json
import math
import os
import re
import sys
import time
import types

import numpy as np
import requests
import torch
from PIL import Image
from pydantic import TypeAdapter

DUMMY_KEYS = {
    "GROQ_API_KEY": "test-key",
    "ORS_API_KEY": "test-key",
    "GEOAPIFY_API_KEY": "test-key",
    "GOOGLE_API_KEY": "test-key",
    "SEARCH_ENGINE_ID": "test-engine",
}

# ----------------------------------------------------------------------------- scenarios
# claims: (claim_text, claim_type, claim_nature, claim_polarity), as the fake LLM extracts them
# places: Nominatim results per query; categories: Geoapify places per category
# google: text snippets and image titles per claim; image titles containing "missing" answer 404
# sufficient_at: evidence pieces the fake LLM needs before it calls the evidence sufficient
# failures: "<kind>:<claim>" entries make that LLM call or web request fail

SCENARIOS = {
    "readme_example": {
        "hotel": "Burj Al Arab",
        "review": "Overpriced hotel in Dubai. Located 15 minutes from Dubai Mall. Staff service was excellent but rooms felt outdated. Food quality was poor. 20 Kilometers from the airport",
        "claims": [
            ("Overpriced hotel in Dubai", "hotel_prices", "opinion", "negative"),
            ("Located 15 minutes from Dubai Mall", "location_general", "factual", "neutral"),
            ("Staff service was excellent", "service_general", "opinion", "positive"),
            ("rooms felt outdated", "rooms_design_and_features", "opinion", "negative"),
            ("Food quality was poor", "food_drinks_quality", "opinion", "negative"),
            ("20 Kilometers from the airport", "location_general", "factual", "neutral"),
        ],
        "destinations": {
            "Located 15 minutes from Dubai Mall": {"destination": "Dubai Mall", "type": "specific", "category": None},
            "20 Kilometers from the airport": {"destination": "airport", "type": "category", "category": "airport"},
        },
        "judgments": {
            "Located 15 minutes from Dubai Mall": "false",
            "20 Kilometers from the airport": "inconclusive",
        },
        "places": {
            "Burj Al Arab": [(25.1413271, 55.1853967)],
            "Dubai Mall": [(25.1205473, 55.2027748), (25.1970440, 55.2789516)],
        },
        "categories": {
            "airport": [
                ("Dubai International Airport", 25.2515424, 55.3666519),
                ("Al Maktoum International Airport", 24.8964, 55.1614),
                ("Marina Seaplane Pier", 25.0903007, 55.1328844),
            ],
        },
    },
    "web_claims": {
        "hotel": "Test Palace Hotel",
        "review": "The hotel has a rooftop infinity pool. Breakfast was included in the room price. The lobby was spotless. The neighborhood felt very safe.",
        "claims": [
            ("The hotel has a rooftop infinity pool.", "hotel_design_and_features", "factual", "positive"),
            ("Breakfast was included in the room price.", "food_drinks_prices", "factual", "positive"),
            ("The lobby was spotless.", "hotel_cleanliness", "opinion", "positive"),
            ("The neighborhood felt very safe.", "location_general", "opinion", "positive"),
        ],
        "google": {
            "The hotel has a rooftop infinity pool.": {
                "text": [
                    "The rooftop infinity pool at Test Palace Hotel overlooks the whole bay area.",
                    "Guests can swim in the heated pool on the top floor of the hotel.",
                    "Test Palace Hotel opened in 2015 and has 300 rooms across twelve floors.",
                    "Short",
                    "Travel magazine readers rated the rooftop terrace and the pool very highly.",
                ],
                "images": ["Rooftop infinity pool at sunset", "Hotel lobby with marble floor", "Pool deck missing photo", "Infinity pool from above"],
            },
            "Breakfast was included in the room price.": {
                "text": [
                    "Breakfast at Test Palace Hotel costs extra and is served from seven to ten.",
                    "Room rates at the hotel do not include breakfast according to the booking page.",
                    "The breakfast buffet offers local pastries, fruit and fresh juice every morning.",
                ],
                "images": ["Breakfast buffet table", "Coffee and pastries on a tray"],
            },
        },
        "sufficient_at": {
            "The hotel has a rooftop infinity pool.": 6,
            "Breakfast was included in the room price.": 1,
        },
        "verdicts": {
            "The hotel has a rooftop infinity pool.": "SUPPORTED",
            "Breakfast was included in the room price.": "CONTRADICTED",
        },
    },
    "failures": {
        "hotel": "Harbour Inn",
        "review": "Five minutes from the old harbour. Ten minutes from the nearest volcano. Two minutes walk from Harbour Tower. The hotel has a heated pool. The spa has a sauna. Breakfast was cold.",
        "claims": [
            ("Five minutes from the old harbour.", "location_general", "factual", "neutral"),
            ("Ten minutes from the nearest volcano.", "location_general", "factual", "neutral"),
            ("Two minutes walk from Harbour Tower.", "location_general", "factual", "neutral"),
            ("The hotel has a heated pool.", "hotel_design_and_features", "factual", "positive"),
            ("The spa has a sauna.", "facilities_design_and_features", "factual", "positive"),
            ("Breakfast was cold.", "food_drinks_quality", "opinion", "negative"),
        ],
        "destinations": {
            "Five minutes from the old harbour.": {"destination": None, "type": "specific", "category": None},
            "Ten minutes from the nearest volcano.": {"destination": "volcano", "type": "category", "category": "volcano"},
            "Two minutes walk from Harbour Tower.": {"destination": "Harbour Tower", "type": "specific", "category": None},
        },
        "judgments": {},
        "places": {
            "Harbour Inn": [(36.8065, 10.1815)],
            "Harbour Tower": [(36.8102, 10.1904)],
        },
        "route_failures": [(36.8102, 10.1904)],
        "google": {
            "The hotel has a heated pool.": {"text": [], "images": []},
            "The spa has a sauna.": {
                "text": [
                    "The spa at Harbour Inn has a sauna, a steam room and two massage rooms.",
                    "Harbour Inn guests get free access to the spa and the sauna every day.",
                ],
                "images": ["Wooden sauna interior"],
            },
        },
        "sufficient_at": {},
        "verdicts": {},
        "failures": [
            "google:The hotel has a heated pool.",
            "rank:The spa has a sauna.",
            "sufficiency:The spa has a sauna.",
            "verify:The spa has a sauna.",
            "stance:reakfast",  # every stance call about a breakfast review claim fails
        ],
    },
    "unknown_hotel": {
        "hotel": "Nowhere Lodge",
        "review": "Close to the train station. Rooms were spotless.",
        "claims": [
            ("Close to the train station.", "location_general", "factual", "neutral"),
            ("Rooms were spotless.", "rooms_cleanliness", "opinion", "positive"),
        ],
        "destinations": {
            "Close to the train station.": {"destination": "train station", "type": "category", "category": "train station"},
        },
        "places": {},
    },
    "no_claims": {
        "hotel": "Burj Al Arab",
        "review": "Nice stay overall.",
        "claims": [],
        "failures": ["extraction"],
    },
}


class LLMOutage(RuntimeError):
    """Raised by the fake LLM for calls a scenario marks as failing."""


def _digest(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest(), 16)


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def _quoted_line(prompt: str, label: str) -> str:
    """The quoted value on the line that starts with `label`, as in: User Claim: "...". """
    return re.search(rf'^{re.escape(label)} "(.*)"$', prompt, re.M).group(1)


def _png(url: str) -> bytes:
    shade = _digest(url) % 200
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, 255 - shade, (shade * 7) % 256)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.content = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error (fake service)", response=self)


class World:
    def __init__(self, scenario):
        self.scenario = scenario
        self.prompts = []
        self.failures = set(scenario.get("failures", ()))

    def _fails(self, kind: str, subject: str) -> bool:
        return f"{kind}:{subject}" in self.failures

    # ---------------------------------------------------------------- the fake LLM
    def answer(self, prompt: str):
        """Return (kind, answer) for a prompt; kind 'stance' answers were "label,confidence" text for Groq."""
        self.prompts.append(prompt)
        s = self.scenario

        if "You are a claim extractor." in prompt:
            if "extraction" in self.failures:
                raise LLMOutage("fake outage during claim extraction")
            return "json", [{"claim_text": text, "claim_type": claim_type} for text, claim_type, _, _ in s["claims"]]

        if "You are a claim classifier." in prompt:
            claims = json.loads(prompt.split("Input claims:\n", 1)[1].split("\nExpected output:", 1)[0])
            labels = {text: (nature, polarity) for text, _, nature, polarity in s["claims"]}
            return "json", [
                {
                    "claim_text": claim["claim_text"],
                    "claim_type": claim["claim_type"],
                    "claim_nature": labels[claim["claim_text"]][0],
                    "claim_polarity": labels[claim["claim_text"]][1],
                }
                for claim in claims
            ]

        if "**Task:** Extract the destination location" in prompt:
            return "json", s["destinations"][_quoted_line(prompt, "**Claim:**")]

        if "**Fact-Checking Task:**" in prompt:
            claim = _quoted_line(prompt, "**Claim:**")
            return "json", {"verdict": s["judgments"][claim], "reason": f"Fake judgment of '{claim}'", "confidence_score": 0.8}

        if prompt.lstrip().startswith("User Claim:"):
            user_claim = _quoted_line(prompt, "User Claim:")
            review_claim = _quoted_line(prompt, "Review Claim:")
            if any(f[len("stance:"):] in review_claim for f in self.failures if f.startswith("stance:")):
                raise LLMOutage("fake outage during stance classification")
            h = _digest(user_claim, review_claim)
            return "stance", {
                "classification": ("supports", "contradicts", "neutral")[h % 3],
                "confidence_score": (0.6, 0.8, 0.95)[(h // 3) % 3],
            }

        if "Your task is to rank the following" in prompt:
            claim = re.search(r'HOTEL CLAIM:\n"(.*)"\n', prompt).group(1)
            if self._fails("rank", claim):
                raise LLMOutage("fake outage during evidence ranking")
            is_text = "rank the following text evidence" in prompt
            count = len(re.findall(r"^Evidence \d+:$", prompt, re.M))
            ranking = [
                {"index": i, "score": (90 - 7 * i) if is_text else (50 + 9 * i), "reasoning": f"evidence {i} is useful"}
                for i in range(1, count + 1)
            ]
            # An index that points past the evidence list must be skipped
            ranking.append({"index": count + 1, "score": 99, "reasoning": "points at evidence that does not exist"})
            return "json", ranking

        if "is sufficient to verify the following claim" in prompt:
            claim = re.search(r'verify the following claim: .*?: "(.*)"\n', prompt).group(1)
            if self._fails("sufficiency", claim):
                raise LLMOutage("fake outage during the sufficiency check")
            pieces = len(re.findall(r"^(?:Text|Visual) Evidence \d+:$", prompt, re.M))
            enough = pieces >= s["sufficient_at"].get(claim, 1)
            return "json", {
                "is_sufficient": enough,
                "reasoning": f"{pieces} pieces of evidence",
                "additional_info_needed": "" if enough else "more independent sources",
            }

        if "HOTEL CLAIM TO VERIFY:" in prompt:
            claim = re.search(r'HOTEL CLAIM TO VERIFY:\n"(.*)"\n', prompt).group(1)
            if self._fails("verify", claim):
                raise LLMOutage("fake outage during claim verification")
            return "json", {
                "verdict": s["verdicts"][claim],
                "confidence_score": 0.7,
                "reasoning": f"Fake verification of '{claim}'",
                "supporting_text_evidence": ["text evidence 1"],
                "supporting_visual_evidence": [],
                "contradicting_text_evidence": [],
                "contradicting_visual_evidence": ["visual evidence 2"],
                "red_flags": [],
                "recommendation": "Fake recommendation",
            }

        raise AssertionError(f"Unexpected prompt:\n{prompt[:300]}")

    # ---------------------------------------------------------------- fake web services
    def request(self, method, url, params=None, json_body=None):
        s = self.scenario
        params = params or {}

        if url == "https://nominatim.openstreetmap.org/search":
            points = s.get("places", {}).get(params["q"], [])
            if params.get("bounded"):
                west, north, east, south = map(float, params["viewbox"].split(","))
                points = [(lat, lon) for lat, lon in points if south <= lat <= north and west <= lon <= east]
            return FakeResponse(200, [{"lat": str(lat), "lon": str(lon)} for lat, lon in points[: int(params["limit"])]])

        if url == "https://api.geoapify.com/v2/places":
            places = s.get("categories", {}).get(params["categories"], [])
            return FakeResponse(200, {"features": [
                {
                    "type": "Feature",
                    "properties": {"name": name, "address_line2": f"{name} Road", "categories": [params["categories"]]},
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                }
                for name, lat, lon in places[: int(params["limit"])]
            ]})

        if url.startswith("https://api.openrouteservice.org/v2/directions/"):
            (lon1, lat1), (lon2, lat2) = json_body["coordinates"]
            if (round(lat2, 4), round(lon2, 4)) in s.get("route_failures", ()):
                return FakeResponse(500, {"error": "fake routing outage"})
            km = _haversine_km(lat1, lon1, lat2, lon2) * 1.3
            return FakeResponse(200, {"routes": [{"summary": {"distance": km * 1000, "duration": km / 5 * 3600}}]})

        if url == "https://www.googleapis.com/customsearch/v1":
            claim = params["q"].split('" ', 1)[1]
            found = s.get("google", {}).get(claim, {"text": [], "images": []})
            slug = re.sub(r"[^a-z]+", "-", claim.lower()).strip("-")
            if params.get("searchType") == "image":
                items = [
                    {
                        "link": f"https://img.test/{'missing-' if 'missing' in title else ''}{slug}-{i}.png",
                        "title": title,
                        "image": {"contextLink": f"https://site.test/{slug}/{i}"},
                        "snippet": f"{title} at the hotel",
                    }
                    for i, title in enumerate(found["images"], 1)
                ]
            else:
                if self._fails("google", claim):
                    return FakeResponse(500, {"error": "fake search outage"})
                items = [
                    {"title": f"{s['hotel']} page {i}", "link": f"https://site.test/{slug}/{i}", "snippet": text}
                    for i, text in enumerate(found["text"], 1)
                ]
            return FakeResponse(200, {"items": items} if items else {})

        if url.startswith("https://img.test/"):
            return FakeResponse(404, b"") if "/missing-" in url else FakeResponse(200, _png(url))

        raise AssertionError(f"Unexpected request: {method} {url}")


# ----------------------------------------------------------------------------- fake models
EMBEDDING_SIZE = 64


def _text_vector(text: str) -> torch.Tensor:
    vector = torch.zeros(EMBEDDING_SIZE)
    for word in re.findall(r"[a-z]+", text.lower()):
        vector[_digest(word) % EMBEDDING_SIZE] += 1.0
    vector[_digest("tie-break", text) % EMBEDDING_SIZE] += 0.01  # different texts never tie exactly
    return vector / vector.norm()


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, sentences, convert_to_tensor=False, **kwargs):
        if isinstance(sentences, str):
            embeddings = _text_vector(sentences)
        else:
            embeddings = torch.stack([_text_vector(sentence) for sentence in sentences])
        return embeddings if convert_to_tensor else embeddings.numpy()


class FakeClipModel:
    def encode_text(self, tokens):
        features = torch.zeros(1, 16)
        for token in tokens[0].tolist():
            if token:
                features[0, token % 16] += 1.0
        return features + 0.01

    def encode_image(self, image_tensor):
        features = torch.zeros(1, 16)
        for i, value in enumerate(image_tensor.flatten().tolist()):
            features[0, i % 16] += value
        return features + 0.01


def fake_preprocess(image):
    pixels = np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1)


def fake_clip_load(name, device="cpu", **kwargs):
    return FakeClipModel(), fake_preprocess


# ----------------------------------------------------------------------------- installation
def _fake_groq_module(world: World) -> types.ModuleType:
    module = types.ModuleType("groq")

    class Groq:
        def __init__(self, api_key=None, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        def _create(self, messages, model=None, temperature=None, response_format=None, **kwargs):
            kind, data = world.answer(messages[0]["content"])
            content = f"{data['classification']},{data['confidence_score']}" if kind == "stance" else json.dumps(data)
            message = types.SimpleNamespace(content=content)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    module.Groq = Groq
    return module


def install(world: World, impl: str) -> None:
    """Replace the LLM, the web services, the embedding models and time.sleep with the fake world."""
    os.environ.update(DUMMY_KEYS)
    requests.get = lambda url, **kwargs: world.request("GET", url, params=kwargs.get("params"))
    requests.post = lambda url, **kwargs: world.request("POST", url, json_body=kwargs.get("json"))
    time.sleep = lambda seconds: None  # the original loop waits 30 s between claims

    import sentence_transformers
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    import clip
    clip.load = fake_clip_load

    if impl == "legacy":
        sys.modules["groq"] = _fake_groq_module(world)
    else:
        from fake_review_detection.clients import llm
        llm.ask_json = lambda prompt, schema: TypeAdapter(schema).validate_python(world.answer(prompt)[1])
