# Fake Review Detection: Modular Multi-Agent LLM Pipeline

This project implements a sophisticated pipeline designed to analyze and fact-check hotel reviews. By breaking down reviews into individual factual and opinion-based claims, it leverages multiple specialized agents, powered by Large Language Models (LLMs) and external data sources, to assess the validity of each claim. The goal is to determine the overall authenticity and reliability of a given hotel review.

## How It Works 

The pipeline operates in a series of coordinated steps, utilizing distinct modules (agents) for specific tasks:

1.  **Claim Extraction (`agents/claim_extractor.py`):**
    *   The process begins by analyzing the input hotel review text.
    *   An LLM identifies and extracts discrete, fact-checkable statements (claims) from the review.
    *   Each extracted claim is classified into a specific category (e.g., `hotel_cleanliness`, `location_general`, `service_general`) and labeled as either `factual` (objectively verifiable) or `opinion` (subjective).

2.  **Parallel Claim Verification:**
    *   The pipeline routes each extracted claim to the most appropriate specialized verification agent based on its type and nature:
        *   **Location Checker (`agents/location_checker.py`):**
            *   **Target Claims:** Claims categorized as `location_general` and marked as `factual`.
            *   **Process:**
                *   **Specific Locations:** For claims mentioning a specific place (e.g., "near the Eiffel Tower"), it uses LLMs to identify the destination. It then geocodes the hotel and the specific destination using services like Nominatim. It calculates walking distances and times using OpenRouteService and uses an LLM to judge if the claimed proximity/duration is reasonable.
                *   **Category-Based Locations:** For claims using general terms (e.g., "near the beach", "close to a restaurant"), it identifies the category ("beach", "restaurant"). It geocodes the hotel and then uses the Geoapify Places API to find the *nearest* places of that specific type. It calculates routes (walking/distance) to these nearby places using OpenRouteService. Finally, an LLM evaluates the claim against the characteristics (distance, time) of these actual nearby locations to determine its validity
        *   **Web Search Checker (`agents/web_search_checker.py`):**
            *   **Target Claims:** Factual claims *not* related to `location_general`.
            *   **Process:** Performs targeted Google Search and Google Image Search based on the claim and hotel name. It uses Sentence-BERT (SBERT) to find the most semantically relevant text snippets from search results and CLIP (Contrastive Language-Image Pre-training) to find the most relevant images. An LLM then analyzes this curated evidence to determine if the claim is supported, contradicted, or if there's insufficient evidence.
        *   **Opinion Checker (`agents/opinion_checker.py`):**
            *   **Target Claims:** Claims classified as `opinion` (regardless of specific type).
            *   **Process:** Compares the opinion claim against a database of pre-processed hotel reviews (stored as JSON files). It uses SBERT to find reviews or specific claims within reviews that are semantically similar to the target claim. An LLM then analyzes these similar claims to determine if they `support`, `contradict`, or are `neutral` towards the original claim. The final verdict for the opinion claim is based on aggregating these stances.

3.  **Final Judgment (`agents/final_judge.py`):**
    *   Once all individual claims have been processed by their respective agents, the `final_judge` aggregates the results.
    *   It tallies the number of claims verified as `true`, `false`, and `unable_to_verify`.
    *   Based on this tally (e.g., majority of verifiable claims being true/false), it renders a final verdict on the overall authenticity of the original hotel review.

## Project Structure

The project is organized into modular components for clarity and maintainability:

*   `agents/`: Contains the core logic for each specialized checker.
    *   `claim_extractor.py`: Handles claim identification and classification.
    *   `location_checker.py`: Verifies location-based factual claims.
    *   `web_search_checker.py`: Verifies general factual claims using web search and image analysis.
    *   `opinion_checker.py`: Verifies opinion-based claims using a database of other reviews.
    *   `final_judge.py`: Aggregates individual claim results to determine the final review verdict.
*   `main.py`: The main orchestrator script that runs the entire pipeline.
*   `.env`: (Not included in repo) Stores sensitive API keys required for external services.

## Key Features

*   **Modular Design:** Clear separation of concerns makes the system easier to understand, debug, and extend.
*   **Multi-Agent Approach:** Different verification strategies are employed for different types of claims, increasing accuracy.
*   **LLM Integration:** Utilizes powerful LLMs for complex tasks like claim extraction, classification, evidence analysis, and final judgment.
*   **Semantic Search:** Uses SBERT and CLIP for finding relevant evidence in text and images, going beyond simple keyword matching.
*   **External Data Verification:** Leverages real-world data sources (maps, web search, review databases) to ground fact-checking.



## Example Workflow

Let's trace the pipeline using the example input:

*   **Hotel Name:** `Burj Al Arab`
*   **Review:** `"Overpriced hotel in Dubai. Located 15 minutes from Dubai Mall. Staff service was excellent but rooms felt outdated. Food quality was poor. 20 Kilometers from the airport"`

1.  **Claim Extraction:**
    *   The `claim_extractor.py` processes the review and identifies the following claims:
        *   Claim 1: `"Overpriced hotel in Dubai."` (Type: `hotel_prices`, Nature: `opinion`)
        *   Claim 2: `"Located 15 minutes from Dubai Mall."` (Type: `location_general`, Nature: `factual`)
        *   Claim 3: `"Staff service was excellent"` (Type: `service_general`, Nature: `opinion`)
        *   Claim 4: `"rooms felt outdated."` (Type: `rooms_design_and_features`, Nature: `opinion`)
        *   Claim 5: `"Food quality was poor."` (Type: `food_drinks_quality`, Nature: `opinion`)
        *   Claim 6: `"20 Kilometers from the airport"` (Type: `location_general`, Nature: `factual`)

2.  **Parallel Claim Verification:**
    *   The pipeline routes each claim:
        *   **Claim 1 (`"Overpriced hotel in Dubai."`)**: Sent to `opinion_checker.py`.
            *   Finds similar opinions in the "Burj Al Arab" review database.
            *   Aggregates stances (support/contradict/neutral) from similar claims.
            *   Returns a verdict, e.g., **`false`**.
        *   **Claim 2 (`"Located 15 minutes from Dubai Mall."`)**: Sent to `location_checker.py`.
            *   The LLM identifies "Dubai Mall" as a specific destination.
            *   The system geocodes "Burj Al Arab" and "Dubai Mall".
            *   It calculates the walking distance/time (likely much longer than 15 mins) and driving time.
            *   The final LLM judge determines this claim is **`false`** because the actual walking time is significantly longer.
        *   **Claim 3 (`"Staff service was excellent"`)**: Sent to `opinion_checker.py`.
            *   Finds similar opinions in the review database.
            *   Aggregates stances.
            *   Returns a verdict, e.g., **`true`** (if many similar positive service claims exist). 
        *   **Claim 4 (`"rooms felt outdated."`)**: Sent to `opinion_checker.py`.
            *   Finds similar opinions in the review database.
            *   Aggregates stances.
            *   Returns a verdict, e.g., **`false`**.     
        *   **Claim 5 (`"Food quality was poor."`)**: Sent to `opinion_checker.py`.
            *   Finds similar opinions in the review database.
            *   Aggregates stances.
            *   Returns a verdict, e.g., **`false`** (if multiple reviews complain about food).                  
        *   **Claim 6 (`"25 Kilometers from the airport"`)**: Sent to `location_checker.py`.
            *   The LLM identifies "airport" as a general category.
            *   The system geocodes "Burj Al Arab" and finds the nearest major airport.
            *   It calculates the distance (approximately 10 km).
            *   The final LLM judge determines this claim is **`false`**.

3.  **Final Judgment:**
    *   `final_judge.py` receives the results:
        *   Claim 1: `false`
        *   Claim 2: `false`
        *   Claim 3: `true`
        *   Claim 4: `false`
        *   Claim 5: `false`
        *   Claim 6: `false`
    *   It aggregates: 5 `false` claims, 1 `true` claim.
    *   Based on the logic (majority of verifiable claims), it determines the **Overall Review Verdict**. In this example, `false`.
    *   The pipeline returns a detailed output containing the **verdict for each individual claim** as well as the **underlying evidence** (matched reviews, geolocation data, web search snippets, images, LLM reasoning) used to reach that verdict, ensuring **transparency and explainability** for the end user.
