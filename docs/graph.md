# Pipeline graphs

Generated from the compiled LangGraph graphs with `python graph.py`.

## Review graph

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	extract_claims(extract_claims)
	location_check(location_check)
	web_check(web_check)
	opinion_check(opinion_check)
	final_judge(final_judge)
	__end__([<p>__end__</p>]):::last
	__start__ --> extract_claims;
	extract_claims -.-> __end__;
	extract_claims -.-> location_check;
	extract_claims -.-> opinion_check;
	extract_claims -.-> web_check;
	location_check --> final_judge;
	opinion_check --> final_judge;
	web_check --> final_judge;
	final_judge --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Location checker (`location_check`)

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	parse_claim(parse_claim)
	geocode_hotel(geocode_hotel)
	find_category_places(find_category_places)
	geocode_destination(geocode_destination)
	judge(judge)
	__end__([<p>__end__</p>]):::last
	__start__ --> parse_claim;
	find_category_places -.-> __end__;
	find_category_places -.-> judge;
	geocode_destination -.-> __end__;
	geocode_destination -.-> judge;
	geocode_hotel -.-> __end__;
	geocode_hotel -.-> find_category_places;
	geocode_hotel -.-> geocode_destination;
	parse_claim -.-> __end__;
	parse_claim -.-> geocode_hotel;
	judge --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Web search checker (`web_check`)

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	search(search)
	select_evidence(select_evidence)
	rank_evidence(rank_evidence)
	check_sufficiency(check_sufficiency)
	verify(verify)
	__end__([<p>__end__</p>]):::last
	__start__ --> search;
	check_sufficiency -.-> verify;
	rank_evidence --> check_sufficiency;
	search -.-> __end__;
	search -.-> select_evidence;
	select_evidence -.-> __end__;
	select_evidence -.-> rank_evidence;
	verify --> __end__;
	check_sufficiency -.-> check_sufficiency;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Opinion checker (`opinion_check`)

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	load_reviews(load_reviews)
	classify_stances(classify_stances)
	aggregate(aggregate)
	__end__([<p>__end__</p>]):::last
	__start__ --> load_reviews;
	classify_stances --> aggregate;
	load_reviews -.-> __end__;
	load_reviews -.-> classify_stances;
	aggregate --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
