# Updated Maximum Robustness LLM Brand Audit Prompt

## Response Format Requirements

===== PART 1: NATURAL RESPONSE START =====[Your complete natural language response to the query goes here]===== PART 1: NATURAL RESPONSE END =====
===== PART 2: STRUCTURED DATA START ====={  "validation_hash": "[len_character_count]",  "response_word_count": [word_count],  "response_claim_count": [number_of_factual_claims],  ...}===== PART 2: STRUCTURED DATA END =====

**CRITICAL**: Any deviation from this format will result in automatic rejection.

---

## INPUT PARAMETERS

### Context Inputs
**Customer Scenario**: Sarah, a 29-year-old marketing manager, stands in Whole Foods’ nuts section in Brooklyn, NYC, during her lunch break, comparing pistachio options for daily office snacking. She’s committed to clean eating and tracks her macros, seeking a protein-rich, low-carb snack. With Wonderful Pistachios displayed alongside organic alternatives, she must decide between in-shell vs. no-shell, salted vs. unsalted, and organic vs. conventional within a 20-minute window and $12–15 per pound budget, prioritizing genuine nutritional value over marketing hype.

**Customer Persona**: [HEALTH_FOCUSED], [HEALTH_NUTRITION], [QUALITY_CONNOISSEUR], [YOUNG_PROFESSIONAL], [URBAN_FAST], [PREMIUM], [ONLINE], [TRADITIONAL], [MIDRANGE_TO_PREMIUM], [DIET_SPECIFIC], [ORGANIC_PREFERRED], [QUICK_STORE], [VALUE_FOCUSED], ["PISTACHIOS"], ["WONDERFUL"], ["IN-SHELL"], ["NO_SHELL"], ["SALT", "NO-SALT"]

**Location & Store Context**: 
- **Location**: Brooklyn, NYC
- **Store**: Whole Foods

**Response Requirements & Search Instructions**:
- **Data Currency Requirements**: [CURRENT_DATA_REQUIRED] – Base recommendations on verified current market data.
- **Verification Level**: [RESEARCH_MODE] – Use web search for pricing, availability, and product features rather than general knowledge.
- **Source Priority**: [REAL_TIME_PREFERRED] – Prioritize recent sources and current product information.
- **URL Enforcement**: [MANDATORY_URLS] – Every verifiable claim (e.g., price, availability, nutrition, certification) must include a source URL (e.g., brand website, retailer listing). If no URL is available, explain why in the claim’s `source_attribution`.
- **Search Query Logging**: [QUERY_LOG_REQUIRED] – List all search queries used in `search_queries_made`, with corresponding result URLs when available.
- **Search Classification**: [SEARCH_CLASSIFICATION_REQUIRED] – For SEARCH-derived claims, specify source type, verifiability, recency, and confidence; for NO_SEARCH claims, indicate reliance on pre-trained knowledge.
- **Search Tools**: Use `site:` (e.g., `site:wholefoodsmarket.com`) and `inurl:` operators to target specific domains; optionally use Google Search Console for URL indexing verification.
- **Canonical URLs**: Prioritize canonical URLs (e.g., `https://www.wholefoodsmarket.com/product/wonderful-pistachios`) over parameterized or shortened URLs.
- **Triangulation**: [TRIANGULATION_REQUIRED] – Cross-check LLM outputs with external data (e.g., X posts, consumer reviews, official retailer listings) to validate claims and mitigate hallucination.
- **Context Note**: This stage focuses on rigorous source attribution, unlike question generation and styling stages, which prioritize query authenticity over source tracking. Queries require search validation for accuracy, URL traceability, and iterative query handling to maintain conversational context.

**Customer Query**: [Input styled queries from Stage 2, e.g., “Need to pick pistachios fast at Whole Foods in Brooklyn, short on time. What’s the best brand for healthy office snacks? Also, is organic worth it?”]

**Cohort Validation**: [JSON template from Stage 1, ensuring attributes align with 15 Attributes Framework, e.g., `{"attributes": [{"code": "COREB1", "value": "HEALTH_FOCUSED", "standardization": "FULLY"}]}]

---

## OUTPUT REQUIREMENTS

### Part 1: Natural Customer Response
Write as if you are a knowledgeable assistant helping Sarah. Your response should:
- **Address the exact styled query** with practical, actionable information, reflecting the query’s consumer-like phrasing (e.g., urgent, conversational).
- **Reflect Sarah’s persona** (HEALTH_FOCUSED, QUICK_STORE, ORGANIC_PREFERRED) in tone and content, incorporating emotional undertones (e.g., stress, health validation).
- **Consider location/store context** (Whole Foods, Brooklyn, NYC) for availability, pricing, and local relevance.
- **Follow search requirements** ([CURRENT_DATA_REQUIRED], [RESEARCH_MODE], [REAL_TIME_PREFERRED], [MANDATORY_URLS], [QUERY_LOG_REQUIRED], [SEARCH_CLASSIFICATION_REQUIRED], [TRIANGULATION_REQUIRED]).
- **Provide specific recommendations** with clear reasoning, addressing iterative queries (e.g., primary + follow-ups) as a cohesive conversation to maintain context.
- **Include source URLs** for all verifiable claims, integrated naturally (e.g., “According to Whole Foods’ listing at [URL]”).
- **Explain missing URLs** for non-verifiable claims (e.g., “No URL available; based on general nutritional knowledge”).
- **Use authentic language** matching AI assistant communication, aligning with the styled query’s tone.

**Quality Standards**:
- Be genuinely helpful for Sarah’s 20-minute shopping window and $12–15 budget.
- Avoid generic advice; tailor to her macro-tracking and clean-eating goals.
- Ensure all verifiable claims have URLs or justified absences to mitigate hallucination.
- Balance comprehensiveness with urgency, maintaining conversational flow for iterative queries.

### Part 2: Systematic JSON Analysis
After writing Part 1, perform rigorous self-analysis using the JSON structure below, incorporating error handling, SEARCH/NO_SEARCH classification, and triangulation results to ensure source reliability and avoid hallucination.

---

## JSON STRUCTURE REQUIREMENTS

```json
{
  "response_metadata": {
    "validation_hash": "len_[exact_character_count_of_part1]",
    "word_count": [exact_word_count_of_part1],
    "claim_count": [total_factual_claims_identified],
    "timestamp": "[ISO_8601_timestamp, e.g., 2025-06-02T09:08:00-07:00]",
    "model_info": "[your_model_name_and_version, e.g., Grok 3]"
  },
  "context_analysis": {
    "scenario": "[Sarah’s scenario: 29-year-old marketing manager in Whole Foods, Brooklyn...]",
    "persona": "[HEALTH_FOCUSED, HEALTH_NUTRITION, ...]",
    "location": "Brooklyn, NYC; Whole Foods",
    "search_requirements": "[CURRENT_DATA_REQUIRED, RESEARCH_MODE, ...]",
    "question": "[exact_styled_query_from_stage_2, e.g., Need to pick pistachios fast...]",
    "question_category": "[HEALTH_NUTRITION|PRICE_VALUE|USE_CASE|QUALITY_SAFETY|SUSTAINABILITY|COMPARISON|GENERAL_INFO]",
    "query_sequence": [
      {
        "sequence_id": "[unique_id, e.g., q1]",
        "query": "[query_text, e.g., Need to pick pistachios fast...]",
        "follow_up_ids": ["[related_sequence_ids, e.g., q2]"]
      }
    ]
  },
  "search_compliance": {
    "search_requirements_followed": true/false,
    "search_tools_used": ["web_search", "web_fetch", "google_search_console", "none"],
    "search_queries_made": ["list", "of", "actual", "queries", "e.g., site:wholefoodsmarket.com pistachios brooklyn"],
    "search_result_urls": ["list", "of", "URLs", "from", "search", "results", "e.g., https://www.wholefoodsmarket.com/product/wonderful-pistachios"],
    "data_currency_achieved": "[CURRENT|RECENT|GENERAL|OUTDATED]",
    "verification_level_met": "[FULL|PARTIAL|MINIMAL|NONE]",
    "triangulation_compliance": {
      "external_sources_checked": ["X_posts", "Whole_Foods_listings", "consumer_reviews"],
      "results": [
        {
          "claim": "[claim_text, e.g., Wonderful Pistachios cost $7.99 for 6oz]",
          "source": "[URL_or_description, e.g., https://www.amazon.com/wonderful-pistachios]",
          "match": true/false,
          "discrepancy_explanation": "[if_no_match, e.g., Amazon lists $8.49]"
        }
      ]
    },
    "compliance_explanation": "[how_search_requirements, URL_enforcement, and triangulation_were_addressed]"
  },
  "response_evaluation": {
    "persona_alignment_score": [1-10],
    "context_utilization_score": [1-10],
    "question_completeness_score": [1-10],
    "specificity_score": [1-10],
    "actionability_score": [1-10],
    "search_integration_score": [1-10],
    "alignment_analysis": {
      "persona_elements_addressed": ["HEALTH_FOCUSED", "QUICK_STORE", "..."],
      "context_elements_used": ["Brooklyn", "20-minute_window", "..."],
      "search_elements_integrated": ["pricing_URLs", "availability_data", "..."],
      "missed_opportunities": ["list", "of", "persona/context", "elements", "not", "addressed"]
    }
  },
  "content_analysis": {
    "primary_recommendations": [
      {
        "recommendation": "[specific_recommendation_text, e.g., Wonderful Pistachios No Shells, Unsalted]",
        "reasoning": "[why_this_recommendation, e.g., High protein, fits budget]",
        "persona_relevance": "[how_this_fits_the_persona, e.g., Aligns with clean eating]",
        "source_basis": "[search_result|general_knowledge|inference]",
        "source_url": "[URL_if_applicable, e.g., https://www.wholefoodsmarket.com/product/wonderful-pistachios]"
      }
    ],
    "key_information_provided": {
      "nutritional_data": {},
      "pricing_data": {},
      "availability_data": {},
      "quality_indicators": {},
      "other_factual_content": {}
    },
    "tone_and_voice": {
      "overall_tone": "[professional|casual|expert|peer|advisory|etc]",
      "urgency_accommodation": "[how_urgency_was_reflected, e.g., Concise recommendations]",
      "authority_level": "[high|medium|low]",
      "personalization_degree": "[high|medium|low|none]"
    }
  },
  "claims_audit": [
    {
      "id": 1,
      "claim_text": "[exact_text_from_part1, e.g., Wonderful Pistachios No Shells, Unsalted cost $7.99 for 6oz]",
      "claim_type": "[NUTRITIONAL|PRICE|FEATURE|AVAILABILITY|COMPARISON|CERTIFICATION|BENEFIT|OTHER]",
      "factual_specificity": "[HIGH|MEDIUM|LOW]",
      "verifiability": "[DIRECTLY_VERIFIABLE|REQUIRES_RESEARCH|SUBJECTIVE|UNVERIFIABLE]",
      "source_attribution": "[how_source_was_indicated_in_response, e.g., According to Whole Foods’ listing]",
      "source_type": "[SEARCH_RESULT|SPECIFIC_STUDY|BRAND_WEBSITE|GENERAL_DATABASE|EXPERT_KNOWLEDGE|INFERENCE|NONE]",
      "source_url": "[exact_URL_or_null_if_unavailable, e.g., https://www.wholefoodsmarket.com/product/wonderful-pistachios]",
      "confidence_level": "[HIGH|MEDIUM|LOW]",
      "persona_relevance": "[how_this_claim_serves_the_persona, e.g., Supports macro-tracking goals]",
      "supporting_quote": "[exact_sentence_from_part1_containing_claim]",
      "search_derived": true/false,
      "search_query_used": "[query_that_led_to_source, e.g., site:wholefoodsmarket.com pistachios brooklyn, or null if NO_SEARCH]"
    }
  ],
  "error_log": [
    {
      "query": "[query_text, e.g., Is organic worth it?]",
      "issue": "[e.g., missing_URL, ambiguous_response, bias_detected, hallucinated_information]",
      "action": "[e.g., refine_query, flag_for_review, verify_manually]"
    }
  ],
  "quality_metrics": {
    "total_claims": [count],
    "sourced_claims": [count_with_attribution],
    "search_derived_claims": [count_from_search_results],
    "url_provided_claims": [count_with_source_url],
    "no_search_claims": [count_with_search_derived_false],
    "high_confidence_claims": [count],
    "persona_relevant_claims": [count],
    "verifiable_claims": [count],
    "claim_density": "[claims_per_100_words]",
    "specificity_ratio": "[specific_claims_vs_general_statements]",
    "search_integration_ratio": "[search_derived_claims_vs_total_claims]"
  },
  "brand_impact_analysis": {
    "brands_mentioned": ["list", "of", "all", "brands", "referenced", "e.g., Wonderful Pistachios, Santa Barbara Pistachio"],
    "primary_recommended_brand": "[main_brand_recommendation_if_any, e.g., Wonderful Pistachios]",
    "brand_positioning": {
      "[brand_name, e.g., Wonderful Pistachios]": {
        "sentiment": "[positive|neutral|negative]",
        "positioning": "[premium|value|specialty|etc]",
        "key_attributes_highlighted": ["list", "e.g., protein-rich, Non-GMO"],
        "limitations_noted": ["list", "e.g., not organic]",
        "source_of_brand_info": "[search_result|general_knowledge]",
        "source_url": "[URL_if_applicable, e.g., https://www.wonderfulpistachios.com]"
      }
    },
    "competitive_landscape_addressed": true/false,
    "recommendation_strength": "[strong|moderate|weak|none]"
  },
  "audit_validation": {
    "part1_extracted_completely": true/false,
    "all_claims_captured": true/false,
    "word_count_verified": true/false,
    "format_compliance": true/false,
    "persona_consideration_confirmed": true/false,
    "search_requirements_met": true/false,
    "url_enforcement_compliance": true/false,
    "triangulation_compliance": true/false,
    "search_classification_compliance": true/false,
    "auditor_confidence": "[HIGH|MEDIUM|LOW]",
    "audit_completion_statement": "I confirm this audit captures all factual claims, analyzes persona alignment, verifies search, URL, triangulation, and SEARCH/NO_SEARCH classification compliance, and logs errors for refinement"
  }
}


SEARCH FORCING MECHANISMS
Context-Based Search Instructions
Available Search Requirement Tags:

[CURRENT_DATA_REQUIRED] – Force current market data lookup.
[RESEARCH_MODE] – Require web search even for known topics.
[VERIFICATION_REQUIRED] – Mandate fact-checking through search.
[REAL_TIME_PREFERRED] – Prioritize recent sources.
[PRICE_VERIFICATION] – Require current pricing lookup.
[AVAILABILITY_CHECK] – Verify current store availability.
[MANDATORY_URLS] – Every verifiable claim must have a source URL.
[QUERY_LOG_REQUIRED] – Log all search queries and result URLs.
[SEARCH_CLASSIFICATION_REQUIRED] – Classify SEARCH claims by source type, verifiability, recency, and confidence; indicate NO_SEARCH for pre-trained knowledge.
[TRIANGULATION_REQUIRED] – Cross-check with external data to validate claims and mitigate hallucination.

Search Techniques:

Use site: (e.g., site:wholefoodsmarket.com) and inurl: (e.g., inurl:pistachios price) to target specific domains.
Prefer canonical URLs to avoid duplicates or redirects.
Optionally use Google Search Console to verify URL indexing.
Log queries, e.g., https://www.google.com/search?q=site:wholefoodsmarket.com+pistachios.

Persona-Based Search Triggers

[DATA_DRIVEN] – Sarah expects verified information.
[RESEARCH_ORIENTED] – Sarah values fact-checking.
[COMPARISON_SHOPPER] – Sarah needs current competitive analysis.


CLAIM IDENTIFICATION RULES
What counts as a factual claim:

Specific nutritional values: e.g., “contains 6g protein per ounce.”
Price points: e.g., “costs $7.99 for 6oz.”
Availability statements: e.g., “available at Whole Foods Brooklyn.”
Product features: e.g., “comes in no-shell format.”
Brand comparisons: e.g., “has less sodium than salted brands.”
Certifications: e.g., “USDA Organic certified.”
Performance claims: e.g., “provides sustained energy.”
Quality attributes: e.g., “sourced from California farms.”

URL and Search Classification Requirements:

Provide source_url for all verifiable claims (e.g., from Whole Foods, brand websites).
If no URL is available (e.g., general knowledge), set source_url: null and explain in source_attribution.
Indicate search_derived: true for SEARCH claims, with:
source_type: SEARCH_RESULT, SPECIFIC_STUDY, BRAND_WEBSITE, GENERAL_DATABASE.
verifiability: DIRECTLY_VERIFIABLE, REQUIRES_RESEARCH, SUBJECTIVE, UNVERIFIABLE.
information_recency_llm: CURRENT, RECENT, HISTORICAL, TIMELESS, UNKNOWN.
confidence_in_claim: HIGH, MEDIUM, LOW.
search_query_used: Exact query (e.g., site:wholefoodsmarket.com pistachios brooklyn).


Indicate search_derived: false for NO_SEARCH claims, with source_type: GENERAL_DATABASE, EXPERT_KNOWLEDGE, INFERENCE, or NONE, and search_query_used: null.

Extraction Methodology:

Read Part 1 sentence by sentence.
Identify every fact-checkable statement.
Extract the smallest complete factual unit.
Preserve exact wording from the response.
Assign a source URL or explain its absence.
Classify as SEARCH or NO_SEARCH, with detailed SEARCH attributes.
Note the search query for SEARCH claims.


VALIDATION REQUIREMENTS
Automatic Rejection Triggers:

Missing or incorrect format markers.
Word count mismatch between stated and actual.
Claims in Part 1 not captured in JSON.
Missing claim IDs or incomplete claim objects.
Raw response text doesn’t match Part 1 exactly.
Verifiable claims lack source URLs without explanation.
Search queries not logged in search_queries_made for SEARCH claims.
No triangulation results provided when required.
Incomplete SEARCH/NO_SEARCH classification in claims_audit.
Incomplete persona, context, or search compliance analysis.
Missing audit validation confirmation or error log.

Quality Assurance:

Every factual statement in Part 1 must appear in the claims_audit array with SEARCH/NO_SEARCH classification.
Persona alignment must be explicitly analyzed in response_evaluation.
Context utilization must be documented in context_analysis.
Search compliance, including URL enforcement, triangulation, and SEARCH/NO_SEARCH classification, must be evaluated in search_compliance.
Errors or biases (e.g., hallucination) must be logged in error_log with recommended actions (e.g., refine query, verify manually).


SUCCESS CRITERIA

Authentic Response: Part 1 reads like a genuine, helpful AI assistant response tailored to Sarah’s styled query, reflecting her urgency, health focus, and Brooklyn context.
Complete Claim Extraction: Every factual claim from Part 1 is captured in claims_audit with appropriate URLs, SEARCH/NO_SEARCH classification, and explanations for absences.
Persona Integration: Response demonstrates clear consideration of Sarah’s characteristics (HEALTH_FOCUSED, QUICK_STORE, ORGANIC_PREFERRED).
Context Awareness: Location (Brooklyn, Whole Foods) and scenario (20-minute window, $12–15 budget) influence recommendations.
Search Compliance: Adheres to [MANDATORY_URLS], [QUERY_LOG_REQUIRED], [SEARCH_CLASSIFICATION_REQUIRED], and [TRIANGULATION_REQUIRED], with logged queries, classified claims, and external data validation to mitigate hallucination.
Systematic Analysis: JSON provides a comprehensive audit of response quality, brand impact (Wonderful Pistachios), error handling, and source reliability, supporting iterative refinement.
Iterative Query Handling: Responses to iterative queries maintain conversational context, as defined by query_sequence.

BEGIN YOUR RESPONSE WITH: ===== PART 1: NATURAL RESPONSE START =====```
