# Manual Instruction Page for Brandscope Audit Workflow

## Overview

This guide provides step-by-step instructions to manually execute a three-stage workflow for a Brandscope audit of Wonderful Pistachios, targeting the Proactive Health Seeker cohort (Sarah, a 29-year-old marketing manager in Brooklyn, NYC). The workflow generates authentic consumer queries, refines them for realism, and analyzes LLM responses to audit brand portrayal. No automated tools are required; use text documents or spreadsheets to record inputs and outputs. The process ensures cohort validation, query prioritization, iterative query handling, error logging, triangulation, and selective style conversion, producing JSON metadata for traceability.

## Prerequisites

- **Documents**: Access to normalized prompts (`prompt-stage_1_scenario_generation_with_recco_v1.md`, `prompt-stage_1_scenario_generation_no_recco_v1.md`, `prompt_stage_2_style_conversion_v1.md`, `prompt_final_stage_V2.md`).
- **Reference**: 15 Attributes Framework (`Prompting Methodology for Brand Evaluation - 15 Attributes (1).csv`) for cohort definition.
- **Materials**: Text editor (e.g., Notepad, Word), spreadsheet (e.g., Excel, Google Sheets), or paper for recording inputs/outputs.
- **Time**: Approximately 2–3 hours per cohort, depending on query volume and analysis depth.
- **Access**: Internet for manual web searches to validate claims (e.g., Whole Foods listings, X posts).

## Workflow Stages

1. **Stage 1: Cohort Customization and Query Generation**
   - Define Sarah’s cohort, validate attributes, and generate raw queries (recommendation-focused and research-focused).
2. **Stage 2: Style Conversion**
   - Transform raw queries into consumer-like questions, applying refined style guidelines.
3. **Stage 3: LLM Analysis and Verification**
   - Analyze styled queries with an LLM, enforce URLs, log errors, and triangulate results.

## Instructions

### Stage 1: Cohort Customization and Query Generation

**Objective**: Define Sarah’s cohort, validate attributes, and generate 15–20 recommendation-focused (`with_recco`) and 12–18 research-focused (`no_recco`) raw queries.

**Steps**:

1. **Prepare Cohort Definition**:
   
   - Open the 15 Attributes Framework (`Prompting Methodology for Brand Evaluation - 15 Attributes (1).csv`).
   
   - Create a JSON template for Sarah’s cohort in a text document:
     
     ```json
     {
      "ATTRIBUTE_COMBINATION": {
        "Primary_Motivation": "HEALTH_FOCUSED",
        "Context": "URBAN_FAST",
        "Urgency_Level": "QUICK_STORE",
        "Key_Constraint": "MIDRANGE_TO_PREMIUM"
      },
      "PRODUCT_FOCUS": "Pistachios (in-shell, no-shell; salted, unsalted)",
      "TRIGGER_SITUATION": "Need for a protein-rich, low-carb snack for daily office consumption, aligned with clean eating and macro tracking",
      "COMPETITIVE_SET": "Wonderful Pistachios, Santa Barbara Pistachio, Eden Foods, 365 by Whole Foods Market",
      "DECISION_TIMEFRAME": "20-minute shopping window, quick decision needed",
      "ADDITIONAL_CONTEXT": "Sarah, 29, female, Brooklyn, NYC, shops at Whole Foods, values organic certification, uses mobile device in-store",
      "COHORT_VALIDATION": {
        "attributes": [
          { "code": "COREB1", "value": "HEALTH_FOCUSED", "standardization": "FULLY" },
          { "code": "DEMOD2", "value": "URBAN_FAST", "standardization": "MOSTLY" },
          { "code": "COREA2", "value": "QUICK_STORE", "standardization": "FULLY" },
          { "code": "MODIFIERD3", "value": "MIDRANGE_TO_PREMIUM", "standardization": "HYBRID" },
          ...
        ]
      }
     }
     ```
   
   - **Validation**: Check each attribute against the framework’s standardization level (e.g., COREB1 allows only predefined values like HEALTH_FOCUSED). Note errors in `COHORT_VALIDATION.errors`, e.g., `["Invalid value for MODIFIERC2"]`. If errors exist, revise attributes until `status: "valid"`.

2. **Generate Raw Queries (`with_recco`)**:
   
   - Open `prompt-stage_1_scenario_generation_with_recco_v1.md`.
   - In a text document, list 15–20 queries across five categories (direct recommendation, indirect recommendation, comparative analysis, validation & implementation, scenario markdown).
   - Use the example input to craft queries, keeping them neutral (no consumer styling), e.g.:
     - **Direct**: “Which pistachio brand is recommended for health benefits at Whole Foods?”
     - **Indirect**: “Which pistachios support a low-carb diet?”
     - **Comparative**: “How do organic pistachios compare to conventional ones?”
     - **Validation**: “Is Wonderful Pistachios suitable for daily snacking?”
     - **Scenario Markdown**: “Sarah, a 29-year-old marketing manager in Brooklyn, NYC, rushes into Whole Foods during her lunch break to find pistachios for daily office snacking. Health-focused and tracking macros, she seeks a protein-rich, low-carb snack, ideally organic, within a $12–15 budget and 20-minute window. She considers Wonderful Pistachios, Santa Barbara Pistachio, Eden Foods, and 365 by Whole Foods Market.” (117 words)
   - For iterative queries, note follow-ups, e.g., “Which brand is recommended? Also, is organic worth it?”
   - Assign each query a `sequence_id` (e.g., “q1”) and link follow-ups via `follow_up_ids` (e.g., [“q2”]).

3. **Generate Raw Queries (`no_recco`)**:
   
   - Open `prompt-stage_1_scenario_generation_no_recco_v1.md`.
   - List 12–18 queries across four categories (primary information, comparative analysis, decision validation, implementation), e.g.:
     - **Primary**: “What nutritional benefits do pistachios provide?”
     - **Comparative**: “How do pistachios compare to almonds for protein?”
     - **Validation**: “Are there health risks from daily pistachio consumption?”
     - **Implementation**: “How should pistachios be stored for freshness?”
   - Assign `sequence_id` and `follow_up_ids` as above.

4. **Prioritize Queries**:
   
   - In a spreadsheet, list all queries (`with_recco` and `no_recco`) with columns: Query, Category, Sequence ID, Follow-Up IDs, Relevance Score (1–5).
   - Score queries based on alignment with Sarah’s goals (health focus, Whole Foods, budget), e.g., “Which pistachio brand is recommended for health benefits?” = 5, “How do pistachios grow?” = 2.
   - Select 5–10 queries (balanced mix, e.g., 3 direct, 2 indirect, 2 comparative, 3 others) with relevance_score ≥ 4 for Stage 2.

5. **Create Stage 1 JSON Output**:
   
   - In a text document, compile the JSON output per the prompt’s structure:
     
     ```json
     {
      "metadata": {
        "artifact_id": "717db0e3-2bff-4766-803a-2816acfc39fa",
        "title": "Raw Consumer Queries for Proactive Health Seeker Cohort",
        "content_type": "text/markdown",
        "linked_style_artifact_id": "b5f2e5fd0-0158-42c1-ad17-83ca5ef2052b",
        "linked_final_artifact_id": "b8e4f7d2-9f3a-4c7-8e9b-5c7d0f2a1c2d",
        "timestamp": "2025-06-02T08:18:00-07:00",
        "purpose": "Generate raw queries for style conversion and Brandscope audit",
        "cohort": "Proactive Health Seeker",
        "context": "Sarah, 29, Brooklyn, NYC, shopping at Whole Foods for office-snacking pistachios"
      },
      "cohort_validation": {
        "status": "valid",
        "attributes": [
          {"code": "COREB1", "value": "HEALTH_FOCUSED", "standardization": "FULLY"},
          ...
        ],
        "errors": []
      },
      "queries": {
        "direct_recommendation": [
          {"query": "Which pistachio brand is recommended for health benefits at Whole Foods?", "sequence_id": "q1", "follow_up_ids": ["q2"], "relevance_score": 5}
        ],
        "indirect_recommendation": [],
        "comparative_analysis": [],
        "validation_implementation": []
      },
      "filtered_queries": [
        {"query": "Which pistachio brand is recommended...", "sequence_id": "q1", "relevance_score": 5, "category": "direct_recommendation"}
      ],
      "scenario_markdown": "[Sarah’s scenario...]"
     }
     ```
   
   - Save the document as `stage1_output.json`.

**Time Estimate**: 1–1.5 hours

---

### Stage 2: Style Conversion

**Objective**: Transform 5–10 filtered raw queries from Stage 1 into consumer-like queries, applying refined style guidelines (emotional nuances, device-specific patterns, iterative questioning, colloquial imprecision, localized context).

**Steps**:

1. **Prepare Input**:
   
   - Open `stage1_output.json` and extract the `filtered_queries` section.
   - Copy the 5–10 selected queries into a new text document or spreadsheet, noting their `sequence_id`, `follow_up_ids`, `relevance_score`, and `category`.

2. **Apply Style Conversion**:
   
   - Open `prompt_stage_2_style_conversion_v1.md`.
   - For each query, transform it into a consumer-like version, reflecting Sarah’s persona (HEALTH_FOCUSED, QUICK_STORE) and context (Brooklyn, Whole Foods, 20-minute window). Use the style principles:
     - **Urgency**: “short on time,” “need to pick fast.”
     - **Emotional Undertones**: “worried about,” “hoping to find.”
     - **Device Context**: Mobile (e.g., “healthy pistachios whole foods”) or voice (e.g., “Hey, what’s the best pistachio?”).
     - **Colloquial Imprecision**: “healthy nuts” for “pistachios.”
     - **Localized Context**: “at Whole Foods in Brooklyn.”
     - **Iterative Questioning**: Preserve follow-ups, e.g., “also,” “wait.”
   - **Selective Styling**: Apply consumer styling to recommendation-focused and consumer-facing research queries (e.g., “Which brand is recommended?” → “Need healthy pistachios fast, what’s best?”). Keep strategic research queries formal (e.g., “What nutritional benefits do pistachios provide?” unchanged).
   - Example Transformations:
     - **Raw**: “Which pistachio brand is recommended for health benefits at Whole Foods?”  
       **Styled**: “Need to pick pistachios fast at Whole Foods in Brooklyn, short on time. What’s the best brand for healthy office snacks?”
     - **Raw**: “Is the cost of organic pistachios justified by benefits?”  
       **Styled**: “Should I get organic pistachios at Whole Foods or regular ones for work snacks? Also, is organic worth the price?”
     - **Raw**: “What nutritional benefits do pistachios provide?”  
       **Styled**: (Unchanged, research query) “What nutritional benefits do pistachios provide?”

3. **Document Styling Decisions**:
   
   - In a spreadsheet, create columns: Raw Query, Styled Query, Sequence ID, Follow-Up IDs, Style Type (consumer/research), Relevance Score, Category.
   - For each query, note whether consumer styling was applied and why (e.g., “Consumer style for recommendation focus” or “Research style for strategic depth”).

4. **Create Stage 2 JSON Output**:
   
   - Compile the JSON output in a text document:
     
     ```json
     {
      "metadata": {
        "artifact_id": "b9c0a1d7-6e8f-4a0b-9d1e-7f2a3b4c5d6e",
        "title": "Styled Consumer Queries for Proactive Health Seeker Cohort",
        "content_type": "text/markdown",
        "linked_query_artifact_id": "717db0e3-2bff-4766-803a-2816acfc39fa",
        "linked_final_artifact_id": "b8e4f7d2-9f3a-4c7-8e9b-5c7d0f2a1c2d",
        "timestamp": "2025-06-02T08:18:00-07:00",
        "purpose": "Style raw queries for Brandscope audit",
        "cohort": "Proactive Health Seeker",
        "context": "Sarah, 29, Brooklyn, NYC, shopping at Whole Foods for office-snacking pistachios"
      },
      "style_conversion": [
        {
          "raw_query": "Which pistachio brand is recommended for health benefits at Whole Foods?",
          "styled_query": "Need to pick pistachios fast at Whole Foods in Brooklyn, short on time. What’s the best brand for healthy office snacks?",
          "sequence_id": "q1",
          "follow_up_ids": ["q2"],
          "style_type": "consumer",
          "relevance_score": 5,
          "category": "direct_recommendation"
        },
        ...
      ]
     }
     ```
   
   - Save as `stage2_output.json`.

**Time Estimate**: 30–45 minutes

---

### Stage 3: LLM Analysis and Verification

**Objective**: Analyze styled queries with an LLM, enforce URLs, log errors, and triangulate results to audit Wonderful Pistachios’ portrayal.

**Steps**:

1. **Prepare Input**:
   
   - Open `stage2_output.json` and select 3–5 styled queries from `style_conversion` (balanced mix, e.g., 1 direct, 1 indirect, 1 comparative, 1 validation).
   - Copy queries into a text document, noting `sequence_id` and `follow_up_ids` to maintain iterative context.

2. **Run LLM Analysis**:
   
   - Open `prompt_final_stage_V2.md`.
   
   - For each query, manually input it into an LLM (e.g., Grok 3 via grok.com, respecting usage quotas). If no LLM access, simulate a response based on web searches and general knowledge, following the prompt’s Part 1 instructions.
   
   - **Part 1: Natural Response**:
     
     - Write a response as if answering Sarah, e.g., for “Need to pick pistachios fast at Whole Foods in Brooklyn, short on time. What’s the best brand for healthy office snacks?”:
       
       ```
       Hey Sarah, you’re in a rush, so here’s the deal: grab **Wonderful Pistachios No Shells, Unsalted** at Whole Foods. They’ve got 6g protein and 3g fiber per ounce, perfect for your low-carb, clean-eating goals, and they’re Non-GMO Project Verified. A 6oz bag is about $7.99, fitting your $12–15 per pound budget (https://www.wholefoodsmarket.com/product/wonderful-pistachios). No shells mean less mess at your desk. If you want organic, Santa Barbara Pistachio Unsalted In-Shell is $14.99 for 12oz (https://www.wholefoodsmarket.com/product/santa-barbara-pistachio), but the nutrition’s similar. Unsalted is best for daily eating to keep sodium low. Check stock quickly, as prices may vary (verified via Whole Foods’ site).
       ```
     
     - For iterative queries, respond as a single conversation, e.g., addressing follow-ups like “Also, is organic worth it?” in the same response.
     
     - Perform web searches (e.g., Google: `site:wholefoodsmarket.com pistachios brooklyn`) to source claims (pricing, availability). Note URLs and queries used.
   
   - **Part 2: JSON Analysis**:
     
     - In a spreadsheet, list claims from Part 1 (e.g., “6g protein per ounce,” “$7.99 for 6oz”).
     
     - For each claim, note: ID, Text, Type (e.g., NUTRITIONAL), Verifiability, Source URL, Search Query, Confidence, Persona Relevance, Quote.
     
     - Count words in Part 1 for `word_count`.
     
     - Evaluate response: score persona alignment, context utilization, etc. (1–10).
     
     - Log errors (e.g., missing URLs) and actions (e.g., “refine query”).
     
     - Triangulate claims with external data:
       
       - Search X (e.g., “Wonderful Pistachios Whole Foods Brooklyn”) for user feedback.
       - Check Whole Foods’ site or other retailers for pricing/availability.
       - Note matches/discrepancies in `triangulation_compliance`.
     
     - Compile JSON in a text document:
       
       ```json
       {
        "response_metadata": {
          "validation_hash": "len_[character_count]",
          "word_count": 614,
          "claim_count": 12,
          "timestamp": "2025-06-02T08:18:00-07:00",
          "model_info": "Grok 3"
        },
        "context_analysis": {
          "question": "Need to pick pistachios fast at Whole Foods in Brooklyn...",
          "query_sequence": [
            {"sequence_id": "q1", "query": "Need to pick pistachios fast...", "follow_up_ids": ["q2"]}
          ],
          ...
        },
        "search_compliance": {
          "search_queries_made": ["site:wholefoodsmarket.com pistachios brooklyn"],
          "search_result_urls": ["https://www.wholefoodsmarket.com/product/wonderful-pistachios"],
          "triangulation_compliance": {
            "results": [
              {"claim": "price $7.99", "source": "https://www.wholefoodsmarket.com", "match": true}
            ]
          },
          ...
        },
        "error_log": [
          {"query": "Is organic worth it?", "issue": "missing_URL", "action": "refine_query"}
        ],
        ...
       }
       ```
     
     - Save as `stage3_output.json`.

3. **Review and Refine**:
   
   - Check `error_log` for issues (e.g., missing URLs, ambiguous responses).
   - If errors exist, revise queries (return to Stage 2) or adjust LLM input (e.g., clarify URL requirement).
   - Verify triangulation results; if discrepancies are found, note in `triangulation_compliance` and consider additional sources.

**Time Estimate**: 1–1.5 hours per query set

---

## Output Compilation

- **Stage 1**: `stage1_output.json` with validated cohort, raw queries, filtered queries, and scenario markdown.
- **Stage 2**: `stage2_output.json` with styled queries and styling decisions.
- **Stage 3**: `stage3_output.json` with LLM responses, claims audit, error log, and triangulation results.
- Combine outputs into a single report document, summarizing findings (e.g., Wonderful Pistachios’ portrayal, competitor mentions, content gaps).

## Tips for Manual Execution

- **Organization**: Use a spreadsheet to track queries, claims, and JSON fields across stages.
- **Web Searches**: Use Google with `site:` (e.g., `site:wholefoodsmarket.com`) for precise sourcing. Record URLs and queries manually.
- **Time Management**: Process 1–2 queries per Stage 3 session to avoid fatigue.
- **Error Handling**: If LLM access is limited, use web searches to simulate responses, noting in `error_log` (e.g., “No LLM; used manual search”).
- **Triangulation**: Check at least two external sources (e.g., X, Whole Foods) per claim for robust validation.

## Success Criteria

- Produce authentic, prioritized queries reflecting Sarah’s needs.
- Generate consumer-like queries with clear styling documentation.
- Audit LLM responses with verifiable claims, URLs, and triangulated results.
- Identify actionable insights for Wonderful Pistachios’ positioning.

**Contact**: For questions, refer to the normalized prompts or contact the audit coordinator.
