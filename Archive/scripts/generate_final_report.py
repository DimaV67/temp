"""
Module for generating a final report from interim LLM evaluation results.

This module aggregates results from multiple LLMs, compares claims to source data,
and produces a structured JSON report summarizing accuracy by category, product,
and provider.
"""

from pathlib import Path
import json
from collections import defaultdict
import datetime
from src.brandscope.database import BrandscopeDB
from scripts.compare_to_source import compare_claims_to_source, calculate_llm_accuracy
from scripts.utils import extract_claims, collect_claims_across_llms, get_latest_truth_file, load_config
import yaml
import logging

logger = logging.getLogger(__name__)

def generate_final_report(company_id: int, config_path: str = None, interim_dir: str = None, output_dir: str = None) -> dict:
    """
    Generate a final report summarizing LLM claim accuracy for a company.

    Args:
        company_id (int): ID of the company.
        config_path (str, optional): Path to the configuration YAML file. If None, uses path from config.yaml.
        interim_dir (str, optional): Directory containing interim JSON files. If None, uses path from config.yaml.
        output_dir (str, optional): Directory to save the final report. If None, uses path from config.yaml.

    Returns:
        dict: The final report dictionary containing accuracy metrics by category,
              product, and provider.
    """
    # Load centralized configuration
    config = load_config()
    config_path = config_path or Path(config['file_paths']['config_dir']) / "config_openinterpreter.yaml"
    interim_dir = interim_dir or config['file_paths']['interim_dir']
    output_dir = output_dir or config['file_paths']['results_dir']
    
    logger.info(f"Generating report with interim_dir={interim_dir}, output_dir={output_dir}")

    db = BrandscopeDB()
    try:
        db.cursor.execute("SELECT url FROM companies WHERE id = ?", (company_id,))
        company = db.cursor.fetchone()
        if not company:
            raise ValueError(f"Company ID {company_id} not found")

        url = company[0]
        truth_file = get_latest_truth_file(url)
        if not truth_file:
            raise ValueError(f"No truth file found for {url}")

        logger.info(f"Using truth file: {truth_file}")
        with open(truth_file, 'r') as f:
            source_json = json.load(f)

        brand = source_json.get("company", {}).get("name", "unknown").lower().replace(' ', '-')
        product_list = [product["product_name"] for product in source_json.get("products", [])][:3]
        if not product_list:
            raise ValueError(f"Truth file {truth_file} has no products")

        logger.info(f"Processing brand: {brand}, products: {product_list}")
        
        with open(config_path, 'r') as f:
            config_yaml = yaml.safe_load(f)

        # Find interim files
        interim_dir = Path(interim_dir)
        interim_files = {
            llm: str(f) for llm, f in [
                ("ollama:llama3.2:latest", next(interim_dir.glob(f"{brand}-ollama-llama3.2-latest-interim-*.json"), None)),
                ("ollama:deepseek-r1:latest", next(interim_dir.glob(f"{brand}-ollama-deepseek-r1-latest-interim-*.json"), None)),
                ("ollama:gemma3:1b", next(interim_dir.glob(f"{brand}-ollama-gemma3-1b-interim-*.json"), None))
            ] if f and f.exists()
        }
        
        if not interim_files:
            raise ValueError(f"No interim files found for {brand} in {interim_dir}")
        
        logger.info(f"Found interim files: {interim_files}")

        # Load interim results
        aggregated_results = {}
        prompt_template_library = config_yaml.get("prompts", [])[:5]
        prompt_id_map = {p["raw"]: p["id"] for p in prompt_template_library}
        
        for llm, file_path in interim_files.items():
            logger.info(f"Processing interim file for {llm}: {file_path}")
            with open(file_path, 'r') as f:
                eval_results = json.load(f)
            
            llm_results = {}
            hash_to_semantic = {}
            
            for prompt in eval_results["results"]["prompts"]:
                raw_prompt = prompt["raw"]
                hashed_id = prompt["id"]
                for raw, semantic_id in prompt_id_map.items():
                    if raw_prompt == raw:
                        hash_to_semantic[hashed_id] = semantic_id
                        break

            results_list = eval_results["results"]["results"]
            for test_result in results_list:
                product = test_result["vars"]["product"]
                hashed_prompt_id = test_result["promptId"]
                semantic_prompt_id = hash_to_semantic.get(hashed_prompt_id, hashed_prompt_id)
                
                # Handle different response formats safely
                response_text = ""
                if isinstance(test_result["response"], dict):
                    # Try different possible keys
                    if "output" in test_result["response"]:
                        response_text = test_result["response"]["output"]
                    elif "content" in test_result["response"]:
                        response_text = test_result["response"]["content"]
                    elif "text" in test_result["response"]:
                        response_text = test_result["response"]["text"]
                    else:
                        # Fallback: convert the entire dict to string
                        response_text = str(test_result["response"])
                        logger.warning(f"Unusual response format: {response_text[:100]}...")
                else:
                    # If it's a string or other type
                    response_text = str(test_result["response"])
                
                if product not in llm_results:
                    llm_results[product] = {}
                llm_results[product][semantic_prompt_id] = response_text
            
            aggregated_results[llm] = llm_results

        logger.info(f"Processed all interim files, creating final report structure")
        
        final_results = {"categories": {}, "products": {}, "brand": {"provider_accuracy": {}}}
        provider_claim_counts = {llm: {"total": 0, "accurate": 0} for llm in interim_files.keys()}

        categories = defaultdict(lambda: defaultdict(list))
        for prompt in prompt_template_library:
            category, question_type, _ = prompt["id"].split(":", 2)
            categories[category][question_type].append(prompt)

        for category, question_types in categories.items():
            final_results["categories"][category] = {}
            for question_type, prompts in question_types.items():
                final_results["categories"][category][question_type] = {}
                for prompt in prompts:
                    prompt_id = prompt["id"]
                    prompt_results = {}

                    for product in product_list:
                        all_claims_with_providers = collect_claims_across_llms(
                            aggregated_results, product, prompt_id, prompt_template_library
                        )
                        logger.info(f"Extracted {len(all_claims_with_providers)} claims for {product}, {prompt_id}")

                        truth_comparison = {}
                        claims = [(claim, provider) for claim, provider in all_claims_with_providers]
                        discrepancies = compare_claims_to_source(claims, source_json)
                        overall_accuracy = calculate_llm_accuracy(discrepancies)

                        for claim, provider in all_claims_with_providers:
                            if claim not in truth_comparison:
                                truth_comparison[claim] = {"agree": [], "disagree": []}
                            match = next((d for d in discrepancies if d["claim"] == claim and d["provider"] == provider), None)
                            provider_claim_counts[provider]["total"] += 1
                            if match and match["is_accurate"]:
                                truth_comparison[claim]["agree"].append(provider)
                                provider_claim_counts[provider]["accurate"] += 1
                            else:
                                truth_comparison[claim]["disagree"].append(provider)

                        all_providers = set(interim_files.keys())
                        for claim in truth_comparison:
                            truth_comparison[claim]["disagree"] = list(set(truth_comparison[claim]["disagree"]) | (all_providers - set(truth_comparison[claim]["agree"])))

                        prompt_results[product] = {
                            "truth_comparison": truth_comparison,
                            "discrepancies": discrepancies,
                            "overall_accuracy": overall_accuracy
                        }

                        if product not in final_results["products"]:
                            final_results["products"][product] = {"prompts": {}, "total_claims": 0, "accurate_claims": 0}
                        final_results["products"][product]["prompts"][prompt_id] = prompt_results[product]
                        final_results["products"][product]["total_claims"] += len(claims)
                        final_results["products"][product]["accurate_claims"] += sum(1 for d in discrepancies if d["is_accurate"])

                    final_results["categories"][category][question_type][prompt_id] = prompt_results

        for product in final_results["products"]:
            total = final_results["products"][product]["total_claims"]
            accurate = final_results["products"][product]["accurate_claims"]
            final_results["products"][product]["accuracy"] = accurate / total if total > 0 else 0.0

        for provider_id, counts in provider_claim_counts.items():
            accuracy = counts["accurate"] / counts["total"] if counts["total"] > 0 else 0.0
            final_results["brand"]["provider_accuracy"][provider_id] = {
                "total_claims": counts["total"],
                "accurate_claims": counts["accurate"],
                "accuracy": accuracy
            }

        total_claims = sum(counts["total"] for counts in provider_claim_counts.values())
        accurate_claims = sum(counts["accurate"] for counts in provider_claim_counts.values())
        final_results["brand"]["overall_accuracy"] = accurate_claims / total_claims if total_claims > 0 else 0.0

        # Save report to file
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{brand}-final-report-{timestamp}.json"
        
        logger.info(f"Saving final report to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Final report saved successfully")

        return final_results

    except Exception as e:
        logger.error(f"Error in generate_final_report: {str(e)}", exc_info=True)
        raise
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate final report from interim LLM results.")
    parser.add_argument("--company-id", type=int, required=True, help="Company ID")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--interim-dir", type=str, help="Interim results directory")
    parser.add_argument("--output-dir", type=str, help="Final report directory")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    final_report = generate_final_report(args.company_id, args.config, args.interim_dir, args.output_dir)
    print(json.dumps(final_report, indent=2))