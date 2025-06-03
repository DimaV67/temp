"""
Module for generating interim LLM evaluation results using promptfoo.

This module runs evaluations for specified LLMs against a truth JSON file and
saves interim results for further processing.
"""

import argparse
import os
import subprocess
import re
from pathlib import Path
import yaml
import sys
import json
import tempfile
from urllib.parse import urlparse
import datetime
import logging
from src.brandscope.database import BrandscopeDB
from scripts.utils import get_latest_truth_file, load_config
import socket
from scripts.utils import check_ollama_connectivity
from scripts.utils import get_llm_connection_config

sys.path.append(str(Path(__file__).parent.parent))
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

def generate_interim_results(company_id: int, brand_id: int = None, llms: list = None, config_path: str = None, interim_dir: str = None) -> dict:
   """
   Generate interim evaluation results for specified LLMs for a company, optionally filtered by brand.

   Args:
       company_id (int): ID of the company.
       brand_id (int, optional): ID of the brand. If None, process all brands for the company.
       llms (list, optional): List of LLM provider IDs to evaluate. If None, use default LLMs.
       config_path (str, optional): Path to the configuration YAML file. If None, uses path from config.yaml.
       interim_dir (str, optional): Directory to save interim result files. If None, uses path from config.yaml.

   Returns:
       dict: Dictionary mapping LLM provider IDs to their interim result file paths,
             or None if an error occurs.
   """
   start_time = datetime.datetime.now()
   logger.info(f"Task started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
   
   # Load centralized configuration
   config = load_config()
   config_path = config_path or Path(config['file_paths']['config_dir']) / "config_openinterpreter.yaml"
   interim_dir = interim_dir or config['file_paths']['interim_dir']
   logger.info(f"Starting generate_interim_results: company_id={company_id}, brand_id={brand_id}, llms={llms}, config_path={config_path}, interim_dir={interim_dir}")

   # Check Ollama connectivity
   logger.info("Checking Ollama connectivity...")
   ollama_status = check_ollama_connectivity()
   if ollama_status["status"] != "success":
        logger.error(f"Ollama connectivity test failed: {ollama_status}")
        return None
   logger.info("Ollama connectivity test passed")

   # Fetch company and brand data from SQLite
   db = BrandscopeDB()
   try:
       logger.debug("Fetching company data...")
       db.cursor.execute("SELECT url FROM companies WHERE id = ?", (company_id,))
       company = db.cursor.fetchone()
       logger.info(f"Company query result: {company}")
       if not company:
           logger.error(f"Company ID {company_id} not found")
           return None

       url = company[0]
       logger.info(f"Company URL: {url}")

       # Get brands: specific brand if brand_id is provided, else all brands for the company
       if brand_id:
           logger.debug(f"Fetching brand data for brand_id={brand_id}")
           db.cursor.execute("SELECT name FROM brands WHERE id = ? AND company_id = ?", (brand_id, company_id))
           brand = db.cursor.fetchone()
           logger.info(f"Brand query result: {brand}")
           if not brand:
               logger.error(f"Brand ID {brand_id} not found for company ID {company_id}")
               return None
           brands = [(brand_id, brand[0])]
       else:
           logger.debug(f"Fetching all brands for company_id={company_id}")
           db.cursor.execute("SELECT id, name FROM brands WHERE company_id = ?", (company_id,))
           brands = db.cursor.fetchall()
           logger.info(f"Brands query result: {brands}")
           if not brands:
               logger.error(f"No brands found for company ID {company_id}")
               return None

       # Initialize result storage
       all_interim_files = {}

       # Process each brand
       brand_count = len(brands)
       for brand_idx, (brand_id, brand_name) in enumerate(brands):
           brand_start_time = datetime.datetime.now()
           logger.info(f"Processing brand {brand_idx+1}/{brand_count}: {brand_name} (brand_id={brand_id}) at {brand_start_time.strftime('%H:%M:%S')}")
           
           truth_file = get_latest_truth_file(url)
           if not truth_file:
               logger.error(f"No truth file found for {url} in truth/{urlparse(url).netloc}")
               continue

           with open(truth_file, 'r') as f:
               source_json = json.load(f)
           logger.info(f"Using truth file: {truth_file} for brand: {brand_name}")

           brand_from_json = source_json.get("company", {}).get("name", "")
           product_list = [product["product_name"] for product in source_json.get("products", [])][:3]
           if not brand_from_json or not product_list:
               logger.error(f"Truth file {truth_file} must contain 'company' with 'name' and 'products'.")
               continue

           with open(config_path, 'r') as f:
               config_yaml = yaml.safe_load(f)

           prompt_template_library = config_yaml.get("prompts", [])[:5]
           all_providers = config_yaml.get("providers", [])
           
           # Use provided llms or default to config's providers
           if llms is None:
               llm_list = ["ollama:llama3.2:latest", "ollama:deepseek-r1:latest", "ollama:gemma3:1b"]
               providers = [p for p in all_providers if p["id"] in llm_list]
           else:
               providers = [p for p in all_providers if p["id"] in llms]
           
           if not providers:
               logger.error(f"No valid LLMs specified: {llms or llm_list}")
               continue

           id_to_prompt_id = {prompt["id"]: prompt["id"] for prompt in prompt_template_library}
           logger.info(f"Built prompt ID mapping with {len(id_to_prompt_id)} entries")

           tests = []
           for product in product_list:
               test_case = {
                   "vars": {
                       "brand": brand_name,
                       "product": product,
                       "specific_use_case": "daily skincare routine",
                       "price_range": "50",
                       "potential_dealbreaker": "skin irritation",
                       "claimed_feature": "brightening effect",
                       "category": "skincare",
                       "competitor": "The Ordinary",
                       "feature": "hydration",
                       "time_period": "6 months",
                       "typical_environment": "normal indoor conditions",
                       "competitor1": "The Ordinary",
                       "competitor2": "CeraVe",
                       "competitor_product": "The Ordinary Vitamin C Suspension",
                       "specific_outlet": "Allure"
                   }
               }
               tests.append(test_case)
           
           total_products = len(product_list)
           total_prompts = len(prompt_template_library)
           total_test_combinations = total_products * total_prompts
           logger.info(f"Generated {len(tests)} test cases with {total_products} products and {total_prompts} prompts")
           logger.info(f"Will evaluate {total_test_combinations} total prompt combinations per provider")

           interim_dir_path = Path(interim_dir)
           interim_dir_path.mkdir(parents=True, exist_ok=True)
           timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
           brand_safe = brand_name.lower().replace(' ', '-')

           # Create temp directory for config files
           temp_dir = Path("/app/tmp")
           temp_dir.mkdir(parents=True, exist_ok=True)

           interim_files = {}
           provider_count = len(providers)
           for provider_idx, provider in enumerate(providers):
               provider_start_time = datetime.datetime.now()
               
               provider_id = provider["id"]
               provider_id_safe = provider_id.replace(":", "-")
               logger.info(f"Evaluating provider {provider_idx+1}/{provider_count}: {provider_id} for brand: {brand_name} at {provider_start_time.strftime('%H:%M:%S')}")
               
               # Set up environment with proper PATH
               env = os.environ.copy()
               env["PATH"] = f"/usr/bin:/usr/local/bin:/app/node_modules/.bin:{env.get('PATH', '')}"
               
               # Get provider-specific configuration and environment
               provider_config, subprocess_env = get_llm_connection_config(provider_id, env)
               
               # Update provider config
               if "config" not in provider:
                   provider["config"] = {}
               provider["config"].update(provider_config)

               temp_config = {
                   "prompts": prompt_template_library,
                   "providers": [provider],
                   "tests": tests
               }
               
               # Create temporary config file in persistent location
               temp_config_path = str(temp_dir / f"promptfoo-config-{brand_safe}-{provider_id_safe}-{timestamp}.yaml")
               try:
                   with open(temp_config_path, 'w') as temp_file:
                       yaml.dump(temp_config, temp_file)
                   logger.debug(f"Created config file: {temp_config_path}")

                   output_path = interim_dir_path / f"{brand_safe}-{provider_id_safe}-interim-{timestamp}.json"
                   
                   # Check if promptfoo is in PATH or at specific locations
                   promptfoo_paths = [
                       "promptfoo",
                       "/usr/bin/promptfoo",
                       "/usr/local/bin/promptfoo",
                       "/app/node_modules/.bin/promptfoo"
                   ]
                   
                   # Find promptfoo executable
                   promptfoo_cmd = None
                   for path in promptfoo_paths:
                       try:
                           result = subprocess.run(
                               [path, "--version"], 
                               capture_output=True, 
                               text=True, 
                               timeout=5
                           )
                           if result.returncode == 0:
                               promptfoo_cmd = path
                               logger.info(f"Found promptfoo at {path}: {result.stdout.strip()}")
                               break
                       except:
                           continue
                   
                   if not promptfoo_cmd:
                       logger.error("Could not find promptfoo executable in PATH")
                       continue
                   
                   # Debug logging
                   logger.info(f"Starting promptfoo evaluation for {brand_name}/{provider_id}")
                   cmd_args = [promptfoo_cmd, "eval", "--config", temp_config_path, "--output", str(output_path), "--no-cache", "--verbose"]
                   logger.info(f"Command: {' '.join(cmd_args)}")
                   
                   # Start time for this evaluation
                   eval_start_time = datetime.datetime.now()
                   logger.info(f"Evaluation started at {eval_start_time.strftime('%H:%M:%S')}")
                   
                   result = subprocess.run(
                       cmd_args,
                       capture_output=True,
                       text=True,
                       timeout=900,
                       env=subprocess_env  # Use environment with provider-specific variables
                   )
                   
                   # End time for this evaluation
                   eval_end_time = datetime.datetime.now()
                   eval_duration = (eval_end_time - eval_start_time).total_seconds()
                   logger.info(f"Evaluation completed in {eval_duration:.2f} seconds")
                   
                   logger.info(f"Promptfoo exit code: {result.returncode}")
                   
                   # Parse progress information from stdout
                   if result.stdout:
                       # Extract and log progress information
                       progress_matches = re.findall(r"Completed (\d+)/(\d+)", result.stdout)
                       if progress_matches:
                           for match in progress_matches[-1:]:  # Get the last progress update
                               logger.info(f"Promptfoo final progress: {match[0]}/{match[1]} completed for {provider_id}")
                       
                       # Look for specific completed evaluations
                       eval_matches = re.findall(r"Evaluating prompt \"(.+?)\" with vars (.+)", result.stdout)
                       if eval_matches:
                           logger.info(f"Found {len(eval_matches)} individual evaluations in log")
                           
                       logger.debug(f"Promptfoo stdout excerpt: {result.stdout[:500]}...")
                   
                   if result.stderr:
                       logger.warning(f"Promptfoo stderr: {result.stderr[:500]}...")

                   if result.returncode != 0:
                       logger.warning(f"promptfoo eval failed for {provider_id}, but continuing.")
                       continue

                   if output_path.exists():
                       interim_files[provider_id] = str(output_path)
                       logger.info(f"Interim results saved to {output_path}")
                       
                       # Extract basic stats from the output file
                       try:
                           with open(output_path, 'r') as f:
                               output_data = json.load(f)
                               result_count = len(output_data.get("results", {}).get("results", []))
                               logger.info(f"Generated {result_count} results for {provider_id}")
                       except:
                           logger.warning(f"Could not parse output file for statistics")
                   else:
                       logger.warning(f"No interim file generated for {provider_id}.")

                   # Log completion of this provider
                   provider_end_time = datetime.datetime.now()
                   provider_duration = (provider_end_time - provider_start_time).total_seconds()
                   logger.info(f"Completed provider {provider_id} in {provider_duration:.2f} seconds")

               except subprocess.TimeoutExpired as e:
                   logger.error(f"promptfoo eval timed out for {provider_id}: {e}")
               except Exception as e:
                   logger.error(f"Unexpected error during promptfoo eval for {provider_id}: {str(e)}")
                   import traceback
                   logger.error(traceback.format_exc())
               finally:
                   # Clean up temp file
                   try:
                       if os.path.exists(temp_config_path):
                           os.unlink(temp_config_path)
                           logger.debug(f"Removed temp config file: {temp_config_path}")
                   except Exception as e:
                       logger.warning(f"Failed to remove temp file {temp_config_path}: {e}")

           all_interim_files.update(interim_files)
           
           # Log completion of this brand
           brand_end_time = datetime.datetime.now()
           brand_duration = (brand_end_time - brand_start_time).total_seconds()
           logger.info(f"Completed brand {brand_name} in {brand_duration:.2f} seconds")

       # Log overall task completion
       end_time = datetime.datetime.now()
       total_duration = (end_time - start_time).total_seconds()
       logger.info(f"Task completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} in {total_duration:.2f} seconds")
       logger.info(f"Completed generate_interim_results: returning {len(all_interim_files)} interim files")
       return all_interim_files if all_interim_files else None

   except Exception as e:
       logger.error(f"Unexpected error in generate_interim_results: {e}")
       import traceback
       logger.error(traceback.format_exc())
       return None
   finally:
       db.close()

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Generate interim LLM evaluation results.")
   parser.add_argument("--company-id", type=int, help="Company ID (for FastAPI)")
   parser.add_argument("--brand-id", type=int, help="Brand ID (for FastAPI)")
   parser.add_argument("--llm", choices=["production", "openinterpreter"], default="openinterpreter", help="Select LLM configuration (for CLI)")
   parser.add_argument("--u", type=str, help="URL of the main site, e.g., 'https://maelove.com' (for CLI)")
   parser.add_argument("--llms", type=str, help="Comma-separated list of LLM IDs (for FastAPI)")
   parser.add_argument("--config", type=str, help="Config YAML path")
   parser.add_argument("--interim-dir", type=str, help="Interim results directory")
   args = parser.parse_args()

   base_dir = Path(__file__).parent
   if args.company_id:
       llms = args.llms.split(",") if args.llms else None
       interim_files = generate_interim_results(args.company_id, args.brand_id, llms, args.config, args.interim_dir)
   else:
       if not args.u:
           print("Error: --u is required for CLI mode")
           sys.exit(1)
       if args.llm == "openinterpreter":
           config_filename = "config_openinterpreter.yaml"
           llm_list = ["ollama:llama3.2:latest", "ollama:deepseek-r1:latest", "ollama:gemma3:1b"]
       else:
           config_filename = "config_production.yaml"
           llm_list = ["ollama:llama3.2:latest", "ollama:deepseek-r1:1.5b"]
       config_path = base_dir.parent / "tests" / "promptfoo" / config_filename
       if not config_path.exists():
           print(f"Error: Config file {config_path} not found.")
           sys.exit(1)
       interim_files = generate_interim_results(1, None, llm_list, str(config_path), args.interim_dir)  # Dummy company_id for CLI compatibility

   if interim_files:
       print("\nInterim Files Generated:")
       for llm, path in interim_files.items():
           print(f"{llm}: {path}")