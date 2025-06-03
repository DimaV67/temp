# scripts/utils.py
"""
Utility functions for claim extraction and data processing.

This module provides helper functions for extracting claims from text responses,
collecting claims across LLMs, and locating the latest truth file for a given URL.
"""

import re
import logging
import os
from pathlib import Path
from urllib.parse import urlparse
import yaml
import socket
import requests
import json

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "utils.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is invalid.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        raise

def extract_claims(response: str, keywords: list = None) -> list:
    """
    Extract factual claims from a text response, filtering out non-factual content.

    Args:
        response (str): The text response to process.
        keywords (list, optional): Not currently used, reserved for future filtering.

    Returns:
        list: List of cleaned claim strings, with non-factual content removed.
    """
    if not response or not isinstance(response, str):
        print(f"Invalid response: {response}")
        return []

    # Remove <think> tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    print(f"Processed response: {response[:200]}...")

    lines = response.split('\n')
    claims = []

    for line in lines:
        line = line.strip()
        if not line:
            print(f"Skipping empty line")
            continue
        if line.endswith('?'):
            print(f"Skipping question: {line}")
            continue
        # Skip non-factual content (instructions, disclaimers, intros)
        if line.lower().startswith((
            "i couldn't find", "i don't have", "sorry", "disclaimer:", "important note:",
            "please note", "i recommend", "if you", "to give you", "let me know", "consider",
            "here are some", "based on general", "i can provide", "it's possible", "i apologize",
            "okay, here are", "to help me tailor", "it’s always best", "based on the product"
        )) or "**Note:**" in line or "*Note:*" in line:
            print(f"Skipping non-factual line: {line}")
            continue
        # Strip bullet/number prefixes and formatting (e.g., **bold**)
        clean_line = re.sub(r'^\s*[\*\-•]|\d+\.\s*|\*{1,2}|\*{1,2}$', '', line).strip()
        if clean_line:
            print(f"Keeping claim: {clean_line}")
            claims.append(clean_line)

    if not claims:
        print(f"No claims extracted from response: {response[:200]}...")

    return claims

def collect_claims_across_llms(aggregated_results: dict, product: str, question: str, prompt_template_library: list) -> list:
    """
    Collect claims for a specific product and question across all LLMs.

    Args:
        aggregated_results (dict): Dictionary of LLM results, mapping LLM IDs to
            product and prompt responses.
        product (str): The product name to filter claims for.
        question (str): The prompt ID to filter claims for.
        prompt_template_library (list): List of prompt templates for metadata.

    Returns:
        list: List of tuples, each containing a claim (str) and its provider (str).
    """
    all_claims_with_providers = []
    for llm, llm_results in aggregated_results.items():
        product_claims = llm_results.get(product, {})
        print(f"Checking {llm}, {product}, {question}: {list(product_claims.keys())}")
        response = product_claims.get(question, None)
        if response:
            prompt = next((p for p in prompt_template_library if p["id"] == question), None)
            print(f"Found response for {llm}, {product}, {question}: {response[:200]}...")
            claims = extract_claims(response, None)
            print(f"Extracted {len(claims)} claims: {claims}")
            for claim in claims:
                all_claims_with_providers.append((claim, llm))
        else:
            print(f"No response found for {llm}, {product}, {question}")
    return all_claims_with_providers

def get_latest_truth_file(url: str) -> str | None:
    """
    Get the latest truth JSON file for a given URL.

    Args:
        url (str): URL of the website (e.g., 'https://maelove.com').

    Returns:
        str | None: Path to the latest truth file, or None if no files are found.
    """
    try:
        domain = url.split("://")[-1].split("/")[0]
        truth_dir = Path(os.getenv("TRUTH_DIR", Path(__file__).parent.parent / "truth")) / domain
        logger.info(f"Looking for truth files in {truth_dir}")

        if not truth_dir.exists():
            logger.error(f"Truth directory {truth_dir} does not exist")
            return None

        truth_files = list(truth_dir.glob("*.json"))
        if not truth_files:
            logger.warning(f"No truth files found in {truth_dir}")
            return None

        latest_file = max(truth_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found latest truth file: {latest_file}")
        return str(latest_file)

    except Exception as e:
        logger.error(f"Error getting latest truth file: {e}")
        return None

def check_ollama_connectivity():
    """
    Check Ollama connectivity and basic functionality.
    Performs socket test, API version check, and a basic generate call.
    Returns a dictionary with status ('success' or 'error') and details.
    """
    ollama_host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")

    result = {"status": "error", "host": ollama_host}
    logger.info(f"Checking Ollama connectivity at {ollama_host}")

    # 1. Socket test
    try:
        # Use urlparse for more robust host/port extraction from the URL
        parsed_url = urlparse(ollama_host)
        host = parsed_url.hostname
        # Determine port based on scheme if not explicitly provided
        port = parsed_url.port if parsed_url.port else (80 if parsed_url.scheme == 'http' else (443 if parsed_url.scheme == 'https' else None))

        if not host or not port:
            # If parsing failed or port is ambiguous (e.g., missing for https), try simple split as fallback
            try:
                clean_host = ollama_host.replace("http://", "").replace("https://", "")
                parts = clean_host.split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else (80 if "http://" in ollama_host else (443 if "https://" in ollama_host else None))
                if not host or not port:
                    raise ValueError("Could not determine host or port")
            except Exception:
                raise ValueError(f"Could not parse host or port from {ollama_host}")

        logger.debug(f"Performing socket test to {host}:{port}")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)  # Increased timeout slightly
        s.connect((host, port))
        s.close()
        logger.debug("Socket test successful")

    except (socket.error, ValueError) as e:
        result["error_step"] = "socket_test"
        result["error_message"] = f"Socket connection failed: {e}"
        logger.error(f"{result['error_step']} failed: {result['error_message']}")
        return result
    except Exception as e:
        result["error_step"] = "socket_test"
        result["error_message"] = f"Unexpected error during socket test: {e}"
        logger.error(f"{result['error_step']} failed: {result['error_message']}")
        return result

    # 2. API tests (/api/version and /api/generate)
    try:
        # API test 1: Get version
        logger.debug(f"Performing GET to {ollama_host}/api/version")
        version_response = requests.get(f"{ollama_host}/api/version", timeout=15)
        version_response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        logger.debug("/api/version test successful")

        # API test 2: Generate test
        logger.debug(f"Performing POST to {ollama_host}/api/generate")
        # Use lowercase 'false' for JSON boolean value
        generate_payload = {"model": "llama3.2:latest", "prompt": "Hello", "stream": False}  # Use lowercase false
        generate_response = requests.post(
            f"{ollama_host}/api/generate",
            json=generate_payload,
            timeout=20  # Increased timeout for model response
        )
        generate_response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        # Optional: Basic check of the generate response structure
        try:
            generate_response_data = generate_response.json()
            if not isinstance(generate_response_data, dict) or 'response' not in generate_response_data:
                logger.warning(f"Generate API response structure unexpected: {generate_response_data}")
                # Depending on strictness, you might want to fail here
                # result["error_step"] = "generate_api_response_check"
                # result["error_message"] = "Generate API response missing 'response' field or unexpected format"
                # return result
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from /api/generate response. Response text snippet: {generate_response.text[:200]}...")
            # Depending on strictness, you might want to fail here
            # result["error_step"] = "generate_api_json_decode"
            # result["error_message"] = "Could not decode JSON from /api/generate response"
            # return result

        logger.debug("/api/generate test successful")

    except requests.exceptions.RequestException as e:
        result["error_step"] = "api_test"
        result["error_message"] = f"API test failed: {e}"
        logger.error(f"{result['error_step']} failed: {result['error_message']}")
        return result
    except Exception as e:
        result["error_step"] = "api_test"
        result["error_message"] = f"Unexpected error during API test: {e}"
        logger.error(f"{result['error_step']} failed: {result['error_message']}")
        return result

    # If all tests pass
    result["status"] = "success"
    logger.info("Ollama connectivity and functionality check successful")
    return result

def get_llm_connection_config(provider_id, env=None):
    """
    Build connection configuration for different LLM providers.
    
    Args:
        provider_id (str): The provider ID string (e.g., "ollama:llama3.2:latest")
        env (dict, optional): Environment dictionary to update with provider-specific variables
    
    Returns:
        tuple: (config_dict, updated_env) where config_dict contains provider configuration
               and updated_env contains any environment variables needed for this provider
    """
    # Get base Ollama host from environment or use default
    ollama_host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
    
    # Initialize config and environment
    config = {}
    updated_env = env.copy() if env is not None else os.environ.copy()
    
    # Set timeout for all providers
    config["timeout"] = 120000
    
    # Set provider-specific configuration
    if provider_id.startswith("ollama:"):
        # For Ollama, we need to set the environment variable
        updated_env["OLLAMA_BASE_URL"] = ollama_host
        
    elif provider_id.startswith("anthropic:"):
        # For Anthropic, set the appropriate configuration
        config["apiBase"] = "https://api.anthropic.com/v1"
        config["apiKey"] = os.environ.get("ANTHROPIC_API_KEY", "")
        
    elif provider_id.startswith("openai:"):
        # For OpenAI, set the appropriate configuration
        config["apiBase"] = "https://api.openai.com/v1"
        config["apiKey"] = os.environ.get("OPENAI_API_KEY", "")
        
    else:
        # Default case for other providers
        config["apiBase"] = ollama_host
    
    return config, updated_env