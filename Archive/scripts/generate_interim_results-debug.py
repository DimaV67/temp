#!/usr/bin/env python
"""
Debug wrapper for generate_interim_results.py
This script enables direct debugging of the generate_interim_results.py script.

Usage:
  docker exec -it brandscope-dashboard-debug python /app/debug_interim.py
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting debug wrapper for generate_interim_results.py")
    
    # Initialize debugpy
    try:
        import debugpy
        logger.info("Starting debugpy server on 0.0.0.0:5678")
        debugpy.listen(("0.0.0.0", 5678))
        logger.info("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        logger.info("Debugger attached!")
    except ImportError:
        logger.error("debugpy not installed. Install with: pip install debugpy")
        return 1
    except Exception as e:
        logger.error(f"Failed to start debugpy: {e}")
        return 1
    
    # Import the actual function after debugpy is listening
    try:
        logger.info("Importing generate_interim_results...")
        from scripts.generate_interim_results import generate_interim_results
    except ImportError as e:
        logger.error(f"Failed to import generate_interim_results: {e}")
        logger.error(f"Current sys.path: {sys.path}")
        return 1
    
    # Get parameters from environment or use defaults
    company_id = int(os.environ.get("DEBUG_COMPANY_ID", "1"))
    brand_id = int(os.environ.get("DEBUG_BRAND_ID", "1")) if os.environ.get("DEBUG_BRAND_ID") else None
    llm_str = os.environ.get("DEBUG_LLMS", "ollama:llama3.2:latest")
    llms = llm_str.split(",") if llm_str else None
    
    logger.info(f"Starting generate_interim_results with parameters:")
    logger.info(f"  company_id = {company_id}")
    logger.info(f"  brand_id = {brand_id}")
    logger.info(f"  llms = {llms}")
    
    # Execute the function
    try:
        result = generate_interim_results(company_id, brand_id, llms)
        logger.info(f"Function completed with result: {result}")
        return 0
    except Exception as e:
        logger.error(f"Function execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())