#!/bin/bash
# debug-helper.sh - Helper script for debugging brandscope containers

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print with timestamp
log() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${GREEN}$1${NC}"
}

# Print warning with timestamp
warn() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${YELLOW}$1${NC}"
}

# Error and exit
error() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${RED}$1${NC}"
  exit 1
}

# Usage info
usage() {
  echo "Debug helper for Brandscope containers"
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  --check     Check debug status of containers"
  echo "  --restart   Restart debug container"
  echo "  --logs      View debug container logs"
  echo "  --interim   Debug generate_interim_results.py"
  echo "  --exec      Execute shell in debug container"
  echo "  --help      Show this help message"
  exit 1
}

# Check if running from the brandscope root directory
if [ ! -f "pyproject.toml" ] || [ ! -d "dashboard" ] || [ ! -f "config.yaml" ]; then
    error "Error: This script must be run from the brandscope root directory with config.yaml present"
fi

# Parse arguments
if [ $# -eq 0 ]; then
  usage
fi

ACTION=""

while [ "$1" != "" ]; do
  case $1 in
    --check)    ACTION="check"
                ;;
    --restart)  ACTION="restart"
                ;;
    --logs)     ACTION="logs"
                ;;
    --interim)  ACTION="interim"
                ;;
    --exec)     ACTION="exec"
                ;;
    --help)     usage
                ;;
    *)          usage
                ;;
  esac
  shift
done

# Check debug status
check_debug() {
  log "Checking debug status..."
  
  # Check if debug container is running
  if ! docker ps | grep -q brandscope-dashboard-debug; then
    warn "Debug container is not running"
  else
    log "Debug container is running"
    
    # Check if debugpy is accessible
    if ! nc -z localhost 5678 >/dev/null 2>&1; then
      warn "debugpy port 5678 is not accessible"
    else
      log "debugpy port 5678 is accessible"
    fi
    
    # Print container info
    docker ps --filter "name=brandscope-dashboard-debug" --format "table {{.ID}}\t{{.Status}}\t{{.Ports}}"
  fi
  
  # Check FastAPI status
  if ! curl -s --fail --connect-timeout 5 --max-time 10 http://localhost:8000/api/v1/health >/dev/null 2>&1; then
    warn "FastAPI API is not healthy"
  else
    log "FastAPI API is healthy"
  fi
}

# Restart debug container
restart_debug() {
  log "Restarting debug container..."
  
  # Stop the container
  CONTAINER_ID=$(docker ps -aq --filter "name=brandscope-dashboard-debug")
  if [ ! -z "$CONTAINER_ID" ]; then
    log "Stopping existing container..."
    docker stop -t 30 $CONTAINER_ID || true
    log "Removing container..."
    docker rm $CONTAINER_ID || true
  fi
  
  # Start the container
  log "Starting debug container..."
  docker compose -f dashboard/docker/docker-compose.yaml up -d dashboard-debug
  
  if [ $? -eq 0 ]; then
    log "Debug container started successfully"
    sleep 5
    check_debug
  else
    error "Failed to start debug container"
  fi
}

# View debug logs
view_logs() {
  log "Viewing debug container logs..."
  docker logs -f brandscope-dashboard-debug
}

# Debug interim results
debug_interim() {
  log "Debugging generate_interim_results.py..."
  
  # First, check if debug container is running
  if ! docker ps | grep -q brandscope-dashboard-debug; then
    warn "Debug container is not running, starting it..."
    restart_debug
  fi
  
  # Copy debug_interim.py to container if it doesn't exist
  if ! docker exec brandscope-dashboard-debug test -f /app/debug_interim.py; then
    log "Copying debug_interim.py to container..."
    cat << 'EOF' > /tmp/debug_interim.py
#!/usr/bin/env python
"""
Debug wrapper for generate_interim_results.py
This script enables direct debugging of the generate_interim_results.py script.
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
EOF
    docker cp /tmp/debug_interim.py brandscope-dashboard-debug:/app/debug_interim.py
    docker exec brandscope-dashboard-debug chmod +x /app/debug_interim.py
  fi
  
  # Prompt for parameters
  read -p "Enter company_id [1]: " company_id
  company_id=${company_id:-1}
  
  read -p "Enter brand_id (leave empty for None): " brand_id
  
  read -p "Enter LLMs (comma-separated) [ollama:llama3.2:latest]: " llms
  llms=${llms:-"ollama:llama3.2:latest"}
  
  # Set environment variables and run the debug script
  log "Starting debug session for generate_interim_results.py..."
  log "Parameters: company_id=$company_id, brand_id=$brand_id, llms=$llms"
  log "Attach your debugger to localhost:5678 now, then press Enter to continue..."
  read -p ""
  
  # Run with the specified parameters
  if [ -z "$brand_id" ]; then
    docker exec -it \
      -e DEBUG_COMPANY_ID=$company_id \
      -e DEBUG_LLMS=$llms \
      brandscope-dashboard-debug python /app/debug_interim.py
  else
    docker exec -it \
      -e DEBUG_COMPANY_ID=$company_id \
      -e DEBUG_BRAND_ID=$brand_id \
      -e DEBUG_LLMS=$llms \
      brandscope-dashboard-debug python /app/debug_interim.py
  fi
}

# Execute shell in debug container
exec_shell() {
  log "Opening shell in debug container..."
  docker exec -it brandscope-dashboard-debug bash
}

# Execute the selected action
case $ACTION in
  "check")    check_debug
              ;;
  "restart")  restart_debug
              ;;
  "logs")     view_logs
              ;;
  "interim")  debug_interim
              ;;
  "exec")     exec_shell
              ;;
esac

exit 0