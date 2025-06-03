#!/bin/bash
# update-dashboard.sh
# Updates code in the running container without rebuilding
# Useful for small changes during development

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

# Print error with timestamp
error() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${RED}$1${NC}"
}

# Check if running from the brandscope root directory
if [ ! -f "pyproject.toml" ] || [ ! -d "dashboard" ]; then
  error "Error: This script must be run from the brandscope root directory"
  error "Please change to the brandscope directory and try again"
  exit 1
fi

# Get container name
CONTAINER_NAME=$(docker ps --filter "name=brandscope-dashboard" --format "{{.Names}}" | head -n 1)

if [ -z "$CONTAINER_NAME" ]; then
  error "No running dashboard container found"
  error "Please start the container first with 'docker compose -f dashboard/docker/compose.yaml up -d'"
  exit 1
fi

log "Found running container: $CONTAINER_NAME"

# Update API files
log "Updating API files in the container..."
docker cp dashboard/api/. $CONTAINER_NAME:/app/dashboard/api/

if [ $? -ne 0 ]; then
  error "Failed to copy API files to container"
  exit 1
fi

# Restart the API service
log "Restarting the API service..."
docker exec $CONTAINER_NAME bash -c "if pgrep -f 'python -m uvicorn' > /dev/null; then kill -HUP \$(pgrep -f 'python -m uvicorn'); else echo 'API service not running, starting it...'; python -m uvicorn dashboard.api.main:app --host 0.0.0.0 --port 8000 & fi"

if [ $? -ne 0 ]; then
  error "Failed to restart API service"
  exit 1
fi

log "Dashboard API updated successfully!"
log "Dashboard is available at: http://localhost:8080"
warn "Note: This update is temporary and will be lost when the container is restarted"
warn "Use rebuild-dashboard.sh to make permanent changes"