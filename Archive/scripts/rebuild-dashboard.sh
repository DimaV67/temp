#!/bin/bash
# rebuild-dashboard.sh
# Rebuilds and restarts the dashboard container

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

# Get container ID (if it exists)
CONTAINER_ID=$(docker ps -q --filter "name=brandscope-dashboard")

# Stop the container if it exists
if [ ! -z "$CONTAINER_ID" ]; then
  log "Stopping existing container..."
  docker stop $CONTAINER_ID
fi

# Rebuild and start the container
log "Rebuilding dashboard container..."
docker compose -f dashboard/docker/compose.yaml build

if [ $? -eq 0 ]; then
  log "Build successful, starting container..."
  docker compose -f dashboard/docker/compose.yaml up -d

  if [ $? -eq 0 ]; then
    log "Container started successfully!"
    log "Dashboard is available at: http://localhost:8080"
  else
    error "Failed to start container"
    exit 1
  fi
else
  error "Build failed"
  exit 1
fi