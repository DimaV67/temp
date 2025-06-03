#!/bin/bash
# clean-rebuild-dashboard.sh

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${GREEN}$1${NC}"
}

warn() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${YELLOW}$1${NC}"
}

error() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${RED}$1${NC}"
  exit 1
}

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  --core     Rebuild dashboard containers only (shuts down and restarts Ollama)"
  echo "  --ollama   Rebuild only the Ollama container"
  echo "  --supastarter Rebuild only the Supastarter container"
  echo "  --all      Rebuild all containers (dashboard, Ollama, Supastarter)"
  echo "  --help     Show this help message"
  exit 1
}

ensure_network() {
  if ! docker network inspect brandscope-network &>/dev/null; then
    log "Creating brandscope-network..."
    docker network create brandscope-network
  else
    log "Network brandscope-network already exists"
  fi
}

if [ $# -eq 0 ]; then
  usage
fi

REBUILD_CORE=0
REBUILD_OLLAMA=0
REBUILD_SUPASTARTER=0

while [ "$1" != "" ]; do
  case $1 in
    --core)     REBUILD_CORE=1
                ;;
    --ollama)   REBUILD_OLLAMA=1
                ;;
    --supastarter) REBUILD_SUPASTARTER=1
                   ;;
    --all)      REBUILD_CORE=1
                REBUILD_OLLAMA=1
                REBUILD_SUPASTARTER=1
                ;;
    --help)     usage
                ;;
    *)          usage
                ;;
  esac
  shift
done

if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
    error "Docker Compose V2 is required. Please install Docker Desktop or update Docker."
fi

if [ ! -f "pyproject.toml" ] || [ ! -d "dashboard" ] || [ ! -f "config.yaml" ]; then
    error "Error: This script must be run from the brandscope root directory with config.yaml present"
fi

COMPOSE_FILE="dashboard/docker/docker-compose.yaml"

shutdown_containers() {
    log "Shutting down all containers..."
    docker compose -f $COMPOSE_FILE down
}

shutdown_ollama() {
    log "Shutting down Ollama container..."
    OLLAMA_CONTAINER_ID=$(docker ps -aq --filter "name=brandscope-ollama")
    if [ ! -z "$OLLAMA_CONTAINER_ID" ]; then
        log "Stopping existing Ollama container..."
        docker stop -t 30 $OLLAMA_CONTAINER_ID || true
    fi
}

shutdown_supastarter() {
    log "Shutting down Supastarter container..."
    SUPASTARTER_CONTAINER_ID=$(docker ps -aq --filter "name=brandscope-supastarter")
    if [ ! -z "$SUPASTARTER_CONTAINER_ID" ]; then
        log "Stopping existing Supastarter container..."
        docker stop -t 30 $SUPASTARTER_CONTAINER_ID || true
    fi
}

rebuild_dashboard() {
    log "Rebuilding dashboard and dashboard-debug containers..."
    export COMPOSE_DOCKER_CLI_BUILD=1
    export DOCKER_BUILDKIT=1
    
    read -p "Remove Docker volumes? This will delete database data (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Removing Docker volumes..."
        docker compose -f $COMPOSE_FILE down -v
    else
        log "Keeping Docker volumes"
        if [ $REBUILD_OLLAMA -eq 0 ] && [ $REBUILD_SUPASTARTER -eq 0 ]; then
            for CONTAINER_NAME in brandscope-dashboard brandscope-dashboard-debug; do
                CONTAINER_ID=$(docker ps -aq --filter "name=$CONTAINER_NAME")
                if [ ! -z "$CONTAINER_ID" ]; then
                    log "Stopping existing container $CONTAINER_NAME..."
                    docker stop -t 30 $CONTAINER_ID || true
                    log "Removing container $CONTAINER_NAME..."
                    docker rm $CONTAINER_ID || true
                fi
            done
        else
            docker compose -f $COMPOSE_FILE down
        fi
    fi

    log "Removing old images..."
    docker rmi brandscope-dashboard-dashboard brandscope-dashboard-dashboard-debug 2>/dev/null || true
    docker image prune -f --filter "label=com.docker.compose.project=brandscope-dashboard"

    log "Rebuilding dashboard and dashboard-debug containers from scratch..."
    docker compose -f $COMPOSE_FILE build --no-cache --progress plain dashboard dashboard-debug | tee build.log

    if [ $? -ne 0 ]; then
        error "Build failed. Check build.log for details."
    fi
}

rebuild_ollama() {
    log "Rebuilding Ollama container..."
    export COMPOSE_DOCKER_CLI_BUILD=1
    export DOCKER_BUILDKIT=1
    
    OLLAMA_CONTAINER_ID=$(docker ps -aq --filter "name=brandscope-ollama")
    if [ ! -z "$OLLAMA_CONTAINER_ID" ]; then
        log "Stopping existing Ollama container..."
        docker stop -t 30 $OLLAMA_CONTAINER_ID || true
        log "Removing Ollama container..."
        docker rm $OLLAMA_CONTAINER_ID || true
    fi
    
    read -p "Keep Ollama model data? Models are large and take time to download again (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Removing Ollama data volume..."
        docker volume rm brandscope-dashboard_ollama_data 2>/dev/null || true
    else
        log "Keeping Ollama model data"
    fi
    
    log "Building Ollama container..."
    docker compose -f $COMPOSE_FILE build --no-cache ollama
    
    if [ $? -ne 0 ]; then
        error "Ollama build failed"
    fi
}

rebuild_supastarter() {
    log "Rebuilding Supastarter container..."
    export COMPOSE_DOCKER_CLI_BUILD=1
    export DOCKER_BUILDKIT=1
    
    SUPASTARTER_CONTAINER_ID=$(docker ps -aq --filter "name=brandscope-supastarter")
    if [ ! -z "$SUPASTARTER_CONTAINER_ID" ]; then
        log "Stopping existing Supastarter container..."
        docker stop -t 30 $SUPASTARTER_CONTAINER_ID || true
        log "Removing Supastarter container..."
        docker rm $SUPASTARTER_CONTAINER_ID || true
    fi
    
    log "Building Supastarter container..."
    docker compose -f $COMPOSE_FILE build --no-cache supastarter
    
    if [ $? -ne 0 ]; then
        error "Supastarter build failed"
    fi
}

start_dashboard() {
    log "Starting dashboard containers..."
    docker compose -f $COMPOSE_FILE up -d dashboard dashboard-debug
    
    if [ $? -eq 0 ]; then
        log "Dashboard containers started successfully!"
        log "Dashboard is available at: http://localhost:8080"
        log "Checking FastAPI health..."
        sleep 10
        
        HEALTHY=0
        for i in {1..10}; do
            for attempt in {1..3}; do
                if curl -s --fail --connect-timeout 10 --max-time 15 --retry 2 --retry-delay 1 http://localhost:8000/api/v1/health -o /dev/null; then
                    log "FastAPI is healthy"
                    HEALTHY=1
                    break
                else
                    warn "FastAPI not ready, attempt $attempt/3 for retry $i/10..."
                    sleep 2
                fi
            done
            if [ "$HEALTHY" = "1" ]; then
                break
            fi
            warn "FastAPI not ready, retrying ($i/10)..."
            sleep 5
            if [ $i -eq 10 ]; then
                error "FastAPI health check failed after 10 retries"
            fi
        done
    else
        error "Failed to start dashboard containers"
    fi
}

start_ollama() {
    log "Starting Ollama container..."
    docker compose -f $COMPOSE_FILE up -d ollama
    
    if [ $? -eq 0 ]; then
        log "Ollama container started successfully!"
        log "Checking if Ollama needs model downloads..."
        sleep 5
        for model in "llama3.2:latest" "deepseek-r1:latest" "gemma3:1b"; do
            if ! docker exec brandscope-ollama ollama list | grep -q "$model"; then
                log "Pulling Ollama model: $model (this may take a while)..."
                docker exec brandscope-ollama ollama pull $model
            else
                log "Ollama model $model already exists"
            fi
        done
        log "Ollama API is available at: http://localhost:11434"
    else
        error "Failed to start Ollama container"
    fi
}

start_supastarter() {
    log "Starting Supastarter container..."
    docker compose -f $COMPOSE_FILE up -d supastarter
    
    if [ $? -eq 0 ]; then
        log "Supastarter container started successfully!"
        log "Supastarter frontend is available at: http://localhost:3000"
    else
        error "Failed to start Supastarter container"
    fi
}

ensure_network

if [ $REBUILD_CORE -eq 1 ] && [ $REBUILD_OLLAMA -eq 0 ] && [ $REBUILD_SUPASTARTER -eq 0 ]; then
    shutdown_ollama
    shutdown_supastarter
    rebuild_dashboard
    start_dashboard
    start_ollama
    start_supastarter
elif [ $REBUILD_CORE -eq 0 ] && [ $REBUILD_OLLAMA -eq 1 ] && [ $REBUILD_SUPASTARTER -eq 0 ]; then
    rebuild_ollama
    start_ollama
elif [ $REBUILD_CORE -eq 0 ] && [ $REBUILD_OLLAMA -eq 0 ] && [ $REBUILD_SUPASTARTER -eq 1 ]; then
    rebuild_supastarter
    start_supastarter
elif [ $REBUILD_CORE -eq 1 ] && [ $REBUILD_OLLAMA -eq 1 ] && [ $REBUILD_SUPASTARTER -eq 1 ]; then
    shutdown_containers
    rebuild_dashboard
    rebuild_ollama
    rebuild_supastarter
    start_dashboard
    start_ollama
    start_supastarter
fi

log "All requested rebuilds complete!"