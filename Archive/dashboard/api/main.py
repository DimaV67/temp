# dashboard/api/main.py
"""
API server for the LLM Competitive Intelligence Dashboard.
Serves brand analysis results from the brandscope system and PostgreSQL database.
"""

import os
import time
import sys
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .data import router as data_router
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from .data import router as data_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start debugpy if DEBUG environment variable is set
if os.environ.get("DEBUG", "0") == "1":
    try:
        import debugpy
        logger.info("Starting debugpy on 0.0.0.0:5678")
        debugpy.listen(("0.0.0.0", 5678))
        # Uncomment the line below if you want to pause execution 
        # until a debugger attaches (recommended for debugging on startup)
        # debugpy.wait_for_client()
        logger.info("debugpy started and listening for connections")
    except ImportError:
        logger.error("debugpy not installed, please install with: pip install debugpy")
    except Exception as e:
        logger.error(f"Failed to initialize debugpy: {e}")

# Log Python path to help with debugging import issues
logger.info(f"Python path: {sys.path}")

app = FastAPI(title="Brandscope Dashboard API")

# Log when FastAPI app is created
logger.info("FastAPI app created")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include data router
app.include_router(data_router)

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup event triggered")
    routes = [{"path": route.path, "methods": list(route.methods)} for route in app.routes if hasattr(route, "path") and hasattr(route, "methods")]
    logger.info("Registered routes at startup: %s", routes)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Brandscope Dashboard API"}

@app.get("/api/v1/health")
async def health():
    logger.info("Health endpoint accessed")
    return {"status": "ok"}

# Debug endpoint
@app.get("/debug/info")
async def debug_info():
    """Returns debug information about the environment"""
    debug_info = {
        "pythonPath": sys.path,
        "environment": dict(os.environ),
        "debugEnabled": os.environ.get("DEBUG", "0") == "1"
    }
    return debug_info

# Debug endpoint to list all routes
@app.get("/debug/routes")
async def debug_routes():
    routes = [{"path": route.path, "methods": list(route.methods)} for route in app.routes if hasattr(route, "path") and hasattr(route, "methods")]
    logger.info("Debug routes accessed. Registered routes: %s", routes)
    return {"routes": routes}

# Catch-all route for undefined endpoints
@app.get("/{path:path}")
async def catch_all(path: str, request: Request):
    logger.info("Catch-all accessed for path: %s", path)
    return JSONResponse(status_code=404, content={"detail": f"Endpoint not found: {path}"})

@app.post("/{path:path}")
async def catch_all_post(path: str, request: Request):
    logger.info("Catch-all (POST) accessed for path: %s", path)
    return JSONResponse(status_code=404, content={"detail": f"Endpoint not found: {path}"})