# dashboard/api/data.py
"""
Data processing module for the Brandscope Dashboard API.
Reads and transforms data from the Brandscope results directory and PostgreSQL database.
Triggers workflow processes for truth file generation, interim results, and final reports.
"""

import json
import os
import socket
import uuid
from datetime import datetime
from pathlib import Path
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from scripts.generate_truth_file import generate_truth_file
from scripts.generate_interim_results import generate_interim_results
from scripts.generate_final_report import generate_final_report
import asyncio
from scripts.utils import load_config
from dashboard.tasks.truth_tasks import generate_truth_file_task
from dashboard.tasks.interim_tasks import generate_interim_results_task
from dashboard.tasks.report_tasks import generate_final_report_task
from celery.result import AsyncResult
from scripts.utils import check_ollama_connectivity
import psycopg2
from psycopg2.extras import RealDictCursor
from .database import get_db

# Configure logging
config = load_config()
log_dir = Path(config['file_paths']['log_dir'])
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "api.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter()

class BrandscopeDataConnector:
    """Connector for Brandscope result data from JSON files."""
    
    def __init__(self, results_dir=None):
        """Initialize with path to results directory."""
        config = load_config()
        self.results_dir = Path(results_dir or config['file_paths']['results_dir'])
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured results directory exists: {self.results_dir}")
        
        if not self.results_dir.exists():
            logger.warning(f"Results directory still not found after creation attempt: {self.results_dir}")
    
    def get_available_brands(self):
        """Get list of brands that have final reports."""
        try:
            brand_files = list(self.results_dir.glob("*-final-report-*.json"))
            brands = set()
            
            for file in brand_files:
                filename = file.name
                if '-final-report-' in filename:
                    brand_name = filename.split('-final-report-')[0]
                    brands.add(brand_name)
            
            return sorted(list(brands))
        except Exception as e:
            logger.error(f"Error getting available brands: {e}")
            return []
    
    def get_brand_report(self, brand_name):
        """Get the latest report for a specific brand, returning the full JSON."""
        try:
            reports = list(self.results_dir.glob(f"{brand_name}-final-report-*.json"))
            
            if not reports:
                logger.warning(f"No reports found for brand: {brand_name}")
                return None
            
            reports.sort(key=lambda x: os.path.getctime(x), reverse=True)
            
            with open(reports[0], 'r') as f:
                raw_data = json.load(f)
            
            logger.info(f"Returning raw data for brand {brand_name}: {raw_data.keys()}")
            raw_data["name"] = brand_name.replace("-", " ").title()
            raw_data["date"] = datetime.now().strftime("%Y-%m-%d")
            return raw_data
        except Exception as e:
            logger.error(f"Error retrieving brand report: {e}")
            return None
    
    def get_model_comparison(self, brand_name):
        """Get model comparison data for a specific brand."""
        try:
            model_files = list(self.results_dir.glob(f"{brand_name}-ollama-*-results.json"))
            
            if not model_files:
                logger.warning(f"No model files found for brand: {brand_name}")
                return []
            
            models_data = []
            
            for file in model_files:
                model_name = file.name.split(f"{brand_name}-ollama-")[1].split("-results.json")[0]
                
                with open(file, 'r') as f:
                    model_data = json.load(f)
                
                model_summary = {
                    "name": model_name,
                    "accuracy": model_data.get("accuracy", 0),
                    "product_count": len(model_data.get("products", [])),
                    "attributes_detected": len(model_data.get("attributes", [])),
                }
                
                models_data.append(model_summary)
            
            return models_data
        except Exception as e:
            logger.error(f"Error retrieving model comparison: {e}")
            return []

# Initialize data connector
data_connector = BrandscopeDataConnector()

# Pydantic models for PostgreSQL request validation
class UserCreate(BaseModel):
    username: str
    email: str

class CompanyCreate(BaseModel):
    name: str
    url: str
    industry: str | None = None

class BrandCreate(BaseModel):
    company_id: str
    name: str
    url: str
    industry: str | None = None

class ProductCreate(BaseModel):
    brand_id: str
    name: str
    url: str

class UserCompanyAccessCreate(BaseModel):
    user_id: str
    company_id: str
    access_level: str

# Pydantic models for workflow orchestration
class TruthRequest(BaseModel):
    company_id: str
    brand_id: str

class InterimRequest(BaseModel):
    company_id: str
    brand_id: Optional[str] = None
    llms: Optional[List[str]] = None

class ReportRequest(BaseModel):
    company_id: str

# Existing routes
@router.get("/brands")
async def get_brands():
    """Get list of available brands from JSON reports."""
    brands = data_connector.get_available_brands()
    if not brands:
        raise HTTPException(status_code=404, detail="No brands found")
    return brands

@router.get("/report/{brand_name}")
async def get_report(brand_name: str):
    """Get the latest report for a specific brand from JSON."""
    report = data_connector.get_brand_report(brand_name)
    if not report:
        raise HTTPException(status_code=404, detail=f"No report found for brand: {brand_name}")
    return report

@router.get("/model_comparison/{brand_name}")
async def get_model_comparison(brand_name: str):
    """Get model comparison data for a specific brand from JSON."""
    comparison = data_connector.get_model_comparison(brand_name)
    if not comparison:
        raise HTTPException(status_code=404, detail=f"No model comparison data found for brand: {brand_name}")
    return comparison

@router.post("/users")
async def create_user(user: UserCreate, db: psycopg2.extensions.connection = Depends(get_db)):
    """Create a new user in PostgreSQL."""
    try:
        user_id = str(uuid.uuid4())
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO profiles (user_id, username, email) VALUES (%s, %s, %s) RETURNING user_id",
                (user_id, user.username, user.email)
            )
            result = cur.fetchone()
            db.commit()
        return {"user_id": result["user_id"], "username": user.username, "email": user.email}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/companies")
async def create_company(company: CompanyCreate, db: psycopg2.extensions.connection = Depends(get_db)):
    """Create a new agency in PostgreSQL."""
    try:
        agency_id = str(uuid.uuid4())
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO agencies (id, name, url, industry) VALUES (%s, %s, %s, %s) RETURNING id",
                (agency_id, company.name, company.url, company.industry)
            )
            result = cur.fetchone()
            db.commit()
        return {"company_id": result["id"], "name": company.name, "url": company.url, "industry": company.industry}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/brands")
async def create_brand(brand: BrandCreate, db: psycopg2.extensions.connection = Depends(get_db)):
    """Create a new brand in PostgreSQL."""
    try:
        brand_id = str(uuid.uuid4())
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO brands (id, agency_id, name, url, industry) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (brand_id, brand.company_id, brand.name, brand.url, brand.industry)
            )
            result = cur.fetchone()
            db.commit()
        return {"brand_id": result["id"], "company_id": brand.company_id, "name": brand.name, "url": brand.url}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/products")
async def create_product(product: ProductCreate, db: psycopg2.extensions.connection = Depends(get_db)):
    """Create a new product in PostgreSQL."""
    try:
        product_id = str(uuid.uuid4())
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO products (id, brand_id, name, url) VALUES (%s, %s, %s, %s) RETURNING id",
                (product_id, product.brand_id, product.name, product.url)
            )
            result = cur.fetchone()
            db.commit()
        return {"product_id": result["id"], "brand_id": product.brand_id, "name": product.name, "url": product.url}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/user_company_access")
async def create_user_company_access(access: UserCompanyAccessCreate, db: psycopg2.extensions.connection = Depends(get_db)):
    """Grant a user access to an agency in PostgreSQL."""
    try:
        with db.cursor() as cur:
            cur.execute(
                "UPDATE profiles SET agency_id = %s WHERE user_id = %s",
                (access.company_id, access.user_id)
            )
            if cur.rowcount == 0:
                raise ValueError("User or agency not found")
            db.commit()
        return {"message": "User access created", "user_id": access.user_id, "company_id": access.company_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/companies/{user_id}")
async def get_companies(user_id: str, db: psycopg2.extensions.connection = Depends(get_db)):
    """Get agencies a user has access to from PostgreSQL."""
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT a.id, a.name, a.url, a.industry FROM agencies a JOIN profiles p ON a.id = p.agency_id WHERE p.user_id = %s",
                (user_id,)
            )
            companies = cur.fetchall()
        if not companies:
            raise HTTPException(status_code=404, detail=f"No agencies found for user_id: {user_id}")
        return companies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/brands/{company_id}")
async def get_brands(company_id: str, db: psycopg2.extensions.connection = Depends(get_db)):
    """Get brands for an agency from PostgreSQL."""
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT id, name, url, industry FROM brands WHERE agency_id = %s",
                (company_id,)
            )
            brands = cur.fetchall()
        if not brands:
            raise HTTPException(status_code=404, detail=f"No brands found for company_id: {company_id}")
        return brands
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/products/{brand_id}")
async def get_products(brand_id: str, db: psycopg2.extensions.connection = Depends(get_db)):
    """Get products for a brand from PostgreSQL."""
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT id, name, url FROM products WHERE brand_id = %s",
                (brand_id,)
            )
            products = cur.fetchall()
        if not products:
            raise HTTPException(status_code=404, detail=f"No products found for brand_id: {brand_id}")
        return products
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/llm_results/{brand_id}")
async def get_llm_results(brand_id: str, db: psycopg2.extensions.connection = Depends(get_db)):
    """Get LLM results for a brand from PostgreSQL."""
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT llm, sentiment, themes, products, visibility, pricing, subscription FROM llm_results WHERE brand_id = %s",
                (brand_id,)
            )
            results = cur.fetchall()
        if not results:
            # Mock data for testing (replace with actual DB data)
            return [
                {
                    "llm": "ClaudeAI",
                    "sentiment": 0.8,
                    "themes": ["quality", "price"],
                    "products": 19,
                    "visibility": "100%",
                    "pricing": [
                        {"product": "Glow Maker", "range": "$27.95-$32.95", "variance": "17.9%", "impact": "High"},
                        {"product": "Moonlight Retinal", "range": "$34.95-$46.95", "variance": "34.3%", "impact": "Severe"}
                    ],
                    "subscription": {"visibility": "Not mentioned", "discount_accuracy": "N/A", "impact": "Recurring revenue invisibility"}
                },
                {
                    "llm": "Grok 3",
                    "sentiment": 0.7,
                    "themes": ["service"],
                    "products": 7,
                    "visibility": "37%",
                    "pricing": [
                        {"product": "Glow Maker", "range": "$27.95-$32.95", "variance": "17.9%", "impact": "High"},
                        {"product": "Moonlight Retinal", "range": "$34.95-$46.95", "variance": "34.3%", "impact": "Severe"}
                    ],
                    "subscription": {"visibility": "Mentioned", "discount_accuracy": "Inaccurate", "impact": "Potential disappointment"}
                }
            ]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user_access/{user_id}")
async def get_user_access(user_id: str, db: psycopg2.extensions.connection = Depends(get_db)):
    """Get user access to agencies from PostgreSQL."""
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT agency_id FROM profiles WHERE user_id = %s",
                (user_id,)
            )
            access = cur.fetchall()
        if not access:
            raise HTTPException(status_code=404, detail=f"No access found for user_id: {user_id}")
        return access
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Workflow orchestration endpoints
@router.post("/generate-truth")
async def generate_truth(request: TruthRequest):
    """Generate a truth JSON file for a company and brand."""
    try:
        config = load_config()
        truth_dir = config['file_paths']['truth_dir']
        Path(truth_dir).mkdir(parents=True, exist_ok=True)
        output_file = await asyncio.to_thread(generate_truth_file, request.company_id, request.brand_id, truth_dir)
        return {"message": "Truth file generated", "output_file": str(output_file)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating truth file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/generate-interim")
async def generate_interim(request: InterimRequest):
    """Generate interim LLM evaluation results for a company, optionally filtered by brand and LLMs."""
    try:
        interim_files = generate_interim_results(request.company_id, request.brand_id, request.llms)
        if not interim_files:
            raise ValueError("No interim files generated")
        return {"message": "Interim results generated", "interim_files": interim_files}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating interim results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/generate-report")
async def generate_report(request: ReportRequest):
    """Generate a final report for a company."""
    try:
        report = await asyncio.to_thread(generate_final_report, request.company_id)
        return {"message": "Final report generated", "report": report}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating final report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/tasks/truth-file")
async def create_truth_file_task(company_id: str, brand_id: str):
    """Start async task to generate truth file."""
    task = generate_truth_file_task.delay(company_id, brand_id)
    return {"task_id": task.id, "status": "started"}

@router.post("/tasks/interim-results")
async def create_interim_results_task(company_id: str, brand_id: Optional[str] = None, llms: Optional[str] = None):
    """Start async task to generate interim results."""
    llm_list = llms.split(",") if llms else None
    task = generate_interim_results_task.delay(company_id, brand_id, llm_list)
    return {"task_id": task.id, "status": "started"}

@router.post("/tasks/final-report")
async def create_final_report_task(company_id: str):
    """Start async task to generate final report."""
    task = generate_final_report_task.delay(company_id)
    return {"task_id": task.id, "status": "started"}

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Check status of an async task."""
    task_result = AsyncResult(task_id)
    response = {
        "task_id": task_id,
        "status": task_result.status,
    }
    
    if task_result.ready():
        if task_result.successful():
            response["result"] = task_result.result
        else:
            response["error"] = str(task_result.result)
    
    return response

@router.get("/ollama-status")
async def check_ollama_status():
    """Check connectivity to Ollama service using the same method as generate_interim_results."""
    ollama_status = check_ollama_connectivity()
    if ollama_status["status"] != "success":
        logger.error(f"Ollama connectivity test failed: {ollama_status}")
        return None
    return ollama_status