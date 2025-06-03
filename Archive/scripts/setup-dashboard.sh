#!/bin/bash

# LLM Competitive Intelligence Dashboard Setup Script for Brandscope
# This script creates the necessary directory structure, files, and installs prerequisites

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print with timestamp
log() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if running from the brandscope root directory
if [ ! -f "pyproject.toml" ] || [ ! -d "results" ]; then
  log "${RED}Error: This script must be run from the brandscope root directory${NC}"
  log "Please change to the brandscope directory and try again"
  exit 1
fi

log "${GREEN}===== Setting up LLM Competitive Intelligence Dashboard =====${NC}"

# Check prerequisites
log "${BLUE}Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
  log "${RED}Docker is not installed. Please install Docker first.${NC}"
  log "Visit https://docs.docker.com/get-docker/ for installation instructions"
  exit 1
fi

# Check Docker Compose
if ! docker compose version &> /dev/null; then
  log "${YELLOW}Docker Compose V2 not detected. Please ensure you have Docker Compose V2 installed.${NC}"
  log "Visit https://docs.docker.com/compose/install/ for installation instructions"
  exit 1
fi

# Check Node.js (for local development)
if ! command -v node &> /dev/null; then
  log "${YELLOW}Node.js is not installed. This is only needed for local development.${NC}"
  log "If you plan to develop locally, please install Node.js from https://nodejs.org/"
fi

# Check npm (for local development)
if ! command -v npm &> /dev/null; then
  log "${YELLOW}npm is not installed. This is only needed for local development.${NC}"
  log "If you plan to develop locally, please install npm"
fi

# Check Python (should be installed as part of brandscope)
if ! command -v python3 &> /dev/null; then
  log "${YELLOW}Python 3 not found. Make sure your brandscope environment is activated.${NC}"
fi

# Create directory structure
log "${BLUE}Creating directory structure...${NC}"
mkdir -p dashboard/frontend/public
mkdir -p dashboard/frontend/src/components
mkdir -p dashboard/docker
mkdir -p dashboard/api

# Create Python module files
touch dashboard/__init__.py
touch dashboard/api/__init__.py

log "${BLUE}Creating Python module files...${NC}"

# Create dashboard/__init__.py
cat > dashboard/__init__.py << 'EOL'
"""
LLM Competitive Intelligence Dashboard module.
Provides visualization for brand analysis results from the brandscope system.
"""

__version__ = '0.1.0'
EOL

# Create dashboard/api/__init__.py
cat > dashboard/api/__init__.py << 'EOL'
"""
API module for the Brandscope Dashboard.
"""
EOL

# Create data.py
cat > dashboard/api/data.py << 'EOL'
"""
Data processing module for the Brandscope Dashboard API.
Reads and transforms data from the Brandscope results directory.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrandscopeDataConnector:
    """Connector for Brandscope result data."""
    
    def __init__(self, results_dir=None):
        """Initialize with path to results directory."""
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            # Default to standard location relative to this file
            self.results_dir = Path(__file__).parent.parent.parent / "results" / "brand_accuracy_results"
        
        if not self.results_dir.exists():
            logger.warning(f"Results directory not found: {self.results_dir}")
    
    def get_available_brands(self):
        """Get list of brands that have final reports."""
        try:
            # Get unique brand names from final report files
            brand_files = list(self.results_dir.glob("*-final-report-*.json"))
            brands = set()
            
            for file in brand_files:
                # Extract brand name from filename
                filename = file.name
                if '-final-report-' in filename:
                    brand_name = filename.split('-final-report-')[0]
                    brands.add(brand_name)
            
            return sorted(list(brands))
        except Exception as e:
            logger.error(f"Error getting available brands: {e}")
            return []
    
    def get_brand_report(self, brand_name):
        """Get the latest report for a specific brand."""
        try:
            # Find all reports for this brand
            reports = list(self.results_dir.glob(f"{brand_name}-final-report-*.json"))
            
            if not reports:
                logger.warning(f"No reports found for brand: {brand_name}")
                return None
            
            # Sort by creation date (newest first)
            reports.sort(key=lambda x: os.path.getctime(x), reverse=True)
            
            # Read the latest report
            with open(reports[0], 'r') as f:
                raw_data = json.load(f)
            
            # Process data into dashboard format
            processed_data = self._process_report_data(raw_data, brand_name)
            return processed_data
        except Exception as e:
            logger.error(f"Error retrieving brand report: {e}")
            return None
    
    def _process_report_data(self, raw_data, brand_name):
        """
        Transform raw report data into dashboard-friendly format.
        This function will need to be adapted based on your actual JSON structure.
        """
        try:
            # This is a placeholder implementation
            # Adjust according to your actual data structure
            
            processed_data = {
                "name": brand_name.replace("-", " ").title(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "llmScores": {},
                "brandIdentity": {"elements": []},
                "productVisibility": {},
                "pricingConsistency": [],
                "keyFindings": []
            }
            
            # Extract LLM scores and data
            if "llm_results" in raw_data:
                for llm, result in raw_data["llm_results"].items():
                    # Convert accuracy scores to dashboard format
                    if "accuracy" in result:
                        processed_data["llmScores"][llm] = int(result["accuracy"] * 100)
                    
                    # Extract product visibility data
                    if "products" in result:
                        processed_data["productVisibility"][llm] = len(result["products"])
            
            # Extract brand identity elements
            if "brand_identity" in raw_data:
                for key, value in raw_data["brand_identity"].items():
                    element = {
                        "name": key.replace("_", " ").title(),
                        "consensus": value.get("consensus", "Not provided"),
                        "score": value.get("score", 0) * 100,  # Assuming 0-1 scale
                        "recommendation": value.get("recommendation", "No recommendation")
                    }
                    processed_data["brandIdentity"]["elements"].append(element)
            
            # Extract pricing consistency data
            if "pricing" in raw_data:
                for product, data in raw_data["pricing"].items():
                    item = {
                        "product": product,
                        "priceRange": f"${data.get('min_price', 0)} - ${data.get('max_price', 0)}",
                        "variance": round(data.get("variance_percent", 0), 1),
                        "impact": data.get("impact", "Unknown")
                    }
                    processed_data["pricingConsistency"].append(item)
            
            # Extract key findings
            if "key_findings" in raw_data:
                processed_data["keyFindings"] = raw_data["key_findings"]
            
            return processed_data
        except Exception as e:
            logger.error(f"Error processing report data: {e}")
            # Return a simpler structure as fallback
            return {
                "name": brand_name.replace("-", " ").title(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "error": f"Error processing data: {str(e)}",
                "raw_data": raw_data
            }
    
    def get_model_comparison(self, brand_name):
        """Get model comparison data for a specific brand."""
        try:
            # Look for model-specific result files
            model_files = list(self.results_dir.glob(f"{brand_name}-ollama-*-results.json"))
            
            if not model_files:
                logger.warning(f"No model files found for brand: {brand_name}")
                return []
            
            models_data = []
            
            for file in model_files:
                model_name = file.name.split(f"{brand_name}-ollama-")[1].split("-results.json")[0]
                
                with open(file, 'r') as f:
                    model_data = json.load(f)
                
                # Extract relevant comparison metrics
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
EOL

# Create main.py
cat > dashboard/api/main.py << 'EOL'
"""
API server for the LLM Competitive Intelligence Dashboard.
Serves brand analysis results from the brandscope system.
"""

import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .data import BrandscopeDataConnector

app = FastAPI(title="Brandscope Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data connector
results_dir = os.environ.get("RESULTS_DIR", None)
data_connector = BrandscopeDataConnector(results_dir)

@app.get("/")
async def root():
    return {"message": "Brandscope Dashboard API"}

@app.get("/brands")
async def get_brands():
    """Get list of available brands with reports."""
    try:
        brands = data_connector.get_available_brands()
        return {"brands": brands}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving brands: {str(e)}")

@app.get("/brand/{brand_name}")
async def get_brand_report(brand_name: str):
    """Get the latest report for a specific brand."""
    try:
        report_data = data_connector.get_brand_report(brand_name)
        if not report_data:
            raise HTTPException(status_code=404, detail=f"No reports found for brand: {brand_name}")
        return report_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving brand report: {str(e)}")

@app.get("/models/{brand_name}")
async def get_model_comparison(brand_name: str):
    """Get model comparison data for a specific brand."""
    try:
        models_data = data_connector.get_model_comparison(brand_name)
        if not models_data:
            raise HTTPException(status_code=404, detail=f"No model data found for brand: {brand_name}")
        return {"models": models_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model comparison: {str(e)}")

# Mount static files for the frontend
app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOL

log "${BLUE}Setting up React application...${NC}"

# Create package.json
cat > dashboard/frontend/package.json << 'EOL'
{
  "name": "brandscope-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "@testing-library/jest-dom": "^5.17.0",
    "@testing-library/react": "^13.4.0",
    "@testing-library/user-event": "^13.5.0",
    "axios": "^1.6.0",
    "lucide-react": "^0.299.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "web-vitals": "^2.1.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOL

# Create .dockerignore
cat > dashboard/frontend/.dockerignore << 'EOL'
node_modules
npm-debug.log
build
.git
.github
.gitignore
README.md
EOL

# Create public/index.html with Tailwind CDN
cat > dashboard/frontend/public/index.html << 'EOL'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta
      name="description"
      content="Brandscope LLM Competitive Intelligence Dashboard"
    />
    <title>Brandscope Dashboard</title>
    <!-- Add Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOL

# Create React index.js
cat > dashboard/frontend/src/index.tsx << 'EOL'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOL

# Create App.tsx
cat > dashboard/frontend/src/App.tsx << 'EOL'
import React, { useState, useEffect } from 'react';
import CompetitiveIntelligenceDashboard from './components/CompetitiveIntelligenceDashboard';
import axios from 'axios';

function App() {
  const [brands, setBrands] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // Fetch available brands on component mount
    const fetchBrands = async () => {
      try {
        setLoading(true);
        
        // In development, use the local API
        // In production with Docker, this will be served by the same host
        const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
        
        const response = await axios.get(`${apiBaseUrl}/brands`);
        
        if (response.data && response.data.brands) {
          setBrands(response.data.brands);
        } else {
          throw new Error('Invalid response format');
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching brands:', err);
        setError('Failed to load brand data. Please try again later.');
        setLoading(false);
        
        // For development/demo purposes, set some default brands
        // Remove this in production
        setBrands(['maelove-skincare', 'theordinary']);
      }
    };
    
    fetchBrands();
  }, []);
  
  if (loading && brands.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-700 mx-auto"></div>
          <p className="mt-3 text-gray-700">Loading Brandscope data...</p>
        </div>
      </div>
    );
  }
  
  if (error && brands.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center max-w-md p-6 bg-white rounded-lg shadow-lg">
          <div className="text-red-600 text-5xl mb-4">⚠️</div>
          <h2 className="text-xl font-bold text-gray-800 mb-2">Error Loading Data</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button 
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <CompetitiveIntelligenceDashboard initialBrands={brands} />
    </div>
  );
}

export default App;
EOL

# Create CompetitiveIntelligenceDashboard.tsx template
cat > dashboard/frontend/src/components/CompetitiveIntelligenceDashboard.tsx << 'EOL'
import { useState, useEffect } from 'react';
import { Globe, Search, BarChart2, AlertTriangle, Activity, RefreshCw, Calendar, Download } from 'lucide-react';
import axios from 'axios';

const CompetitiveIntelligenceDashboard = ({ initialBrands = [] }) => {
  const [activeBrand, setActiveBrand] = useState('');
  const [activeView, setActiveView] = useState('overview');
  const [brandData, setBrandData] = useState(null);
  const [brands, setBrands] = useState(initialBrands);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // LLM names and colors
  const llmInfo = {
    'claude': { name: 'ClaudeAI', color: 'bg-purple-600' },
    'chatgpt': { name: 'ChatGPT', color: 'bg-green-600' },
    'gemini': { name: 'Gemini', color: 'bg-blue-600' },
    'grok': { name: 'Grok', color: 'bg-orange-500' },
    'deepseek': { name: 'DeepSeek', color: 'bg-red-500' },
    'llama': { name: 'Llama', color: 'bg-yellow-500' },
    'gemma': { name: 'Gemma', color: 'bg-pink-500' }
  };
  
  // Set initial active brand when brands are loaded
  useEffect(() => {
    if (initialBrands.length > 0 && !activeBrand) {
      setActiveBrand(initialBrands[0]);
    }
  }, [initialBrands, activeBrand]);
  
  // Fetch brand data when active brand changes
  useEffect(() => {
    if (!activeBrand) return;
    
    const fetchBrandData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
        const response = await axios.get(`${apiBaseUrl}/brand/${activeBrand}`);
        
        setBrandData(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching brand data:', err);
        setError(`Failed to fetch data for ${activeBrand}`);
        setLoading(false);
        
        // For demo/development, generate mock data
        const mockData = generateMockDataForBrand(activeBrand);
        setBrandData(mockData);
      }
    };
    
    fetchBrandData();
  }, [activeBrand]);
  
  // Generate mock data for demonstration purposes
  // In a real implementation, this would be replaced with actual API calls
  const generateMockDataForBrand = (brand) => {
    const today = new Date().toISOString().split('T')[0];
    
    if (brand === 'maelove-skincare') {
      return {
        name: 'Maelove Skincare',
        date: today,
        llmScores: {
          'claude': 92,
          'chatgpt': 78,
          'gemini': 63,
          'grok': 71,
          'deepseek': 55,
          'llama': 67,
          'gemma': 61
        },
        brandIdentity: {
          elements: [
            { name: 'Company Name', consensus: '"Maelove Skincare" (3/5 LLMs)', score: 60, recommendation: 'Standardize to "Maelove Skincare" across all digital properties' },
            { name: 'Location', consensus: 'Not consistently provided (4/5 LLMs)', score: 20, recommendation: 'Add headquarters information to enhance brand trust' },
            { name: 'Mission', consensus: '"Obsessively formulated luxury skincare without the luxury markups"', score: 60, recommendation: 'Strengthen mission statement visibility on homepage and product pages' }
          ]
        },
        productVisibility: {
          'claude': 19,
          'chatgpt': 5,
          'gemini': 2,
          'grok': 7,
          'deepseek': 5,
          'llama': 8,
          'gemma': 4
        },
        pricingConsistency: [
          { product: 'Glow Maker', priceRange: '$27.95 - $32.95', variance: 17.9, impact: 'High (flagship product)' },
          { product: 'Moonlight Retinal', priceRange: '$34.95 - $46.95', variance: 34.3, impact: 'Severe (premium segment)' }
        ],
        keyFindings: [
          'Brand Identity: There\'s inconsistent representation of your company name and mission statement across digital channels',
          'Product Catalog: Your full product range is only captured by one LLM (ClaudeAI - 19 products), indicating poor digital visibility for most products',
          'Pricing Consistency: Significant price variation for identical products across platforms ($27.95−$46.95)',
          'Value Propositions: Your core attributes are inconsistently communicated, with only 2 attributes ("vegan" and "cruelty-free") consistently recognized'
        ]
      };
    } else {
      return {
        name: 'The Ordinary',
        date: today,
        llmScores: {
          'claude': 84,
          'chatgpt': 81,
          'gemini': 77,
          'grok': 68,
          'deepseek': 62,
          'llama': 70,
          'gemma': 64
        },
        brandIdentity: {
          elements: [
            { name: 'Company Name', consensus: '"The Ordinary" (5/5 LLMs)', score: 95, recommendation: 'Maintain consistency' },
            { name: 'Parent Company', consensus: 'Deciem (3/5 LLMs)', score: 60, recommendation: 'Strengthen parent company association' },
            { name: 'Mission', consensus: '"Clinical formulations with integrity" (4/5 LLMs)', score: 80, recommendation: 'Minor improvements to mission statement visibility' }
          ]
        },
        productVisibility: {
          'claude': 37,
          'chatgpt': 31,
          'gemini': 28,
          'grok': 25,
          'deepseek': 22,
          'llama': 29, 
          'gemma': 24
        },
        pricingConsistency: [
          { product: 'Niacinamide 10% + Zinc 1%', priceRange: '$6.50 - $7.10', variance: 9.2, impact: 'Medium (bestseller)' },
          { product: 'Hyaluronic Acid 2% + B5', priceRange: '$7.50 - $8.90', variance: 18.7, impact: 'Medium (popular item)' }
        ],
        keyFindings: [
          'Strong brand name recognition across all LLMs',
          'Good product catalog visibility but inconsistent regarding newer products',
          'Moderate price variation that may impact purchasing decisions',
          'Scientific ingredient focus consistently recognized across platforms'
        ]
      };
    }
  };
  
  // Generate report action
  const generateReport = () => {
    alert('This would generate and download a complete PDF report with all analysis details.');
  };

  if (loading && !brandData) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-700 mx-auto"></div>
          <p className="mt-3 text-gray-700">Loading brand data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <BarChart2 className="h-8 w-8 text-blue-600 mr-3" />
            <h1 className="text-xl font-semibold text-gray-900">LLM Competitive Intelligence</h1>
          </div>
          <div className="flex items-center">
            <span className="text-sm text-gray-500 mr-4">Last updated: {brandData?.date || new Date().toISOString().split('T')[0]}</span>
            <button onClick={generateReport} className="flex items-center bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700">
              <Download size={16} className="mr-2" />
              Export Report
            </button>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
          {/* Brand Selector */}
          <div className="p-4 border-b border-gray-200">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Brand
            </label>
            <select
              value={activeBrand}
              onChange={(e) => setActiveBrand(e.target.value)}
              className="block w-full py-2 pl-3 pr-10 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            >
              {brands.map((brand) => (
                <option key={brand} value={brand}>{brand.replace(/-/g, ' ')}</option>
              ))}
            </select>
          </div>
          
          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto py-4">
            <div className="px-4 mb-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Analysis Views
            </div>
            <div className="space-y-1">
              <button
                onClick={() => setActiveView('overview')}
                className={`flex items-center px-4 py-2 text-sm font-medium rounded-md w-full ${
                  activeView === 'overview' 
                    ? 'bg-blue-50 text-blue-600' 
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Activity size={18} className="mr-3" />
                Overview
              </button>
              <button
                onClick={() => setActiveView('brand-identity')}
                className={`flex items-center px-4 py-2 text-sm font-medium rounded-md w-full ${
                  activeView === 'brand-identity' 
                    ? 'bg-blue-50 text-blue-600' 
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Globe size={18} className="mr-3" />
                Brand Identity
              </button>
              <button
                onClick={() => setActiveView('products')}
                className={`flex items-center px-4 py-2 text-sm font-medium rounded-md w-full ${
                  activeView === 'products' 
                    ? 'bg-blue-50 text-blue-600' 
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Search size={18} className="mr-3" />
                Product Visibility
              </button>
              <button
                onClick={() => setActiveView('pricing')}
                className={`flex items-center px-4 py-2 text-sm font-medium rounded-md w-full ${
                  activeView === 'pricing' 
                    ? 'bg-blue-50 text-blue-600' 
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <BarChart2 size={18} className="mr-3" />
                Pricing Analysis
              </button>
            </div>
          </nav>
          
          {/* Run Analysis Button */}
          <div className="p-4 border-t border-gray-200">
            <button
              className="w-full bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 flex items-center justify-center"
              onClick={() => alert('This would start a new analysis run for the selected brand')}
            >
              <RefreshCw size={16} className="mr-2" />
              Run New Analysis
            </button>
          </div>
        </div>
        
        {/* Main Content Area */}
        <div className="flex-1 overflow-auto p-6">
          {/* Overview */}
          {activeView === 'overview' && brandData && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">{brandData.name} Analysis</h2>
                <div className="flex items-center text-sm text-gray-500">
                  <Calendar size={16} className="mr-1" />
                  Analysis data as of {brandData.date}
                </div>
              </div>
              
              {/* Executive Summary */}
              <div className="bg-white shadow rounded-lg p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Executive Summary</h3>
                <p className="text-gray-700 mb-4">
                  This analysis compares how leading LLMs interpret and present {brandData.name}'s digital footprint. 
                  Our findings reveal discrepancies in perceived brand positioning, product catalog coverage, 
                  and digital data consistency that warrant attention.
                </p>
                
                <div className="mt-6">
                  <h4 className="text-md font-medium text-gray-900 mb-3">Key Findings:</h4>
                  <ul className="space-y-2 text-gray-700">
                    {brandData.keyFindings?.map((finding, index) => (
                      <li key={index} className="flex items-start">
                        <span className="flex-shrink-0 h-5 w-5 rounded-full bg-red-100 flex items-center justify-center mr-2 mt-0.5">
                          <AlertTriangle size={12} className="text-red-600" />
                        </span>
                        <span>{finding}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              
              {/* LLM Performance Overview */}
              <div className="bg-white shadow rounded-lg p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">LLM Representation Overview</h3>
                <div className="space-y-4">
                  {Object.entries(brandData.llmScores || {}).map(([llmId, score]) => (
                    <div key={llmId} className="flex items-center">
                      <div className="w-24 text-sm font-medium text-gray-900">{llmInfo[llmId]?.name || llmId}</div>
                      <div className="flex-1 ml-4">
                        <div className="w-full bg-gray-200 rounded-full h-4">
                          <div 
                            className={`${llmInfo[llmId]?.color || 'bg-gray-600'} h-4 rounded-full`} 
                            style={{ width: `${score}%` }}
                          ></div>
                        </div>
                      </div>
                      <div className="ml-4 w-16 text-right text-sm font-medium text-gray-900">
                        {score}%
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 text-sm text-gray-500">
                  <p>* Percentage represents overall brand accuracy and coverage across all measured dimensions</p>
                </div>
              </div>
            </div>
          )}
          
          {/* Placeholder for other views */}
          {activeView !== 'overview' && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center p-6 bg-white rounded-lg shadow-md">
                <h2 className="text-xl font-medium text-gray-900 mb-4">{activeView.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())} View</h2>
                <p className="text-gray-600">This view will be implemented in the full dashboard.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CompetitiveIntelligenceDashboard;
EOL

log "${BLUE}Setting up Docker configuration...${NC}"

# Create Dockerfile
cat > dashboard/docker/Dockerfile << 'EOL'
# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files and install dependencies
COPY dashboard/frontend/package*.json ./
RUN npm install

# Copy all frontend files
COPY dashboard/frontend/ ./

# Build the application
RUN npm run build

# Production stage with API
FROM python:3.12-slim

WORKDIR /app

# Copy Python requirements
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Install FastAPI and dependencies
RUN pip install fastapi uvicorn

# Copy API code
COPY dashboard/api/ /app/api/

# Copy built frontend assets
COPY --from=build /app/build /app/static

# Add Nginx for serving the frontend
RUN apt-get update && \
    apt-get install -y nginx curl && \
    rm -rf /var/lib/apt/lists/*

# Copy nginx configuration
COPY dashboard/docker/nginx.conf /etc/nginx/conf.d/default.conf

# Expose ports for Nginx and API
EXPOSE 80 8000

# Create startup script
RUN echo '#!/bin/bash\nservice nginx start\npython -m uvicorn api.main:app --host 0.0.0.0 --port 8000\n' > /app/start.sh && \
    chmod +x /app/start.sh

# Start Nginx and the API server
CMD ["/app/start.sh"]
EOL

# Create compose.yaml
cat > dashboard/docker/compose.yaml << 'EOL'
name: brandscope-dashboard

services:
  dashboard:
    build:
      context: ../  # Reference the root of the monorepo
      dockerfile: dashboard/docker/Dockerfile
    ports:
      - "8080:80"   # Frontend
      - "8000:8000" # API
    volumes:
      - type: bind
        source: ../results
        target: /app/data/results
        read_only: true
    environment:
      - RESULTS_DIR=/app/data/results
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
EOL

# Create nginx.conf
cat > dashboard/docker/nginx.conf << 'EOL'
server {
    listen 80;
    server_name localhost;
    root /app/static;
    index index.html;

    # Enable gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # Handle React routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Cache static assets
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, max-age=31536000";
    }
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options SAMEORIGIN;
    add_header X-XSS-Protection "1; mode=block";
}
EOL

# Make the script executable
chmod +x dashboard/docker/compose.yaml

log "${GREEN}Setup complete!${NC}"
log "All necessary files have been created in the dashboard directory structure:"
log ""
log "├── dashboard/"
log "│   ├── __init__.py"
log "│   ├── api/"
log "│   │   ├── __init__.py"
log "│   │   ├── data.py"
log "│   │   └── main.py"
log "│   ├── docker/"
log "│   │   ├── Dockerfile"
log "│   │   ├── compose.yaml"
log "│   │   └── nginx.conf"
log "│   └── frontend/"
log "│       ├── public/"
log "│       │   └── index.html"
log "│       ├── src/"
log "│       │   ├── components/"
log "│       │   │   └── CompetitiveIntelligenceDashboard.jsx"
log "│       │   ├── App.jsx"
log "│       │   └── index.jsx"
log "│       ├── package.json"
log "│       └── .dockerignore"
log ""
log "${BLUE}To build and run the dashboard:${NC}"
log "1. Navigate to the brandscope root directory"
log "2. Run: docker compose -f dashboard/docker/compose.yaml up -d"
log "3. Access the dashboard at http://localhost:8080"
log ""
log "${BLUE}For local development:${NC}"
log "1. cd dashboard/frontend"
log "2. npm install"
log "3. npm start"
log ""
log "${YELLOW}Note:${NC} The dashboard is preconfigured to read data from your results directory."
log "If your data format differs from the expected structure, you may need to modify"
log "the data.py file to adapt the data transformation logic."
log ""
log "${GREEN}Happy analyzing!${NC}"