"""
Brand Information Baseline Prompt
Priority: Highest

Provides factual overview of brands with structured data output
"""
from typing import Dict, Any

from brandscope.prompts.base import BasePrompt

class BrandBaselinePrompt(BasePrompt):
    """
    Implements the baseline prompt for brand information
    """
    name = "brand_information_baseline"
    description = "Retrieves factual baseline information about a brand"
    priority = 100  # Highest priority
    
    @staticmethod
    def get_prompt(params: Dict[str, Any]) -> str:
        """
        Generate the prompt for a given brand
        
        Args:
            params: Dictionary containing at least a 'brand' key
            
        Returns:
            Formatted prompt string
        """
        brand = params.get("brand", "")
        
        return f"""
        You are a brand information system providing factual data.
        
        Provide a factual overview of {brand} including:
        - Core products/services
        - Key features
        - Price points
        - Availability
        - Target market
        - Notable ethical considerations or commitments
        
        Format the response as structured data with clear category headings.
        """