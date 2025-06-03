"""
Competitive Context Detector Prompt
Priority: High

Compares a brand with its key competitors
"""
from typing import Dict, Any, List

from brandscope.prompts.base import BasePrompt

class CompetitiveContextPrompt(BasePrompt):
    """
    Implements the competitive context prompt
    """
    name = "competitive_context_detector"
    description = "Compares a brand with its key competitors"
    priority = 80  # High priority
    
    @staticmethod
    def get_prompt(params: Dict[str, Any]) -> str:
        """
        Generate the competitive comparison prompt
        
        Args:
            params: Dictionary containing 'brand', 'industry', and 'competitors' keys
            
        Returns:
            Formatted prompt string
        """
        brand = params.get("brand", "")
        industry = params.get("industry", "")
        competitors = params.get("competitors", [])
        
        # Handle competitors as either a list or comma-separated string
        if isinstance(competitors, str):
            competitor_str = competitors
        elif isinstance(competitors, list) and len(competitors) >= 3:
            competitor_str = f"{competitors[0]}, {competitors[1]}, and {competitors[2]}"
        else:
            competitor_str = ", ".join(competitors)
        
        return f"""
        Compare {brand} with its top competitors in the {industry} space. 
        Include direct competitors {competitor_str}.
        
        Structure your response with clear headings for:
        - Product offerings
        - Pricing
        - Unique selling propositions
        - Market positioning
        - Consumer perception
        
        Format as a comparative analysis.
        """
    
    @classmethod
    def get_promptfoo_template(cls) -> Dict[str, Any]:
        """
        Generate a promptfoo template configuration with competitors
        
        Returns:
            Dictionary with promptfoo configuration
        """
        return {
            "id": cls.name,
            "raw": f"Compare {{{{brand}}}} with its top competitors in the {{{{industry}}}} space. "
               f"Include direct competitors {{{{competitor1}}}}, {{{{competitor2}}}}, and {{{{competitor3}}}}. "
               f"Structure your response with clear headings for: product offerings, pricing, "
               f"unique selling propositions, market positioning, and consumer perception. "
               f"Format as a comparative analysis."
        }