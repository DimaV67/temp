"""Base class for all prompt templates"""
from typing import Dict, Any, List, Optional

class BasePrompt:
    """Base class for all BrandScope prompts"""
    name = "base_prompt"
    description = "Base prompt template"
    priority = 0  # Higher numbers indicate higher priority
    
    @staticmethod
    def get_prompt(params: Dict[str, Any]) -> str:
        """
        Generate a prompt with the given parameters
        
        Args:
            params: Dictionary of parameters to include in the prompt
            
        Returns:
            Formatted prompt string
        """
        raise NotImplementedError("Subclasses must implement get_prompt")
    
    @classmethod
    def get_promptfoo_template(cls) -> Dict[str, Any]:
        """
        Generate a promptfoo template configuration
        
        Returns:
            Dictionary with promptfoo configuration
        """
        return {
            "id": cls.name,
            "raw": cls.get_prompt({"brand": "{{brand}}"})
        }