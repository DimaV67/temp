"""
Registry of all prompt templates
"""
from typing import Dict, Type, List

from brandscope.prompts.base import BasePrompt
from brandscope.prompts.baseline import BrandBaselinePrompt
from brandscope.prompts.competitive import CompetitiveContextPrompt

class PromptRegistry:
    """
    Registry of all available prompts in the system
    """
    _prompts: Dict[str, Type[BasePrompt]] = {}
    
    @classmethod
    def register(cls, prompt_class: Type[BasePrompt]) -> None:
        """
        Register a prompt class
        
        Args:
            prompt_class: The prompt class to register
        """
        cls._prompts[prompt_class.name] = prompt_class
    
    @classmethod
    def get_prompt(cls, name: str) -> Type[BasePrompt]:
        """
        Get a prompt by name
        
        Args:
            name: The name of the prompt
            
        Returns:
            The prompt class
            
        Raises:
            KeyError: If the prompt is not registered
        """
        if name not in cls._prompts:
            raise KeyError(f"Prompt '{name}' not found in registry")
        return cls._prompts[name]
    
    @classmethod
    def get_all_prompts(cls) -> List[Type[BasePrompt]]:
        """
        Get all registered prompts
        
        Returns:
            List of all registered prompt classes
        """
        return list(cls._prompts.values())
    
    @classmethod
    def get_promptfoo_templates(cls) -> List[Dict]:
        """
        Get all prompts in promptfoo template format
        
        Returns:
            List of promptfoo template configurations
        """
        return [prompt.get_promptfoo_template() for prompt in cls.get_all_prompts()]


# Register all prompts
PromptRegistry.register(BrandBaselinePrompt)
PromptRegistry.register(CompetitiveContextPrompt)