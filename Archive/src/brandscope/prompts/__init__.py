"""Prompt templates for BrandScope"""
from brandscope.prompts.registry import PromptRegistry
from brandscope.prompts.base import BasePrompt
from brandscope.prompts.baseline import BrandBaselinePrompt
from brandscope.prompts.competitive import CompetitiveContextPrompt

# Make classes available at the module level
__all__ = [
    'PromptRegistry',
    'BasePrompt',
    'BrandBaselinePrompt',
    'CompetitiveContextPrompt',
]