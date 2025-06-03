"""
Metrics calculation for BrandScope
"""
from typing import Dict, Any, List, Union
import json
from pathlib import Path


class BrandMetrics:
    """
    Calculate and process metrics from promptfoo evaluation results
    """

    @staticmethod
    def calculate_accuracy(results: Dict[str, Any]) -> float:
        """
        Calculate accuracy score from evaluation results

        Args:
            results: promptfoo evaluation results

        Returns:
            Accuracy score between 0 and 1
        """
        # This is a placeholder - implement actual accuracy calculation
        tests = results.get("results", [])
        if not tests:
            return 0.0

        total_assertions = 0
        passed_assertions = 0

        for test in tests:
            assertions = test.get("assertions", [])
            total_assertions += len(assertions)
            passed_assertions += sum(1 for a in assertions if a.get("pass", False))

        return passed_assertions / total_assertions if total_assertions > 0 else 0.0

    @staticmethod
    def calculate_consistency(results: Dict[str, Any]) -> float:
        """
        Calculate consistency score across different providers

        Args:
            results: promptfoo evaluation results

        Returns:
            Consistency score between 0 and 1
        """
        # This is a placeholder - implement actual consistency calculation
        return 0.5

    @staticmethod
    def calculate_visibility_score(metrics: Dict[str, float]) -> float:
        """
        Calculate the BrandScope™ Visibility Score

        Args:
            metrics: Dictionary of individual metrics

        Returns:
            Composite visibility score between 0 and 100
        """
        weights = {
            "accuracy": 0.4,
            "completeness": 0.25,
            "competitor_presence": 0.15,
            "sentiment": 0.1,
            "consistency": 0.1
        }

        # Invert competitor presence (lower is better)
        competitor_score = 1.0 - metrics.get("competitor_presence", 0.0)

        # Normalize sentiment from [-10, 10] to [0, 1]
        sentiment = metrics.get("sentiment", 0.0)
        sentiment_normalized = (sentiment + 10) / 20 if -10 <= sentiment <= 10 else 0.5

        weighted_score = (
                weights["accuracy"] * metrics.get("accuracy", 0.0) +
                weights["completeness"] * metrics.get("completeness", 0.0) +
                weights["competitor_presence"] * competitor_score +
                weights["sentiment"] * sentiment_normalized +
                weights["consistency"] * metrics.get("consistency", 0.0)
        )

        # Convert to 0-100 scale
        return weighted_score * 100

    @staticmethod
    def process_results(results_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process promptfoo results into BrandScope metrics

        Args:
            results_path: Path to the promptfoo results JSON file

        Returns:
            Dictionary of calculated metrics
        """
        with open(results_path, 'r') as f:
            results = json.load(f)

        accuracy = BrandMetrics.calculate_accuracy(results)

        # Placeholder values for other metrics
        metrics = {
            "accuracy": accuracy,
            "completeness": 0.7,  # Placeholder
            "competitor_presence": 0.3,  # Placeholder
            "sentiment": 5.0,  # Placeholder, on scale of -10 to 10
            "consistency": BrandMetrics.calculate_consistency(results)
        }

        # Calculate visibility score
        visibility_score = BrandMetrics.calculate_visibility_score(metrics)

        return {
            **metrics,
            "visibility_score": visibility_score,
            "total_tests": len(results.get("results", [])),
            "raw_results": results
        }