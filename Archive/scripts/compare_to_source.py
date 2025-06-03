"""
Module for comparing claims against source data using semantic similarity and price accuracy checks.

This module provides functions to evaluate the accuracy of claims by comparing them to source texts
extracted from a JSON file. It uses a sentence transformer model for semantic similarity and includes
specific logic for price accuracy checks.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_source_texts(source_json: dict) -> list:
    """
    Extract relevant texts from source JSON for comparison.

    Args:
        source_json (dict): JSON object containing product data with fields like
            'products', 'product_name', 'description', and 'price'.

    Returns:
        list: List of non-empty strings extracted from product names, descriptions,
            and prices.
    """
    texts = []
    for product in source_json.get("products", []):
        texts.append(product.get("product_name", ""))
        texts.append(product.get("description", ""))
        for key, value in product.get("price", {}).items():
            texts.append(str(value))
    texts = [t for t in texts if t]
    return texts


def check_price_accuracy(claim: str, truth_price: str) -> bool:
    """
    Check if a price claim matches the source price within a small tolerance.

    Args:
        claim (str): The price claim to evaluate (e.g., "$29.99").
        truth_price (str): The source price to compare against (e.g., "$29.99").

    Returns:
        bool: True if the prices match within 0.01, False otherwise.
    """
    try:
        claim_price = float(claim.replace('$', '').strip())
        truth_price = float(truth_price.replace('$', '').strip())
        return abs(claim_price - truth_price) < 0.01
    except (ValueError, AttributeError):
        return False


def compare_claims_to_source(claims_with_providers: list, source_json: dict) -> list:
    """
    Compare claims to source texts and evaluate their accuracy.

    Uses semantic similarity for non-price claims and exact matching for price claims.

    Args:
        claims_with_providers (list): List of tuples, each containing a claim (str)
            and its provider (str).
        source_json (dict): Source JSON containing product data for comparison.

    Returns:
        list: List of dictionaries, each containing:
            - claim (str): The evaluated claim.
            - provider (str): The provider of the claim.
            - closest_source (str): The most similar source text.
            - similarity (float): Cosine similarity score.
            - is_accurate (bool): Whether the claim is deemed accurate.
    """
    discrepancies = []
    source_texts = extract_source_texts(source_json)
    if not source_texts:
        return [{"claim": claim, "provider": provider, "closest_source": "", "similarity": 0.0, "is_accurate": False}
                for claim, provider in claims_with_providers]

    source_embeddings = model.encode(source_texts, convert_to_numpy=True)

    for claim, provider in claims_with_providers:
        claim_embedding = model.encode([claim], convert_to_numpy=True)
        similarities = cosine_similarity(claim_embedding, source_embeddings)[0]
        max_similarity = similarities.max()
        closest_idx = similarities.argmax()
        closest_source = source_texts[closest_idx]

        if '$' in claim and 'price' in source_json.get('products', [{}])[0]:
            is_accurate = check_price_accuracy(claim, source_json['products'][0]['price'].get('one_time', ''))
        else:
            is_accurate = max_similarity >= 0.7

        discrepancies.append({
            "claim": claim,
            "provider": provider,
            "closest_source": closest_source,
            "similarity": float(max_similarity),  # Ensure float
            "is_accurate": bool(is_accurate)  # Ensure explicit bool
        })
    return discrepancies


def calculate_llm_accuracy(discrepancies: list) -> float:
    """
    Calculate the accuracy of claims based on their evaluated accuracy.

    Args:
        discrepancies (list): List of discrepancy dictionaries, each containing
            an 'is_accurate' field.

    Returns:
        float: The fraction of accurate claims, or 0.0 if no claims are provided.
    """
    if not discrepancies:
        return 0.0
    accurate_count = sum(1 for d in discrepancies if d["is_accurate"])
    return accurate_count / len(discrepancies)