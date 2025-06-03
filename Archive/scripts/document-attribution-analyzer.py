import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
import networkx as nx # Not explicitly used in the provided DocumentSummaryAttributionAnalyzer, but was in the initial imports. Will keep for now.
from itertools import combinations # Not explicitly used, will keep for now.
import pandas as pd
import matplotlib.pyplot as plt # Not explicitly used in analyzer, but for visualizations in main script.
import seaborn as sns # Not explicitly used in analyzer, but for visualizations in main script.
from scipy.stats import entropy
import textstat
import re
import warnings # Not explicitly used, will keep for now.
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import logging
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os # For directory creation in main script
import argparse # For CLI in main script
from sklearn.model_selection import KFold # For cross-validation
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix # For evaluation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentSummaryAttributionAnalyzer:
    """
    A comprehensive analyzer for determining the probability that a summary was generated from a specific document.
    Uses multiple NLP techniques and metrics to provide a robust analysis with supporting evidence.
    """

    def __init__(self, cache_models: bool = True, verbose: bool = True):
        """
        Initialize the analyzer with necessary NLP models and resources.

        Args:
            cache_models: Whether to keep models in memory (Note: models are loaded once per instance).
            verbose: Whether to print detailed information during analysis.
        """
        self.verbose = verbose
        self.cache_models = cache_models # This flag is noted but not actively used to change model loading behavior within this class structure.
                                         # Models are loaded upon instantiation.
        self._log("Initializing DocumentSummaryAttributionAnalyzer...")

        # Download required NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self._log("Downloading NLTK punkt...")
            nltk.download('punkt', quiet=not verbose)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            self._log("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=not verbose)

        # Initialize models
        self._log("Loading SpaCy model (en_core_web_md)...")
        try:
            self.nlp = spacy.load('en_core_web_md')
        except OSError:
            self._log("SpaCy model 'en_core_web_md' not found. Downloading...")
            spacy.cli.download('en_core_web_md')
            self.nlp = spacy.load('en_core_web_md')


        self._log("Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

        self._log("Loading NLI model (roberta-large-mnli) for entailment detection...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

        self.stopwords = set(stopwords.words('english'))
        self._log("Initialization complete.")

        # Thresholds for probability calculation (can be optimized)
        self.thresholds = {
            'lexical_overlap_threshold': 0.4,
            'semantic_similarity_threshold': 0.65,
            'entity_overlap_threshold': 0.3,
            'entailment_threshold': 0.7, # Used for entailment_ratio in _analyze_entailment
            'rare_phrase_threshold': 0.25, # Not directly used in _calculate_probability scaling, but could be
            'structural_similarity_threshold': 0.5, # Not directly used in _calculate_probability scaling, but could be
            # Thresholds for scaling in _calculate_probability
            'lexical_overlap_scaling_threshold': 0.2,
            'semantic_similarity_scaling_threshold': 0.5,
            'entity_match_scaling_threshold': 0.2,
            'entailment_score_scaling_threshold': 0.6,
            'rare_phrase_match_scaling_threshold': 0.1,
            'structural_similarity_scaling_threshold': 0.4
        }

        # Weights for different metrics in final probability calculation
        self.weights = {
            'lexical_overlap': 0.15,
            'semantic_similarity': 0.25,
            'entity_match': 0.15,
            'entailment_score': 0.20,
            'rare_phrase_match': 0.15,
            'structural_similarity': 0.10
        }

    def _log(self, message: str) -> None:
        """Log messages if verbose is enabled"""
        if self.verbose:
            logger.info(message)

    def analyze(self, document: str, summary: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Analyze the probability that the summary was derived from the document.

        Args:
            document: The source document text.
            summary: The summary text to analyze.
            threshold: Probability threshold for attribution decision.

        Returns:
            Dict containing analysis results, probability, and supporting evidence.
        """
        self._log(f"Starting analysis of document ({len(document)} chars) and summary ({len(summary)} chars)")

        # Pre-process texts
        doc_processed = self._preprocess_text(document)
        summary_processed = self._preprocess_text(summary)

        # Skip processing if either text is empty after preprocessing
        if not doc_processed or not summary_processed:
            self._log("Warning: Empty text after preprocessing")
            return {
                'probability': 0.0,
                'attribution': False,
                'confidence': 0.0,
                'explanation': "Empty text after preprocessing, or text contains only special characters.",
                'metrics': {},
                'evidence': []
            }

        # Analyze document and summary with various methods
        results = {}

        # 1. Lexical Analysis
        lexical_results = self._analyze_lexical_overlap(doc_processed, summary_processed)
        results['lexical_overlap'] = lexical_results

        # 2. Semantic Similarity
        semantic_results = self._analyze_semantic_similarity(document, summary) # Use original for sentence transformer
        results['semantic_similarity'] = semantic_results

        # 3. Entity Analysis
        entity_results = self._analyze_entity_overlap(document, summary) # Use original for SpaCy
        results['entity_match'] = entity_results

        # 4. Entailment Analysis
        entailment_results = self._analyze_entailment(document, summary) # Use original for NLI
        results['entailment_score'] = entailment_results

        # 5. Rare Phrase Detection
        rare_phrase_results = self._analyze_rare_phrases(document, summary) # Use original for n-gram extraction
        results['rare_phrase_match'] = rare_phrase_results

        # 6. Structural Analysis
        structural_results = self._analyze_structural_similarity(document, summary) # Use original for SpaCy & textstat
        results['structural_similarity'] = structural_results

        # Calculate final probability
        probability, confidence, evidence = self._calculate_probability(results)

        # Determine attribution based on probability and threshold
        attribution = probability >= threshold

        # Generate summary of findings
        explanation = self._generate_explanation(results, probability, attribution)

        return {
            'probability': round(probability, 4),
            'attribution': attribution,
            'confidence': round(confidence, 4),
            'explanation': explanation,
            'metrics': results,
            'evidence': evidence
        }

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing extra whitespace, etc.
        Keeps original case for models that might need it (e.g., SpaCy for NER).
        Removes special characters that might interfere with tokenizers or TF-IDF.

        Args:
            text: Text to preprocess.

        Returns:
            Preprocessed text.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return ""
        
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        # Remove characters that are not alphanumeric, common punctuation, or whitespace
        # This aims to reduce noise while preserving sentence structure.
        text = re.sub(r'[^\w\s.,;?!:\'"\-\(\)\[\]\{\}]', '', text)
        if not text.strip(): # Check if text became empty after removing special chars
            return ""
        return text


    def _analyze_lexical_overlap(self, document: str, summary: str) -> Dict[str, Any]:
        """
        Analyze lexical overlap between document and summary using TF-IDF.

        Args:
            document: Preprocessed document text.
            summary: Preprocessed summary text.

        Returns:
            Dict containing lexical overlap metrics.
        """
        self._log("Analyzing lexical overlap...")

        doc_sentences = sent_tokenize(document)
        summary_sentences = sent_tokenize(summary)

        if not doc_sentences or not summary_sentences:
            return {
                'score': 0.0,
                'unique_overlap_ratio': 0.0,
                'significant_terms': [],
                'matched_ratio': 0.0,
                'details': "Empty sentences after tokenization for document or summary."
            }

        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))

        try:
            tfidf_matrix = vectorizer.fit_transform(doc_sentences + summary_sentences)
            doc_vectors = tfidf_matrix[:len(doc_sentences)]
            summary_vectors = tfidf_matrix[len(doc_sentences):]

            if summary_vectors.shape[0] == 0 or doc_vectors.shape[0] == 0:
                 return {
                    'score': 0.0, 'unique_overlap_ratio': 0.0, 'significant_terms': [], 'matched_ratio': 0.0,
                    'details': "TF-IDF resulted in zero vectors for summary or document."
                }

            similarities = cosine_similarity(summary_vectors, doc_vectors)
            max_similarities = np.max(similarities, axis=1) if similarities.size > 0 else np.array([0.0])
            avg_similarity = float(np.mean(max_similarities))

            doc_words = set(w.lower() for w in re.findall(r'\b\w+\b', document)
                           if w.lower() not in self.stopwords and len(w) > 2)
            summary_words = set(w.lower() for w in re.findall(r'\b\w+\b', summary)
                               if w.lower() not in self.stopwords and len(w) > 2)

            unique_overlap_ratio = 0.0
            if summary_words:
                overlap_words = doc_words.intersection(summary_words)
                unique_overlap_ratio = len(overlap_words) / len(summary_words)

            significant_terms = []
            if overlap_words and hasattr(vectorizer, 'get_feature_names_out'):
                feature_names = vectorizer.get_feature_names_out()
                summary_tfidf_sums = np.sum(summary_vectors.toarray(), axis=0)
                term_importance = {feature_names[i]: summary_tfidf_sums[i]
                                   for i in range(len(feature_names))
                                   if feature_names[i] in overlap_words}
                sorted_terms = sorted(term_importance.items(), key=lambda x: x[1], reverse=True)
                significant_terms = [term for term, score in sorted_terms[:10]]

            return {
                'score': avg_similarity,
                'unique_overlap_ratio': unique_overlap_ratio,
                'significant_terms': significant_terms,
                'matched_ratio': len([s for s in max_similarities if s > 0.5]) / len(max_similarities) if max_similarities.size > 0 else 0.0
            }

        except Exception as e:
            self._log(f"Error in lexical analysis: {str(e)}")
            return {'score': 0.0, 'unique_overlap_ratio': 0.0, 'significant_terms': [], 'matched_ratio': 0.0, 'error': str(e)}

    def _analyze_semantic_similarity(self, document: str, summary: str) -> Dict[str, Any]:
        """
        Analyze semantic similarity using sentence embeddings.

        Args:
            document: Original document text.
            summary: Original summary text.

        Returns:
            Dict containing semantic similarity metrics.
        """
        self._log("Analyzing semantic similarity...")
        try:
            doc_sentences = sent_tokenize(document)
            summary_sentences = sent_tokenize(summary)

            if not doc_sentences or not summary_sentences:
                return {'score': 0.0, 'document_similarity': 0.0, 'best_matches': [], 'matched_ratio': 0.0, 'details': "Empty sentences."}

            doc_embeddings = self.sentence_transformer.encode(doc_sentences)
            summary_embeddings = self.sentence_transformer.encode(summary_sentences)

            sentence_similarities = cosine_similarity(summary_embeddings, doc_embeddings)
            max_similarities = np.max(sentence_similarities, axis=1) if sentence_similarities.size > 0 else np.array([0.0])
            avg_similarity = float(np.mean(max_similarities))

            best_matches = []
            for i, summary_sent in enumerate(summary_sentences[:min(5, len(summary_sentences))]):
                if sentence_similarities.shape[1] > 0: # Ensure there are document sentences to compare against
                    best_doc_idx = np.argmax(sentence_similarities[i])
                    similarity = sentence_similarities[i][best_doc_idx]
                    if similarity > 0.7:
                        best_matches.append({
                            'summary_sentence': summary_sent,
                            'document_sentence': doc_sentences[best_doc_idx],
                            'similarity': float(similarity)
                        })
            
            # Overall document similarity (handle empty document/summary for encoding)
            doc_embedding_input = [document] if document.strip() else [""]
            summary_embedding_input = [summary] if summary.strip() else [""]

            doc_embedding = self.sentence_transformer.encode(doc_embedding_input)[0]
            summary_embedding = self.sentence_transformer.encode(summary_embedding_input)[0]
            
            doc_similarity = 0.0
            if np.any(doc_embedding) and np.any(summary_embedding): # Check for non-zero embeddings
                 doc_similarity = float(cosine_similarity([doc_embedding], [summary_embedding])[0][0])


            return {
                'score': avg_similarity,
                'document_similarity': doc_similarity,
                'best_matches': best_matches,
                'matched_ratio': len([s for s in max_similarities if s > 0.7]) / len(max_similarities) if max_similarities.size > 0 else 0.0
            }
        except Exception as e:
            self._log(f"Error in semantic similarity analysis: {str(e)}")
            return {'score': 0.0, 'document_similarity': 0.0, 'best_matches': [], 'matched_ratio': 0.0, 'error': str(e)}

    def _analyze_entity_overlap(self, document: str, summary: str) -> Dict[str, Any]:
        """
        Analyze named entity overlap between document and summary.

        Args:
            document: Original document text.
            summary: Original summary text.

        Returns:
            Dict containing entity overlap metrics.
        """
        self._log("Analyzing entity overlap...")
        try:
            doc_spacy = self.nlp(document)
            summary_spacy = self.nlp(summary)

            doc_entities = {(ent.text.lower(), ent.label_) for ent in doc_spacy.ents}
            summary_entities = {(ent.text.lower(), ent.label_) for ent in summary_spacy.ents}

            if not summary_entities:
                return {'score': 0.0, 'overlapping_entities': [], 'type_overlap': {}, 'summary_entity_count': 0, 'overlapping_count': 0, 'details': "No entities found in summary"}

            overlapping_entities = doc_entities.intersection(summary_entities)
            entity_overlap_ratio = len(overlapping_entities) / len(summary_entities) if summary_entities else 0.0

            entity_types_summary = Counter(label for _, label in summary_entities)
            entity_types_overlap = Counter(label for _, label in overlapping_entities)
            
            type_overlap = {}
            for label, count in entity_types_summary.items():
                type_overlap[label] = entity_types_overlap.get(label, 0) / count if count > 0 else 0.0
            
            return {
                'score': entity_overlap_ratio,
                'overlapping_entities': [{'text': e, 'type': t} for e, t in overlapping_entities],
                'type_overlap': type_overlap,
                'summary_entity_count': len(summary_entities),
                'overlapping_count': len(overlapping_entities)
            }
        except Exception as e:
            self._log(f"Error in entity analysis: {str(e)}")
            return {'score': 0.0, 'overlapping_entities': [], 'type_overlap': {}, 'summary_entity_count': 0, 'overlapping_count': 0, 'error': str(e)}

    def _analyze_entailment(self, document: str, summary: str) -> Dict[str, Any]:
        """
        Analyze logical entailment between document and summary sentences.

        Args:
            document: Original document text.
            summary: Original summary text.

        Returns:
            Dict containing entailment metrics.
        """
        self._log("Analyzing logical entailment...")
        try:
            summary_sentences = sent_tokenize(summary)
            if not summary_sentences:
                return {'score': 0.0, 'entailment_ratio': 0.0, 'sentence_analysis': [], 'details': "No sentences in summary"}

            sample_size = min(10, len(summary_sentences)) # Performance consideration
            selected_sentences = summary_sentences[:sample_size]
            
            entailment_scores = []
            sentence_results = []

            # Use the whole document as premise for better context, but be mindful of NLI model's max sequence length.
            # The tokenizer will handle truncation.
            premise = document 

            for hypothesis in selected_sentences:
                if not hypothesis.strip(): continue # Skip empty sentences

                encoded_pair = self.nli_tokenizer(
                    premise,
                    hypothesis,
                    padding=True,
                    truncation=True, # Crucial for long documents
                    max_length=512, # Max length for roberta-large-mnli
                    return_tensors="pt"
                )
                with torch.no_grad():
                    output = self.nli_model(**encoded_pair)
                
                probs = torch.nn.functional.softmax(output.logits, dim=1)
                # MNLI labels: 0 = contradiction, 1 = neutral, 2 = entailment
                entailment_prob = float(probs[0][self.nli_model.config.label2id['entailment']].item())
                contradiction_prob = float(probs[0][self.nli_model.config.label2id['contradiction']].item())
                
                entailment_scores.append(entailment_prob)
                sentence_results.append({
                    'summary_sentence': hypothesis,
                    'entailment_probability': entailment_prob,
                    'contradiction_probability': contradiction_prob
                })

            avg_entailment = float(np.mean(entailment_scores)) if entailment_scores else 0.0
            strongly_entailed_count = len([s for s in entailment_scores if s > self.thresholds.get('entailment_threshold', 0.7)])
            entailment_ratio = strongly_entailed_count / len(entailment_scores) if entailment_scores else 0.0

            return {
                'score': avg_entailment,
                'entailment_ratio': entailment_ratio,
                'sentence_analysis': sentence_results
            }
        except Exception as e:
            self._log(f"Error in entailment analysis: {str(e)}")
            return {'score': 0.0, 'entailment_ratio': 0.0, 'sentence_analysis': [], 'error': str(e)}

    def _analyze_rare_phrases(self, document: str, summary: str) -> Dict[str, Any]:
        """
        Detect distinctive or rare phrases from the document that appear in the summary.

        Args:
            document: Original document text.
            summary: Original summary text.

        Returns:
            Dict containing rare phrase metrics.
        """
        self._log("Analyzing rare phrases...")
        try:
            def extract_ngrams(text, n_val):
                words = [w.lower() for w in re.findall(r'\b\w+\b', text) # Simple word tokenization
                         if w.lower() not in self.stopwords and len(w) > 2]
                return [' '.join(words[i:i + n_val]) for i in range(len(words) - n_val + 1)]

            doc_ngrams_all = {}
            summary_ngrams_sets = {}
            total_summary_ngrams_count = 0

            for n in range(2, 5): # 2, 3, 4-grams
                doc_ngrams_all[n] = Counter(extract_ngrams(document, n))
                current_summary_ngrams = extract_ngrams(summary, n)
                summary_ngrams_sets[n] = set(current_summary_ngrams)
                total_summary_ngrams_count += len(current_summary_ngrams)


            if total_summary_ngrams_count == 0:
                return {'score': 0.0, 'matching_phrases': [], 'match_count': 0, 'details': "No valid ngrams found in summary."}

            matching_phrases_details = []
            weighted_match_score_numerator = 0

            # Prioritize longer n-grams as more distinctive
            for n in range(4, 1, -1): # Check 4-grams, then 3-grams, then 2-grams
                if n in summary_ngrams_sets:
                    for phrase in summary_ngrams_sets[n]:
                        if phrase in doc_ngrams_all[n] and doc_ngrams_all[n][phrase] > 0: # Phrase exists in document
                            # Consider rarity: if a phrase is very common in the doc, it's less distinctive
                            # This is a simple heuristic; true rarity requires a background corpus.
                            rarity_factor = 1 / (1 + np.log1p(doc_ngrams_all[n][phrase])) # Penalize common phrases in doc
                            
                            matching_phrases_details.append({'phrase': phrase, 'length': n, 'doc_freq': doc_ngrams_all[n][phrase]})
                            weighted_match_score_numerator += (n * 0.5) * rarity_factor # Weight by n-gram length and rarity factor

            # Normalize the score by the total number of n-grams in the summary (weighted by length)
            # This gives a sense of density of matched rare phrases.
            # Max possible score if all summary ngrams are matched perfectly (length * 0.5)
            # For normalization, consider a max possible weighted score for summary n-grams
            max_possible_summary_weighted_score = 0
            for n_val_s in range(2,5):
                max_possible_summary_weighted_score += len(summary_ngrams_sets[n_val_s]) * (n_val_s * 0.5)

            if max_possible_summary_weighted_score == 0: # Avoid division by zero if no summary ngrams
                 rare_phrase_score = 0.0
            else:
                 rare_phrase_score = weighted_match_score_numerator / max_possible_summary_weighted_score
            
            # Sort matched phrases by length (desc) then by phrase for consistent output
            top_matches = sorted(matching_phrases_details, key=lambda x: (x['length'], x['phrase']), reverse=True)[:10]

            return {
                'score': min(rare_phrase_score, 1.0), # Cap score at 1.0
                'matching_phrases': [item['phrase'] for item in top_matches],
                'match_count': len(matching_phrases_details)
            }
        except Exception as e:
            self._log(f"Error in rare phrase analysis: {str(e)}")
            return {'score': 0.0, 'matching_phrases': [], 'match_count': 0, 'error': str(e)}

    def _analyze_structural_similarity(self, document: str, summary: str) -> Dict[str, Any]:
        """
        Analyze structural similarities between document and summary using stylistic features.

        Args:
            document: Original document text.
            summary: Original summary text.

        Returns:
            Dict containing structural similarity metrics.
        """
        self._log("Analyzing structural similarity...")
        try:
            # Use SpaCy for more robust sentence tokenization and POS tagging
            doc_spacy = self.nlp(document)
            summary_spacy = self.nlp(summary)

            def extract_features(spacy_doc, text_str): # Pass original text_str for textstat
                if not list(spacy_doc.sents) or not text_str.strip():
                    return {}

                sent_lengths = [len(sent) for sent in spacy_doc.sents] # Number of tokens per sentence
                
                content_words = len([token for token in spacy_doc if not token.is_stop and token.is_alpha])
                total_words = len([token for token in spacy_doc if token.is_alpha])
                lexical_density = content_words / total_words if total_words > 0 else 0.0
                
                readability = 0.0
                try: # textstat can fail on very short or unusual texts
                    readability = textstat.flesch_reading_ease(text_str)
                except:
                    pass # Keep readability as 0.0

                pos_counts = Counter(token.pos_ for token in spacy_doc if token.pos_)
                total_tokens_for_pos = sum(pos_counts.values())
                pos_distrib = {pos: count / total_tokens_for_pos for pos, count in pos_counts.items()} if total_tokens_for_pos > 0 else {}

                # Sentence complexity (average number of verbs per sentence as a proxy for clauses)
                verb_counts_per_sentence = []
                for sent in spacy_doc.sents:
                    verb_counts_per_sentence.append(len([token for token in sent if token.pos_ == "VERB"]))
                avg_complexity = np.mean(verb_counts_per_sentence) if verb_counts_per_sentence else 0.0

                return {
                    'avg_sent_length': np.mean(sent_lengths) if sent_lengths else 0.0,
                    'lexical_density': lexical_density,
                    'readability': readability, # Range: 0-100 (higher is easier)
                    'pos_distribution': pos_distrib,
                    'avg_complexity': avg_complexity
                }

            doc_features = extract_features(doc_spacy, document)
            summary_features = extract_features(summary_spacy, summary)

            if not doc_features or not summary_features:
                return {'score': 0.0, 'feature_similarities': {}, 'details': "Could not extract features for document or summary."}

            feat_similarity = {}
            # Sentence Length: Ratio of min to max
            if doc_features.get('avg_sent_length', 0) > 0 and summary_features.get('avg_sent_length', 0) > 0:
                feat_similarity['sent_length_ratio'] = min(doc_features['avg_sent_length'], summary_features['avg_sent_length']) / \
                                                      max(doc_features['avg_sent_length'], summary_features['avg_sent_length'])
            else:
                feat_similarity['sent_length_ratio'] = 0.0

            # Lexical Density: Ratio of min to max
            if doc_features.get('lexical_density', 0) > 0 and summary_features.get('lexical_density', 0) > 0:
                 feat_similarity['lexical_density_ratio'] = min(doc_features['lexical_density'], summary_features['lexical_density']) / \
                                                       max(doc_features['lexical_density'], summary_features['lexical_density'])
            else: # Handle cases where one might be zero if text is all stopwords
                 feat_similarity['lexical_density_ratio'] = 0.0 if max(doc_features.get('lexical_density',0), summary_features.get('lexical_density',0)) > 0 else 1.0


            # Readability: Normalized difference (0-1 scale, 1 is more similar)
            # Flesch Reading Ease: higher is easier. Max diff is 100.
            feat_similarity['readability_similarity'] = 1 - abs(doc_features.get('readability', 50) - summary_features.get('readability', 50)) / 100.0

            # POS Distribution: Cosine similarity of POS distribution vectors
            all_pos_tags = set(doc_features.get('pos_distribution', {}).keys()) | set(summary_features.get('pos_distribution', {}).keys())
            if all_pos_tags:
                doc_pos_vec = np.array([doc_features.get('pos_distribution', {}).get(tag, 0) for tag in all_pos_tags])
                summary_pos_vec = np.array([summary_features.get('pos_distribution', {}).get(tag, 0) for tag in all_pos_tags])
                if np.any(doc_pos_vec) and np.any(summary_pos_vec): # Ensure not all zeros
                     feat_similarity['pos_similarity'] = cosine_similarity([doc_pos_vec], [summary_pos_vec])[0][0]
                elif not np.any(doc_pos_vec) and not np.any(summary_pos_vec): # Both empty
                    feat_similarity['pos_similarity'] = 1.0 
                else: # One is empty
                    feat_similarity['pos_similarity'] = 0.0
            else:
                feat_similarity['pos_similarity'] = 1.0 # Both had no POS tags

            # Sentence Complexity: Ratio of min to max
            if doc_features.get('avg_complexity', 0) > 0 and summary_features.get('avg_complexity', 0) > 0:
                feat_similarity['complexity_ratio'] = min(doc_features['avg_complexity'], summary_features['avg_complexity']) / \
                                                    max(doc_features['avg_complexity'], summary_features['avg_complexity'])
            else:
                feat_similarity['complexity_ratio'] = 0.0


            structural_score = float(np.mean(list(feat_similarity.values()))) if feat_similarity else 0.0
            
            return {
                'score': structural_score,
                'feature_similarities': feat_similarity,
                'doc_features': doc_features, # For inspection
                'summary_features': summary_features # For inspection
            }
        except Exception as e:
            self._log(f"Error in structural analysis: {str(e)}")
            return {'score': 0.0, 'feature_similarities': {}, 'error': str(e)}

    def _calculate_probability(self, results: Dict[str, Dict[str, Any]]) -> Tuple[float, float, List[Dict[str, Any]]]:
        """
        Calculate the final probability and confidence based on all metrics.

        Args:
            results: Dict containing results from all analysis methods.

        Returns:
            Tuple of (probability, confidence, evidence list).
        """
        self._log("Calculating final probability...")
        evidence = []
        weighted_scores_sum = 0.0
        total_weights_applied = 0.0
        adjusted_scores_for_confidence = []

        for method_key, weight_val in self.weights.items():
            raw_score = 0.0
            details_for_evidence = {}
            
            if method_key in results and isinstance(results[method_key], dict) and 'score' in results[method_key]:
                raw_score = results[method_key].get('score', 0.0)
                # Populate details for evidence from the specific method's results
                if method_key == 'lexical_overlap':
                    details_for_evidence['significant_terms'] = results[method_key].get('significant_terms', [])
                    details_for_evidence['unique_overlap_ratio'] = results[method_key].get('unique_overlap_ratio', 0.0)
                elif method_key == 'semantic_similarity':
                    details_for_evidence['best_matches'] = results[method_key].get('best_matches', [])
                    details_for_evidence['document_similarity'] = results[method_key].get('document_similarity', 0.0)
                elif method_key == 'entity_match':
                    details_for_evidence['overlapping_entities'] = results[method_key].get('overlapping_entities', [])
                    details_for_evidence['type_overlap'] = results[method_key].get('type_overlap', {})
                elif method_key == 'entailment_score':
                    details_for_evidence['entailment_ratio'] = results[method_key].get('entailment_ratio', 0.0)
                    strong_results = [s for s in results[method_key].get('sentence_analysis', [])
                                      if s.get('entailment_probability', 0) > 0.7 or s.get('contradiction_probability', 0) > 0.7]
                    details_for_evidence['strong_sentence_results_sample'] = strong_results[:3]
                elif method_key == 'rare_phrase_match':
                    details_for_evidence['matching_phrases'] = results[method_key].get('matching_phrases', [])
                    details_for_evidence['match_count'] = results[method_key].get('match_count', 0)
                elif method_key == 'structural_similarity':
                    details_for_evidence['feature_similarities'] = results[method_key].get('feature_similarities', {})
            else: # Method result might be missing or malformed
                 self._log(f"Warning: Results for method '{method_key}' are missing or malformed. Assigning raw_score=0.")


            # Threshold-based scaling for adjusted score
            # Use specific scaling thresholds for each method
            scaling_threshold = self.thresholds.get(f'{method_key}_scaling_threshold', 0.5) # Default scaling threshold
            
            adjusted_score = 0.0
            if scaling_threshold == 0 and raw_score > 0: # Avoid division by zero if threshold is 0 but score is positive
                adjusted_score = 1.0 
            elif scaling_threshold > 0 : # Proceed with scaling if threshold is positive
                if raw_score >= scaling_threshold:
                    # Scale scores above threshold to [0.5, 1.0]
                    # Ensure 1 - scaling_threshold is not zero
                    denominator = (1.0 - scaling_threshold)
                    if denominator > 1e-6: # Avoid division by very small number or zero
                         adjusted_score = 0.5 + 0.5 * (raw_score - scaling_threshold) / denominator
                    else: # If threshold is 1.0 and raw_score is 1.0
                         adjusted_score = 1.0 if raw_score >= scaling_threshold else 0.5 * raw_score / scaling_threshold
                else:
                    # Scale scores below threshold to [0, 0.5)
                    adjusted_score = 0.5 * raw_score / scaling_threshold
            # Ensure adjusted_score is capped between 0 and 1
            adjusted_score = max(0.0, min(1.0, adjusted_score))


            weighted_scores_sum += adjusted_score * weight_val
            total_weights_applied += weight_val
            adjusted_scores_for_confidence.append(adjusted_score)

            evidence.append({
                'method': method_key,
                'raw_score': round(raw_score, 4),
                'adjusted_score': round(adjusted_score, 4),
                'weight': weight_val,
                'relevance': "strong" if raw_score >= scaling_threshold else ("moderate" if raw_score > scaling_threshold * 0.5 else "weak"),
                'details': details_for_evidence
            })
        
        final_probability = 0.0
        if total_weights_applied > 1e-6: # Avoid division by zero if all weights are zero
            final_probability = weighted_scores_sum / total_weights_applied
        final_probability = max(0.0, min(1.0, final_probability)) # Ensure probability is in [0,1]

        # Confidence: 1 - standard deviation of adjusted scores (normalized)
        # High std dev means methods disagree, so lower confidence.
        confidence = 0.0
        if len(adjusted_scores_for_confidence) > 1:
            std_dev = np.std(adjusted_scores_for_confidence)
            # Normalize std_dev: max possible std_dev for scores in [0,1] is 0.5 (e.g. [0,0,1,1])
            # We want confidence to be high when std_dev is low.
            confidence = max(0.0, 1.0 - (std_dev / 0.5)) # Max std dev is 0.5 for scores in [0,1]
        elif len(adjusted_scores_for_confidence) == 1:
            confidence = 0.75 # Only one method, give moderate confidence.

        return final_probability, confidence, evidence

    def _generate_explanation(self, results: Dict[str, Any], probability: float, attribution: bool) -> str:
        """
        Generate a human-readable explanation of the analysis results.
        """
        explanation = f"Overall Attribution Probability: {probability:.2%} -> {'DERIVED' if attribution else 'NOT DERIVED'}.\n"
        explanation += "Contributing factors:\n"

        sorted_evidence = sorted(self.weights.keys(), key=lambda k: self.weights[k], reverse=True)

        for method in sorted_evidence:
            if method in results and isinstance(results[method], dict) and 'score' in results[method]:
                score = results[method]['score']
                threshold = self.thresholds.get(f'{method}_scaling_threshold', 0.5) # Use scaling threshold for relevance
                relevance = "strong" if score >= threshold else ("moderate" if score > threshold * 0.5 else "weak")
                explanation += f"  - {method.replace('_', ' ').title()}: Score={score:.3f} (Relevance: {relevance}).\n"
                
                # Add specific details for key methods
                if method == 'lexical_overlap' and 'significant_terms' in results[method] and results[method]['significant_terms']:
                    explanation += f"    Significant overlapping terms: {', '.join(results[method]['significant_terms'][:3])}...\n"
                elif method == 'semantic_similarity' and 'best_matches' in results[method] and results[method]['best_matches']:
                    explanation += f"    Found {len(results[method]['best_matches'])} semantically similar sentence pairs.\n"
                elif method == 'entity_match' and 'overlapping_entities' in results[method] and results[method]['overlapping_entities']:
                    explanation += f"    Found {len(results[method]['overlapping_entities'])} overlapping named entities.\n"
                elif method == 'entailment_score' and 'entailment_ratio' in results[method]:
                    explanation += f"    Entailment ratio for summary sentences: {results[method]['entailment_ratio']:.2%}.\n"
        
        if not any(results.values()): # If all results were empty
            explanation = "Analysis could not be performed due to empty or invalid input."

        return explanation.strip()

    # --- Methods for evaluation and optimization (from the original script structure) ---
    def calculate_confidence_intervals(self, document: str, summary: str, n_bootstrap: int = 100, sample_size_ratio: float = 0.8) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for the attribution probability.
        This involves re-analyzing subsets of the document and summary.
        Note: This is a simplified bootstrap focusing on sentence-level resampling if applicable,
        or just re-running analyze if direct resampling isn't straightforward for all metrics.
        For a more robust CI, resampling should ideally happen at the level of input units for each metric.
        Here, we'll simulate by perturbing the input slightly or just re-running.
        A more direct way for this complex pipeline is to bootstrap the *final probability score* if we had multiple (doc, summary) pairs.
        Given a single pair, we can bootstrap by resampling sentences.
        """
        self._log(f"Calculating confidence intervals with {n_bootstrap} bootstrap samples...")
        probabilities = []

        doc_sents = sent_tokenize(document)
        sum_sents = sent_tokenize(summary)

        if not doc_sents or not sum_sents:
            self._log("Cannot perform bootstrap: document or summary has no sentences.")
            return {'mean_probability': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'std_dev': 0.0}

        for _ in range(n_bootstrap):
            # Resample sentences with replacement
            # Ensure at least one sentence if original has sentences
            resampled_doc_sents = np.random.choice(doc_sents, size=max(1, int(len(doc_sents) * sample_size_ratio)), replace=True)
            resampled_sum_sents = np.random.choice(sum_sents, size=max(1, int(len(sum_sents) * sample_size_ratio)), replace=True)
            
            resampled_doc = " ".join(resampled_doc_sents)
            resampled_sum = " ".join(resampled_sum_sents)

            if not resampled_doc.strip() or not resampled_sum.strip():
                continue # Skip if resampling leads to empty text

            # Analyze the resampled texts (turn off verbosity for bootstrap runs)
            original_verbose = self.verbose
            self.verbose = False 
            try:
                result = self.analyze(resampled_doc, resampled_sum)
                probabilities.append(result['probability'])
            except Exception as e:
                self._log(f"Error during bootstrap sample analysis: {e}")
            finally:
                self.verbose = original_verbose # Restore verbosity

        if not probabilities: # If all bootstrap samples failed
            self._log("All bootstrap samples failed analysis.")
            # Fallback to a single analysis if bootstrapping fails completely
            main_analysis = self.analyze(document, summary)
            prob = main_analysis['probability']
            return {'mean_probability': prob, 'ci_lower': prob, 'ci_upper': prob, 'std_dev': 0.0}


        mean_prob = float(np.mean(probabilities))
        std_dev = float(np.std(probabilities))
        # Calculate 95% confidence interval
        ci_lower = float(np.percentile(probabilities, 2.5))
        ci_upper = float(np.percentile(probabilities, 97.5))

        return {
            'mean_probability': round(mean_prob, 4),
            'ci_lower': round(ci_lower, 4),
            'ci_upper': round(ci_upper, 4),
            'std_dev': round(std_dev, 4)
        }

    def test_adversarial_robustness(self, document: str, summary: str, n_perturbations: int = 5) -> Dict[str, Any]:
        """
        Test robustness against simple adversarial perturbations (e.g., synonym replacement, sentence reordering).
        This is a basic test; more sophisticated adversarial attacks exist.
        """
        self._log("Testing adversarial robustness...")
        results = {'original_probability': self.analyze(document, summary)['probability']}
        perturbed_summaries_results = []

        summary_sents = sent_tokenize(summary)
        if len(summary_sents) < 2: # Need at least 2 sentences for reordering
            self._log("Summary too short for some perturbations.")
            # Could add synonym replacement here even for short summaries
            results['perturbations'] = []
            results['mean_perturbed_probability'] = results['original_probability']
            results['robustness_score'] = 1.0 # No change if no perturbation
            return results


        # Perturbation 1: Sentence Reordering (if summary has multiple sentences)
        if len(summary_sents) > 1:
            for _ in range(n_perturbations // 2 if n_perturbations > 1 else 1):
                permuted_sents = np.random.permutation(summary_sents)
                perturbed_summary = " ".join(permuted_sents)
                prob = self.analyze(document, perturbed_summary)['probability']
                perturbed_summaries_results.append({'type': 'reorder', 'probability': prob})
        
        # Perturbation 2: Simple Paraphrase (e.g., adding neutral phrases - very basic)
        # A more robust paraphrase would use a paraphrase generation model.
        # For now, let's simulate by removing a non-critical sentence if summary is long enough
        if len(summary_sents) > 2: # Need at least 3 sentences to remove one and keep some content
             for _ in range(n_perturbations - (n_perturbations // 2 if n_perturbations > 1 else 1) ):
                idx_to_remove = np.random.randint(0, len(summary_sents))
                perturbed_sents = [s for i, s in enumerate(summary_sents) if i != idx_to_remove]
                if perturbed_sents: # Ensure not empty
                    perturbed_summary = " ".join(perturbed_sents)
                    prob = self.analyze(document, perturbed_summary)['probability']
                    perturbed_summaries_results.append({'type': 'remove_sentence', 'probability': prob})


        if perturbed_summaries_results:
            mean_perturbed_prob = np.mean([r['probability'] for r in perturbed_summaries_results])
            # Robustness score: 1 - (absolute change / original probability)
            # Closer to 1 means more robust.
            if results['original_probability'] > 1e-6 :
                 robustness_score = 1.0 - (abs(results['original_probability'] - mean_perturbed_prob) / results['original_probability'])
            else: # if original prob is 0, any change in perturbed makes it non-robust unless perturbed is also 0
                 robustness_score = 1.0 if abs(mean_perturbed_prob) < 1e-6 else 0.0
            results['robustness_score'] = max(0.0, robustness_score) # Ensure score is not negative
            results['mean_perturbed_probability'] = mean_perturbed_prob
        else:
            results['robustness_score'] = 1.0 # No effective perturbations applied
            results['mean_perturbed_probability'] = results['original_probability']
            
        results['perturbations'] = perturbed_summaries_results
        return results

    def generate_report(self, document: str, summary: str, analysis_results: Dict[str, Any], report_path: str) -> None:
        """
        Generate a markdown report of the analysis.
        """
        self._log(f"Generating report at {report_path}...")
        report_content = f"# Document-Summary Attribution Analysis Report\n\n"
        report_content += f"**Document Snippet (first 500 chars):**\n```\n{document[:500]}...\n```\n\n"
        report_content += f"**Summary:**\n```\n{summary}\n```\n\n"
        report_content += f"## Overall Assessment\n"
        report_content += f"- **Attribution Probability:** {analysis_results['probability']:.4f}\n"
        report_content += f"- **Attribution Decision:** {'DERIVED FROM DOCUMENT' if analysis_results['attribution'] else 'NOT DERIVED FROM DOCUMENT'}\n"
        report_content += f"- **Confidence in Assessment:** {analysis_results['confidence']:.4f}\n\n"
        report_content += f"**Explanation:**\n{analysis_results['explanation']}\n\n"

        if 'confidence_intervals' in analysis_results:
            ci = analysis_results['confidence_intervals']
            report_content += f"## Bootstrap Confidence Interval (95%)\n"
            report_content += f"- Mean Probability: {ci.get('mean_probability', 'N/A'):.4f}\n"
            report_content += f"- CI: [{ci.get('ci_lower', 'N/A'):.4f}, {ci.get('ci_upper', 'N/A'):.4f}]\n"
            report_content += f"- Standard Deviation: {ci.get('std_dev', 'N/A'):.4f}\n\n"

        if 'robustness' in analysis_results:
            rb = analysis_results['robustness']
            report_content += f"## Adversarial Robustness\n"
            report_content += f"- Original Probability: {rb.get('original_probability', 'N/A'):.4f}\n"
            report_content += f"- Mean Perturbed Probability: {rb.get('mean_perturbed_probability', 'N/A'):.4f}\n"
            report_content += f"- Robustness Score (0-1, 1 is best): {rb.get('robustness_score', 'N/A'):.4f}\n"
            if rb.get('perturbations'):
                report_content += "Details of perturbations:\n"
                for p_idx, p_detail in enumerate(rb['perturbations']):
                    report_content += f"  - Perturbation {p_idx+1} (Type: {p_detail.get('type', 'unknown')}): Probability = {p_detail.get('probability', 'N/A'):.4f}\n"
            report_content += "\n"


        report_content += "## Detailed Metric Scores & Evidence\n"
        if 'evidence' in analysis_results:
            for ev in analysis_results['evidence']:
                report_content += f"### {ev['method'].replace('_', ' ').title()}\n"
                report_content += f"- Raw Score: {ev['raw_score']:.4f}\n"
                report_content += f"- Adjusted Score (used in final probability): {ev['adjusted_score']:.4f}\n"
                report_content += f"- Weight: {ev['weight']:.2f}\n"
                report_content += f"- Relevance: {ev['relevance']}\n"
                if ev['details']:
                    report_content += "Key Details:\n"
                    for detail_key, detail_value in ev['details'].items():
                        if isinstance(detail_value, list) and detail_value and isinstance(detail_value[0], dict):
                             report_content += f"  - {detail_key.replace('_', ' ').title()}: (Sample of {len(detail_value)})\n"
                             for item_idx, item_val in enumerate(detail_value[:3]): # Show sample
                                 report_content += f"    - Item {item_idx+1}: {json.dumps(item_val, indent=2)}\n"
                        elif isinstance(detail_value, dict) and detail_value:
                             report_content += f"  - {detail_key.replace('_', ' ').title()}:\n"
                             for sub_k, sub_v in detail_value.items():
                                 report_content += f"    - {sub_k.replace('_', ' ').title()}: {sub_v if not isinstance(sub_v, float) else f'{sub_v:.3f}'}\n"

                        else:
                            report_content += f"  - {detail_key.replace('_', ' ').title()}: {detail_value}\n"
                report_content += "\n"
        else:
            report_content += "No detailed evidence available.\n"
            
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self._log(f"Report saved to {report_path}")
        except Exception as e:
            self._log(f"Error saving report: {e}")

    def visualize_results(self, analysis_results: Dict[str, Any], viz_path: str) -> None:
        """
        Generate and save visualizations of the analysis results.
        This is a basic visualization; more sophisticated plots can be added.
        """
        self._log(f"Generating visualizations at {viz_path}...")
        if 'evidence' not in analysis_results or not analysis_results['evidence']:
            self._log("No evidence to visualize.")
            return

        methods = [ev['method'] for ev in analysis_results['evidence']]
        raw_scores = [ev['raw_score'] for ev in analysis_results['evidence']]
        adj_scores = [ev['adjusted_score'] for ev in analysis_results['evidence']]
        weights = [ev['weight'] for ev in analysis_results['evidence']]

        df_data = {
            'Method': methods,
            'Raw Score': raw_scores,
            'Adjusted Score': adj_scores,
            'Weight': weights
        }
        df = pd.DataFrame(df_data)

        if df.empty:
            self._log("Cannot generate plot: DataFrame is empty.")
            return

        try:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Raw Score', y='Method', data=df, color='skyblue', label='Raw Score')
            sns.barplot(x='Adjusted Score', y='Method', data=df, color='steelblue', alpha=0.7, label='Adjusted Score')
            
            # Add weights as text
            for i, row in df.iterrows():
                plt.text(row['Adjusted Score'] + 0.01, i, f"W: {row['Weight']:.2f}", va='center', color='black')


            plt.title(f"Analysis Metrics Overview (Prob: {analysis_results['probability']:.3f})")
            plt.xlabel("Score Value")
            plt.ylabel("Analysis Method")
            plt.legend()
            plt.tight_layout()
            plt.savefig(viz_path)
            plt.close() # Close the plot to free memory
            self._log(f"Visualization saved to {viz_path}")
        except Exception as e:
            self._log(f"Error generating visualization: {e}")
    
    def perform_ablation_study(self, dataset: List[Tuple[str, str, bool]], metric_to_optimize: str = 'f1') -> Dict[str, Any]:
        """
        Perform an ablation study to determine the importance of each component.
        It evaluates performance by removing one analytical method at a time.

        Args:
            dataset: List of (document, summary, ground_truth_label) tuples.
            metric_to_optimize: The metric to track for ablation (e.g., 'f1', 'precision', 'recall').

        Returns:
            Dict containing ablation study results.
        """
        self._log("Performing ablation study...")
        original_weights = self.weights.copy()
        baseline_performance = evaluate_performance(self, dataset) # Assuming evaluate_performance is defined elsewhere
        baseline_metric = baseline_performance.get(metric_to_optimize, 0.0)
        
        ablation_results = [{'ablated': 'None (baseline)', metric_to_optimize: baseline_metric, 'metrics': baseline_performance}]

        for method_to_ablate in original_weights.keys():
            self._log(f"Ablating: {method_to_ablate}")
            # Temporarily set the weight of the ablated method to 0
            self.weights[method_to_ablate] = 0.0
            
            # Re-evaluate
            performance_after_ablation = evaluate_performance(self, dataset)
            metric_after_ablation = performance_after_ablation.get(metric_to_optimize, 0.0)
            
            ablation_results.append({
                'ablated': method_to_ablate,
                metric_to_optimize: metric_after_ablation,
                f'{metric_to_optimize}_change': metric_after_ablation - baseline_metric,
                'metrics': performance_after_ablation
            })
            
            # Restore original weight for the next iteration
            self.weights = original_weights.copy()
            
        # Convert to DataFrame for easier viewing/saving
        ablation_df = pd.DataFrame(ablation_results)
        self._log("\nAblation Study Results:")
        self._log(ablation_df.to_string())

        return {
            'dataframe': ablation_df,
            'baseline_metric_value': baseline_metric,
            'metric_optimized': metric_to_optimize
        }

    def optimize_thresholds(self, dataset: List[Tuple[str, str, bool]], metric_to_optimize: str = 'f1') -> Dict[str, float]:
        """
        Optimize decision thresholds for individual metrics and the final probability.
        This is a simplified example; more sophisticated optimization can be done.
        Here, we focus on optimizing the final attribution threshold.
        Optimizing individual method thresholds (`self.thresholds`) is more complex
        and would likely involve a grid search or similar over many parameters.

        Args:
            dataset: List of (document, summary, ground_truth_label) tuples.
            metric_to_optimize: The metric to optimize (e.g., 'f1', 'precision').

        Returns:
            Dict of optimized thresholds (primarily the final attribution threshold).
        """
        self._log(f"Optimizing final attribution threshold for {metric_to_optimize}...")
        
        # Get probabilities for the entire dataset
        probabilities = []
        true_labels = []
        for doc, summ, label in dataset:
            # Use a non-verbose analysis for speed during optimization
            original_verbose = self.verbose
            self.verbose = False
            analysis_result = self.analyze(doc, summ) 
            self.verbose = original_verbose
            probabilities.append(analysis_result['probability'])
            true_labels.append(label)

        if not probabilities:
            self._log("No probabilities generated, cannot optimize thresholds.")
            return {'final_attribution_threshold': 0.5} # Default

        best_threshold = 0.5 # Default
        best_metric_value = -1.0

        # Test a range of possible thresholds for the final probability
        for p_thresh in np.arange(0.05, 1.0, 0.01): # Iterate from 0.05 to 0.99
            predictions = [p >= p_thresh for p in probabilities]
            
            current_metric_value = 0.0
            if metric_to_optimize == 'f1':
                current_metric_value = f1_score(true_labels, predictions, zero_division=0)
            elif metric_to_optimize == 'precision':
                current_metric_value = precision_score(true_labels, predictions, zero_division=0)
            elif metric_to_optimize == 'recall':
                current_metric_value = recall_score(true_labels, predictions, zero_division=0)
            # Add other metrics if needed
            
            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_threshold = p_thresh
        
        self._log(f"Optimized final attribution threshold: {best_threshold:.4f} (achieved {metric_to_optimize}: {best_metric_value:.4f})")
        
        # This function primarily optimizes the *final decision threshold* used after `analyze` returns a probability.
        # The internal `self.thresholds` for scaling are not optimized here but could be with a more complex search.
        return {'final_attribution_threshold': round(best_threshold, 4)}


# --- Main script functions (CLI, evaluation, dataset creation) ---

def compare_texts(document_path: str, summary_path: str, output_dir: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Compare a document and summary to determine if the summary was derived from the document.

    Args:
        document_path: Path to the document file.
        summary_path: Path to the summary file.
        output_dir: Optional directory for saving reports and visualizations.
        verbose: Whether to print detailed information.

    Returns:
        Dict containing analysis results.
    """
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            document = f.read()
    except FileNotFoundError:
        logger.error(f"Error: Document file not found at {document_path}")
        return {} # Or raise error
    except Exception as e:
        logger.error(f"Error reading document file {document_path}: {e}")
        return {}


    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = f.read()
    except FileNotFoundError:
        logger.error(f"Error: Summary file not found at {summary_path}")
        return {}
    except Exception as e:
        logger.error(f"Error reading summary file {summary_path}: {e}")
        return {}

    # Initialize analyzer
    analyzer = DocumentSummaryAttributionAnalyzer(verbose=verbose)

    # Analyze texts
    # The 'threshold' in analyze() is the one for the 'attribution' boolean flag.
    # This can be a fixed value or an optimized one if available.
    # For single analysis, a default like 0.5 or 0.7 is common.
    results = analyzer.analyze(document, summary, threshold=0.7) 

    # Calculate confidence intervals
    # Check if results were successfully produced before proceeding
    if results and results.get('probability') is not None: # Check if analysis was successful
        ci_results = analyzer.calculate_confidence_intervals(document, summary, n_bootstrap=100)
        results['confidence_intervals'] = ci_results

        # Test adversarial robustness
        robustness_results = analyzer.test_adversarial_robustness(document, summary)
        results['robustness'] = robustness_results
    else:
        logger.warning("Analysis did not produce valid results. Skipping CI and robustness.")


    # Generate report and visualizations if output directory is specified
    if output_dir and results: # Ensure results exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate and save report
        report_path = os.path.join(output_dir, 'attribution_report.md')
        analyzer.generate_report(document, summary, results, report_path)

        # Generate and save visualizations
        viz_path = os.path.join(output_dir, 'attribution_visualization.png')
        analyzer.visualize_results(results, viz_path)

        # Save raw results as JSON
        json_path = os.path.join(output_dir, 'attribution_results.json')
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                # Convert numpy values to native Python types for JSON serialization
                def convert_for_json(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)): # Handle pandas time objects if they appear
                        return str(obj)
                    return obj
                
                # Apply conversion recursively if needed, or ensure results are already clean
                # json.dumps with default can handle many cases, but direct conversion is safer.
                # For simplicity here, we assume 'results' is mostly serializable after basic numpy conversion.
                # A more robust approach would be to clean 'results' more thoroughly.
                json_results = json.loads(json.dumps(results, default=convert_for_json))
                json.dump(json_results, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving JSON results: {e}")
            
    return results


def evaluate_performance(analyzer: DocumentSummaryAttributionAnalyzer, test_dataset: List[Tuple[str,str,bool]], attribution_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Evaluate analyzer performance on labeled test data.

    Args:
        analyzer: The DocumentSummaryAttributionAnalyzer instance.
        test_dataset: List of (document, summary, ground_truth_label) tuples.
        attribution_threshold: The probability threshold to decide attribution.

    Returns:
        Dict containing precision, recall, F1, and ROC-AUC metrics.
    """
    predictions = []
    true_labels = []
    probabilities = []
    
    original_verbose = analyzer.verbose
    analyzer.verbose = False # Suppress logs during batch evaluation

    for i, (document, summary, ground_truth) in enumerate(test_dataset):
        # logger.info(f"Evaluating item {i+1}/{len(test_dataset)}") # Optional progress
        if not isinstance(document, str) or not isinstance(summary, str):
            logger.warning(f"Skipping item {i+1} due to invalid document/summary type.")
            continue
        if not document.strip() or not summary.strip():
            logger.warning(f"Skipping item {i+1} due to empty document/summary string.")
            # Assign a default prediction for empty inputs if necessary, or skip.
            # For now, skipping. This might affect dataset size consistency if not handled.
            continue

        try:
            results = analyzer.analyze(document, summary, threshold=attribution_threshold)
            # Use the fixed attribution_threshold for decision, not the one from analyze() if it differs.
            predictions.append(results['probability'] >= attribution_threshold) 
            true_labels.append(ground_truth)
            probabilities.append(results['probability'])
        except Exception as e:
            logger.error(f"Error analyzing item {i+1} during evaluation: {e}. Skipping.")
            # Decide how to handle: skip, or assign a default prediction (e.g., False)
            # and ensure true_labels also reflects this handling if an item is effectively removed.

    analyzer.verbose = original_verbose # Restore verbosity

    if not true_labels or not predictions: # If all items were skipped or failed
        logger.warning("No valid items processed in evaluation. Returning empty metrics.")
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': None,
            'specificity': 0.0, 'npv': 0.0, 'confusion_matrix': [[0,0],[0,0]],
            'sample_size': 0, 'processed_samples': 0
        }

    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    auc = None
    if len(set(true_labels)) > 1 and probabilities: # ROC AUC needs at least two classes and valid probabilities
        try:
            auc = roc_auc_score(true_labels, probabilities)
        except ValueError as e:
            logger.warning(f"Could not calculate AUC: {e}. Usually means only one class present in y_true after filtering.")
            auc = None # Or 0.5 if only one class, depends on convention
    else:
        logger.info("AUC not calculated (single class in true labels or no probabilities).")


    cm = confusion_matrix(true_labels, predictions).tolist() # Ensure it's a list for JSON
    # Handle cases where cm might not be 2x2 if only one class predicted/true
    if len(cm) == 1: # Only one class (e.g. all true, all predicted true)
        if true_labels[0] == 1: # All positive
            tn, fp, fn, tp = 0,0,0, cm[0][0]
        else: # All negative
            tn, fp, fn, tp = cm[0][0], 0,0,0
    elif len(cm) == 2 :
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    else: # Should not happen with boolean labels
        tn,fp,fn,tp = 0,0,0,0


    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc) if auc is not None else None,
        'specificity': float(specificity),
        'npv': float(npv),
        'confusion_matrix': cm,
        'sample_size': len(test_dataset), # Original size
        'processed_samples': len(true_labels) # Actual processed
    }


def create_synthetic_test_dataset(n_samples: int = 100, document_sources: Optional[List[str]] = None) -> List[Tuple[str, str, bool]]:
    """
    Create a synthetic test dataset with labeled examples.

    Args:
        n_samples: Number of test samples to create.
        document_sources: Optional list of document texts to use as sources.

    Returns:
        List of (document, summary, label) tuples.
    """
    logger.info(f"Creating synthetic dataset with {n_samples} samples...")
    # NLTK resources should be downloaded by Analyzer init, but good to ensure here too for standalone use
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt', quiet=True)

    documents = []
    if document_sources:
        documents = document_sources
    else:
        # Create some generic synthetic documents if none provided
        topics = ["artificial intelligence", "climate change", "global economy", "space exploration", "biotechnology"]
        for i in range(max(10, n_samples // 5)): # Ensure a decent pool of base documents
            topic = random.choice(topics)
            doc_len = random.randint(8, 25) # sentences
            doc_sentences = []
            base_phrases = [
                f"The domain of {topic} is rapidly evolving.", f"Recent advancements in {topic} show significant promise.",
                f"Challenges in {topic} include various complex factors.", f"The future of {topic} is a subject of ongoing debate.",
                f"Researchers are exploring new frontiers in {topic}.", f"Understanding {topic} requires a multidisciplinary approach.",
                f"Ethical considerations are paramount in {topic}.", f"The societal impact of {topic} cannot be understated."
            ]
            for _ in range(doc_len):
                doc_sentences.append(random.choice(base_phrases) + " " + " ".join(random.sample(base_phrases, min(2, len(base_phrases)-1)) if len(base_phrases)>1 else [""]))
            documents.append(" ".join(doc_sentences))
    
    if not documents: # Fallback if still no documents
        documents = ["This is a default document about science. It has several sentences. Science is important.", "Another default document about technology. Technology changes fast. It impacts our lives."]

    test_data = []
    for i in range(n_samples):
        document = random.choice(documents)
        doc_sents = sent_tokenize(document)
        if not doc_sents: doc_sents = ["Fallback sentence."] # Ensure not empty

        label = (i < n_samples // 2) # Roughly half positive, half negative

        if label: # Positive example (derived summary)
            if len(doc_sents) <= 2:
                summary = " ".join(doc_sents)
            else:
                num_summary_sents = max(1, min(len(doc_sents) -1, random.randint(1, len(doc_sents) // 2 + 1)))
                summary_sents_indices = sorted(random.sample(range(len(doc_sents)), num_summary_sents))
                summary_sents = [doc_sents[j] for j in summary_sents_indices]
                # Simple paraphrasing: shuffle sentence order sometimes, or slightly alter one sentence
                if random.random() < 0.3 and len(summary_sents) > 1: random.shuffle(summary_sents)
                if random.random() < 0.2 and summary_sents: # Replace a word
                    s_idx = random.randint(0, len(summary_sents)-1)
                    words = summary_sents[s_idx].split()
                    if len(words) > 3:
                         words[random.randint(0,len(words)-1)] = "altered"
                         summary_sents[s_idx] = " ".join(words)
                summary = " ".join(summary_sents)
        else: # Negative example (unrelated summary)
            other_doc_choices = [d for d in documents if d != document]
            if not other_doc_choices: other_doc_choices = documents # Use any if only one base doc
            
            other_doc = random.choice(other_doc_choices)
            other_doc_sents = sent_tokenize(other_doc)
            if not other_doc_sents: other_doc_sents = ["Unrelated fallback sentence."]

            if len(other_doc_sents) <= 2:
                summary = " ".join(other_doc_sents)
            else:
                num_summary_sents = max(1, min(len(other_doc_sents)-1, random.randint(1, len(other_doc_sents) // 2 + 1)))
                summary_sents_indices = sorted(random.sample(range(len(other_doc_sents)), num_summary_sents))
                summary = " ".join([other_doc_sents[j] for j in summary_sents_indices])
        
        if not summary.strip(): summary = "Default summary text." # Ensure summary is not empty
        test_data.append((document, summary, label))
        
    logger.info(f"Synthetic dataset created with {len(test_data)} samples.")
    return test_data


def create_challenging_test_cases() -> List[Tuple[str, str, bool]]:
    """
    Create a set of challenging test cases to evaluate the analyzer.
    (Using the definitions from the original script)
    """
    logger.info("Creating challenging test cases...")
    test_cases = []
    doc1 = "The effects of climate change are becoming increasingly evident worldwide. Rising global temperatures have led to more frequent and severe weather events, including hurricanes, floods, and droughts. Melting ice caps and glaciers contribute to rising sea levels, threatening coastal communities. Ecosystems are being disrupted, with many species facing extinction as their habitats change faster than they can adapt. Agriculture is also affected, with changing growing seasons and increased water scarcity impacting crop yields. To address these challenges, countries worldwide are working to reduce carbon emissions through policies promoting renewable energy, energy efficiency, and conservation."
    summary1 = "Global warming impacts are now unmistakable across the planet. Temperature increases are causing weather extremes like cyclones, inundations, and water shortages. As ice formations dissolve, oceans rise, endangering shoreline populations. Natural systems face upheaval, pushing numerous organisms toward extinction when environmental changes outpace adaptation. Farming suffers from altered growing periods and water limitations affecting harvests. Nations are responding with carbon reduction strategies, emphasizing sustainable power, efficient energy usage, and environmental protection."
    test_cases.append((doc1, summary1, True))  # Heavily paraphrased but derived

    doc2 = "Machine learning algorithms can be broadly categorized into supervised, unsupervised, and reinforcement learning. Supervised learning involves training models on labeled data to make predictions or classifications. Unsupervised learning discovers patterns in unlabeled data through techniques like clustering and dimensionality reduction. Reinforcement learning trains agents to make sequences of decisions by rewarding desired behaviors and penalizing undesired ones. Each approach has specific applications, strengths, and limitations. Hybrid approaches combining multiple learning paradigms are increasingly common in complex real-world applications."
    summary2 = "Artificial intelligence encompasses various learning methodologies. In supervised approaches, algorithms learn from annotated examples to generalize to new cases. Unsupervised techniques identify structures in untagged information sets. Deep learning represents a specialized subset using neural networks with multiple layers to process complex data representations. Transfer learning allows models trained on one task to be adapted for another related task. The ethical implications of AI decision-making have become increasingly important considerations in system design and implementation."
    test_cases.append((doc2, summary2, False)) # Same topic but not derived

    doc3 = "The Renaissance was a period of European cultural, artistic, political, and scientific rebirth following the Middle Ages. Beginning in the 14th century in Italy, this movement revitalized interest in classical learning and values. Renaissance art emphasized realism and humanist themes, with masters like Leonardo da Vinci and Michelangelo creating enduring masterpieces. The invention of the printing press by Johannes Gutenberg revolutionized communication and knowledge sharing. The Renaissance worldview placed increasing emphasis on individual experience, critical thinking, and a more humanistic perspective."
    summary3 = """The European cultural movement known as the Enlightenment prioritized reason and individualism over tradition. While unrelated to "the Renaissance" period, both eras valued rational thought. The Enlightenment emphasized scientific methodology and challenged religious dogma, occurring primarily in the 18th century. Like how "the printing press by Johannes Gutenberg revolutionized communication," the Enlightenment saw innovations in spreading ideas through journals and newspapers. Both periods shared some values, such as "individual experience, critical thinking, and a more humanistic perspective," but represented distinct historical movements."""
    test_cases.append((doc3, summary3, False)) # Contains quotes but overall not derived (this label might be debatable, depends on definition of "derived")

    doc4 = "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy. This process occurs primarily in the chloroplasts of plant cells, specifically in the chlorophyll-containing thylakoid membranes. The process has two main stages: the light-dependent reactions and the Calvin cycle. In the light-dependent reactions, energy from sunlight is used to split water molecules, releasing oxygen as a byproduct. The Calvin cycle uses the energy produced in the light-dependent reactions to convert carbon dioxide into glucose. Photosynthesis is essential for life on Earth, as it produces oxygen and serves as the primary entry point for energy into the food web."
    summary4 = "The food web's primary energy source comes from photosynthesis, which also produces the oxygen we breathe. This biochemical pathway involves a two-phase process: initially, light-driven reactions split H2O molecules, releasing O2, followed by the Calvin cycle where CO2 is transformed into glucose using energy from the first phase. Various organisms including green plants, certain bacteria, and algae can perform this transformation of solar energy into chemical energy. The process takes place in specialized cell structures - chloroplasts - specifically within their chlorophyll-containing thylakoid membranes."
    test_cases.append((doc4, summary4, True))  # Different structure but derived

    doc5 = "Regular physical activity provides numerous health benefits. Exercise strengthens the cardiovascular system, reducing the risk of heart disease and stroke. It helps maintain healthy body weight and can prevent or manage conditions like type 2 diabetes. Physical activity strengthens bones and muscles, improving overall mobility and reducing the risk of falls in older adults. Exercise also has mental health benefits, reducing symptoms of depression and anxiety while improving mood and cognitive function."
    summary5 = "Regular physical activity offers multiple health advantages, particularly for cardiovascular health and weight management. Exercise significantly reduces the risk of developing type 2 diabetes and helps those with the condition manage it more effectively. Additionally, physical activity provides important bone and muscle benefits. Recent studies have shown that high-intensity interval training (HIIT) may be more time-efficient than traditional endurance training, though both are beneficial. Research also indicates that exercise can reduce cancer risk, particularly for colon and breast cancers. The World Health Organization now recommends at least 150 minutes of moderate-intensity exercise weekly."
    test_cases.append((doc5, summary5, False)) # Summary contains significant information not in document
    
    logger.info(f"Created {len(test_cases)} challenging test cases.")
    return test_cases


def evaluate_with_cross_validation(analyzer: DocumentSummaryAttributionAnalyzer, dataset: List[Tuple[str,str,bool]], n_folds: int = 5) -> Dict[str, Any]:
    """
    Evaluate the analyzer using k-fold cross-validation.
    """
    logger.info(f"Starting {n_folds}-fold cross-validation...")
    # Ensure dataset is a list for KFold compatibility
    dataset_list = list(dataset)
    if not dataset_list:
        logger.error("Dataset is empty. Cannot perform cross-validation.")
        return {} # Return empty or default dict
        
    # Convert to numpy array for KFold indexing if needed, or ensure labels are extractable
    # For simplicity, we'll work with list of indices from kf.split
    
    labels_for_stratification = [label for _, _, label in dataset_list]
    if len(set(labels_for_stratification)) < 2:
         # StratifiedKFold needs at least 2 classes. Fallback to KFold if only one class.
         kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
         logger.warning("Only one class present in dataset labels. Using KFold without stratification.")
    else:
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)


    all_fold_metrics = []
    
    # Optimized thresholds can be determined per fold on train set, or a global one can be used.
    # For this example, we'll use a default attribution_threshold for evaluate_performance,
    # but one could call analyzer.optimize_thresholds on train_data within each fold.

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset_list, labels_for_stratification)):
        logger.info(f"Evaluating fold {fold + 1}/{n_folds}")
        
        train_data = [dataset_list[i] for i in train_idx]
        test_data = [dataset_list[i] for i in test_idx]

        # Optional: Optimize thresholds on this training fold
        # optimized_fold_thresholds = analyzer.optimize_thresholds(train_data)
        # attribution_decision_threshold = optimized_fold_thresholds.get('final_attribution_threshold', 0.7)
        attribution_decision_threshold = 0.7 # Using a fixed threshold for simplicity here

        fold_metrics = evaluate_performance(analyzer, test_data, attribution_threshold=attribution_decision_threshold)
        fold_metrics['fold'] = fold + 1
        all_fold_metrics.append(fold_metrics)

    # Calculate average metrics across folds
    avg_metrics = {}
    # Metrics that can be averaged: precision, recall, f1, specificity, npv, auc (if not None)
    metric_keys_to_average = ['precision', 'recall', 'f1', 'specificity', 'npv']
    
    for key in metric_keys_to_average:
        values = [m[key] for m in all_fold_metrics if key in m and m[key] is not None]
        if values:
            avg_metrics[f'avg_{key}'] = float(np.mean(values))
            avg_metrics[f'std_{key}'] = float(np.std(values))
        else:
            avg_metrics[f'avg_{key}'] = 0.0
            avg_metrics[f'std_{key}'] = 0.0

    # Handle AUC separately due to potential None values
    auc_values = [m['auc'] for m in all_fold_metrics if 'auc' in m and m['auc'] is not None]
    if auc_values:
        avg_metrics['avg_auc'] = float(np.mean(auc_values))
        avg_metrics['std_auc'] = float(np.std(auc_values))
    else:
        avg_metrics['avg_auc'] = None
        avg_metrics['std_auc'] = None
        
    avg_metrics['fold_results'] = all_fold_metrics # Store individual fold results
    logger.info("Cross-validation complete.")
    return avg_metrics


def analyze_error_patterns(analyzer: DocumentSummaryAttributionAnalyzer, test_dataset: List[Tuple[str,str,bool]], attribution_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Analyze and categorize error patterns in the attribution analysis.
    """
    logger.info("Analyzing error patterns...")
    error_cases = {'false_positives': [], 'false_negatives': []}
    fp_count = 0
    fn_count = 0

    original_verbose = analyzer.verbose
    analyzer.verbose = False # Suppress logs

    for i, (document, summary, ground_truth) in enumerate(test_dataset):
        if not isinstance(document, str) or not isinstance(summary, str) or not document.strip() or not summary.strip():
            continue # Skip invalid data

        try:
            results = analyzer.analyze(document, summary, threshold=attribution_threshold)
            # Use the fixed attribution_threshold for decision
            prediction = results['probability'] >= attribution_threshold

            if prediction and not ground_truth:  # False positive
                fp_count += 1
                if len(error_cases['false_positives']) < 20: # Limit stored cases
                    error_cases['false_positives'].append({
                        'document_snippet': document[:200] + "...",
                        'summary': summary,
                        'probability': results['probability'],
                        'metrics_sample': {k: v.get('score', 'N/A') for k,v in results.get('metrics', {}).items() if isinstance(v,dict)}
                    })
            elif not prediction and ground_truth:  # False negative
                fn_count += 1
                if len(error_cases['false_negatives']) < 20: # Limit stored cases
                    error_cases['false_negatives'].append({
                        'document_snippet': document[:200] + "...",
                        'summary': summary,
                        'probability': results['probability'],
                        'metrics_sample': {k: v.get('score', 'N/A') for k,v in results.get('metrics', {}).items() if isinstance(v,dict)}
                    })
        except Exception as e:
            logger.error(f"Error analyzing item {i+1} during error pattern analysis: {e}")

    analyzer.verbose = original_verbose # Restore

    return {
        'false_positive_analysis': {'count': fp_count},
        'false_negative_analysis': {'count': fn_count},
        'error_cases_sample': error_cases # Contains samples of FP/FN
    }


# --- Main execution block ---
if __name__ == "__main__":
    import random # For synthetic dataset generation

    parser = argparse.ArgumentParser(description='Analyze if a summary was derived from a document, or evaluate the analyzer.')
    subparsers = parser.add_subparsers(dest='operation', help='Operation to perform', required=True)

    # Analyze operation
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single document-summary pair')
    analyze_parser.add_argument('document', help='Path to the document file')
    analyze_parser.add_argument('summary', help='Path to the summary file')
    analyze_parser.add_argument('--output', '-o', help='Output directory for reports and visualizations', default=None)
    analyze_parser.add_argument('--quiet', '-q', action='store_true', help='Suppress detailed output (sets verbose=False for analyzer)')

    # Evaluate operation
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate analyzer on a test dataset')
    eval_parser.add_argument('--dataset', '-d', help='Path to dataset file (CSV or JSON lines)', required=True)
    eval_parser.add_argument('--output', '-o', help='Output directory for evaluation results', default=None)
    eval_parser.add_argument('--folds', '-f', type=int, default=5, help='Number of folds for cross-validation (0 for simple train/test split if supported, or full dataset eval)')
    eval_parser.add_argument('--attribution_threshold', type=float, default=0.7, help='Probability threshold for attribution decision during evaluation.')


    # Synthetic dataset creation
    synth_parser = subparsers.add_parser('create-dataset', help='Create a synthetic test dataset')
    synth_parser.add_argument('--samples', '-n', type=int, default=100, help='Number of samples to generate')
    synth_parser.add_argument('--output', '-o', help='Output file path for dataset (e.g., dataset.jsonl or dataset.csv)', required=True)

    # Optimization operation (Placeholder for more advanced optimization)
    optim_parser = subparsers.add_parser('optimize', help='Optimize analyzer parameters (e.g., final decision threshold)')
    optim_parser.add_argument('--dataset', '-d', help='Path to dataset file (CSV or JSON lines)', required=True)
    optim_parser.add_argument('--output', '-o', help='Output file for optimized parameters (e.g., optimized_params.json)', default='optimized_params.json')
    optim_parser.add_argument('--metric', choices=['f1', 'precision', 'recall'], default='f1', help="Metric to optimize for.")


    args = parser.parse_args()
    analyzer_verbosity = not args.quiet if hasattr(args, 'quiet') else True


    if args.operation == 'analyze':
        results = compare_texts(args.document, args.summary, args.output, verbose=analyzer_verbosity)
        if results: # If compare_texts ran successfully
            print(f"\nAnalysis Results:")
            print(f"Attribution Probability: {results.get('probability', 'N/A'):.4f}")
            # The 'attribution' key in results already used a threshold (default 0.7 in analyze method)
            print(f"Attribution Decision: {'DERIVED FROM DOCUMENT' if results.get('attribution') else 'NOT DERIVED FROM DOCUMENT'}")
            print(f"Confidence: {results.get('confidence', 'N/A'):.4f}")
            print(f"\nExplanation:\n{results.get('explanation', 'N/A')}")

            if 'confidence_intervals' in results and results['confidence_intervals']:
                ci = results['confidence_intervals']
                print(f"\nBootstrap Confidence Interval (95%):")
                print(f"  Mean Probability: {ci.get('mean_probability', 'N/A'):.4f}")
                print(f"  CI: [{ci.get('ci_lower', 'N/A'):.4f}, {ci.get('ci_upper', 'N/A'):.4f}]")
            
            if args.output:
                print(f"\nDetailed results, report, and visualizations saved to {args.output}")
        else:
            print("Analysis could not be completed. Check logs for errors.")


    elif args.operation == 'evaluate':
        dataset = []
        try:
            if args.dataset.endswith('.csv'):
                df = pd.read_csv(args.dataset)
                # Assuming columns 'document', 'summary', 'label' (0 or 1)
                for _, row in df.iterrows():
                    dataset.append((str(row['document']), str(row['summary']), bool(row['label'])))
            elif args.dataset.endswith('.jsonl') or args.dataset.endswith('.json'): # JSON Lines or a list of JSON objects
                with open(args.dataset, 'r', encoding='utf-8') as f:
                    if args.dataset.endswith('.jsonl'):
                        for line in f:
                            item = json.loads(line)
                            dataset.append((str(item['document']), str(item['summary']), bool(item['label'])))
                    else: # Assume a single JSON array of objects
                        data_list = json.load(f)
                        for item in data_list:
                             dataset.append((str(item['document']), str(item['summary']), bool(item['label'])))
            else:
                print("Unsupported dataset format. Use CSV or JSON/JSONL with 'document', 'summary', 'label' fields.")
                exit(1)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            exit(1)
        
        if not dataset:
            print("Dataset is empty or failed to load.")
            exit(1)

        analyzer_instance = DocumentSummaryAttributionAnalyzer(verbose=analyzer_verbosity)
        
        if args.folds > 0:
            cv_results = evaluate_with_cross_validation(analyzer_instance, dataset, n_folds=args.folds)
            print("\nCross-Validation Results:")
            for key, value in cv_results.items():
                if key != 'fold_results': # Don't print all individual fold details here
                    if isinstance(value, float): print(f"{key.replace('_', ' ').title()}: {value:.4f}")
                    else: print(f"{key.replace('_', ' ').title()}: {value}")
        else: # Evaluate on the whole dataset without CV
            print("\nEvaluating on the full dataset (no cross-validation):")
            full_eval_results = evaluate_performance(analyzer_instance, dataset, attribution_threshold=args.attribution_threshold)
            for key, value in full_eval_results.items():
                 if isinstance(value, float): print(f"{key.replace('_', ' ').title()}: {value:.4f}")
                 else: print(f"{key.replace('_', ' ').title()}: {value}")
            cv_results = {'full_dataset_evaluation': full_eval_results} # Store for saving


        # Perform error analysis on the full dataset
        error_analysis_results = analyze_error_patterns(analyzer_instance, dataset, attribution_threshold=args.attribution_threshold)
        print("\nError Analysis (on full dataset):")
        print(f"False Positives Count: {error_analysis_results['false_positive_analysis'].get('count', 0)}")
        print(f"False Negatives Count: {error_analysis_results['false_negative_analysis'].get('count', 0)}")

        if args.output:
            os.makedirs(args.output, exist_ok=True)
            cv_path = os.path.join(args.output, 'evaluation_results.json')
            with open(cv_path, 'w') as f:
                # Need a robust way to serialize, especially if DataFrames or numpy arrays are in cv_results
                json.dump(cv_results, f, indent=2, default=lambda o: str(o) if isinstance(o, (np.ndarray, pd.DataFrame)) else o.__dict__ if hasattr(o, '__dict__') else str(o))
            
            error_path = os.path.join(args.output, 'error_analysis_details.json')
            with open(error_path, 'w') as f:
                json.dump(error_analysis_results, f, indent=2, default=str)
            print(f"\nEvaluation and error analysis results saved to {args.output}")


    elif args.operation == 'create-dataset':
        print(f"Creating synthetic dataset with {args.samples} samples...")
        dataset = create_synthetic_test_dataset(n_samples=args.samples)
        
        # Optionally add challenging test cases
        if input("Add challenging test cases to the synthetic dataset? (y/n): ").lower() == 'y':
            challenging_cases = create_challenging_test_cases()
            dataset.extend(challenging_cases)
            print(f"Added {len(challenging_cases)} challenging cases. Total samples: {len(dataset)}")

        if args.output.endswith('.csv'):
            df = pd.DataFrame([{'document': doc, 'summary': summary, 'label': 1 if label else 0} 
                               for doc, summary, label in dataset])
            df.to_csv(args.output, index=False)
        elif args.output.endswith('.jsonl'):
            with open(args.output, 'w', encoding='utf-8') as f:
                for doc, summary, label in dataset:
                    json.dump({'document': doc, 'summary': summary, 'label': label}, f)
                    f.write('\n')
        else: # Default to JSON array
            data_out = [{'document': doc, 'summary': summary, 'label': label} 
                        for doc, summary, label in dataset]
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(data_out, f, indent=2)
        print(f"Dataset saved to {args.output}")


    elif args.operation == 'optimize':
        dataset = []
        # Load dataset (similar to 'evaluate' block)
        try:
            if args.dataset.endswith('.csv'):
                df = pd.read_csv(args.dataset)
                for _, row in df.iterrows(): dataset.append((str(row['document']), str(row['summary']), bool(row['label'])))
            elif args.dataset.endswith('.jsonl') or args.dataset.endswith('.json'):
                with open(args.dataset, 'r', encoding='utf-8') as f:
                    if args.dataset.endswith('.jsonl'):
                        for line in f: dataset.append((str(json.loads(line)['document']), str(json.loads(line)['summary']), bool(json.loads(line)['label'])))
                    else:
                        for item in json.load(f): dataset.append((str(item['document']), str(item['summary']), bool(item['label'])))
            else:
                print("Unsupported dataset format for optimization. Use CSV or JSON/JSONL.")
                exit(1)
        except Exception as e:
            print(f"Error loading dataset for optimization: {e}")
            exit(1)

        if not dataset:
            print("Dataset for optimization is empty.")
            exit(1)

        analyzer_instance = DocumentSummaryAttributionAnalyzer(verbose=analyzer_verbosity)

        # 1. Optimize final decision threshold
        print(f"\nOptimizing final attribution threshold for metric: {args.metric}...")
        optimized_thresholds_result = analyzer_instance.optimize_thresholds(dataset, metric_to_optimize=args.metric)
        print(f"Optimized final attribution threshold: {optimized_thresholds_result.get('final_attribution_threshold', 'N/A'):.4f}")

        # 2. Perform Ablation Study (optional, can be computationally intensive)
        ablation_results_data = None
        if input("Perform ablation study? (y/n, can be slow): ").lower() == 'y':
            print("\nPerforming ablation study...")
            ablation_results_data = analyzer_instance.perform_ablation_study(dataset, metric_to_optimize=args.metric)
            print("\nAblation Study Summary (change in F1 score):")
            if 'dataframe' in ablation_results_data:
                 print(ablation_results_data['dataframe'][['ablated', f'{args.metric}_change']].to_string())


        if args.output:
            output_data = {'optimized_final_threshold': optimized_thresholds_result}
            if ablation_results_data and 'dataframe' in ablation_results_data:
                # Convert DataFrame to dict for JSON serialization
                output_data['ablation_study_results'] = ablation_results_data['dataframe'].to_dict(orient='records')
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nOptimization results saved to {args.output}")
            
    else:
        parser.print_help()

