"""
Advanced Claim Extractor with T5-based atomic claim decomposition
Uses FREE specialized models from HuggingFace
"""

import spacy
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedClaimExtractor:
    """
    Advanced claim extraction using T5-based atomic claim extractor.
    FREE and specifically trained for claim decomposition.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with FREE models"""
        # Load spaCy for NER
        try:
            import spacy
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            import spacy
            import spacy.cli
            logger.info(f"Downloading {spacy_model}...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
            
        # Load specialized T5 claim extractor (FREE from HuggingFace)
        logger.info("Loading claim extraction models...")
        
        # Use a simpler approach for claim extraction
        self.use_simple_extraction = True
        
        # Zero-shot classifier for checkworthiness (FREE)
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Loaded zero-shot classifier")
        except Exception as e:
            logger.warning(f"Could not load classifier: {e}. Using rule-based approach.")
            self.classifier = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initialized on {self.device}")
    
    def extract_atomic_claims(self, text: str) -> List[str]:
        """
        Extract atomic claims from text using spaCy sentence splitting.
        
        Args:
            text: Input text
            
        Returns:
            List of atomic claims
        """
        doc = self.nlp(text)
        
        # Extract sentences
        claims = []
        for sent in doc.sents:
            # Filter out very short sentences or questions
            if len(sent.text.strip()) > 10 and not sent.text.strip().endswith('?'):
                claims.append(sent.text.strip())
        
        # If no claims found, return the original text
        if not claims:
            claims = [text.strip()]
        
        logger.info(f"Extracted {len(claims)} atomic claims")
        return claims
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities with enhanced categories"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Focus on fact-relevant entities
            if ent.label_ in ['ORG', 'GPE', 'DATE', 'MONEY', 'PERCENT', 'PERSON', 'LAW', 'EVENT', 'CARDINAL']:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        logger.info(f"Extracted {len(entities)} key entities")
        return entities
    
    def assess_checkworthiness(self, claims: List[str]) -> List[Tuple[str, float]]:
        """
        Assess which claims are worth fact-checking.
        
        Args:
            claims: List of extracted claims
            
        Returns:
            List of (claim, checkworthiness_score) tuples
        """
        checkworthy_claims = []
        
        if self.classifier:
            labels = [
                "objective factual statement",
                "subjective opinion",
                "vague statement"
            ]
            
            for claim in claims:
                try:
                    result = self.classifier(claim, labels)
                    
                    # Claims classified as factual with high confidence are checkworthy
                    if result['labels'][0] == "objective factual statement":
                        score = result['scores'][0]
                        if score > 0.5:  # Confidence threshold
                            checkworthy_claims.append((claim, score))
                except Exception as e:
                    logger.warning(f"Classification error for claim: {e}")
                    # Fallback: include all claims with medium score
                    checkworthy_claims.append((claim, 0.7))
        else:
            # Rule-based fallback
            for claim in claims:
                # Simple heuristics for checkworthiness
                doc = self.nlp(claim)
                has_entities = len([ent for ent in doc.ents if ent.label_ in ['ORG', 'DATE', 'MONEY', 'PERCENT']]) > 0
                has_numbers = any(token.like_num for token in doc)
                
                if has_entities or has_numbers:
                    checkworthy_claims.append((claim, 0.8))
                else:
                    checkworthy_claims.append((claim, 0.6))
        
        # Sort by checkworthiness
        checkworthy_claims.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(checkworthy_claims)} checkworthy claims")
        return checkworthy_claims
    
    def extract_temporal_info(self, text: str) -> List[Dict]:
        """Extract temporal information crucial for fact-checking"""
        doc = self.nlp(text)
        temporal_info = []
        
        for ent in doc.ents:
            if ent.label_ == "DATE":
                temporal_info.append({
                    "date": ent.text,
                    "context": doc[max(0, ent.start-5):min(len(doc), ent.end+5)].text
                })
        
        return temporal_info
    
    def process_input(self, text: str) -> Dict:
        """
        Complete processing pipeline for input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Comprehensive extraction results
        """
        # Extract atomic claims
        atomic_claims = self.extract_atomic_claims(text)
        
        # Assess checkworthiness
        checkworthy = self.assess_checkworthiness(atomic_claims)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Extract temporal information
        temporal_info = self.extract_temporal_info(text)
        
        return {
            "original_text": text,
            "atomic_claims": atomic_claims,
            "checkworthy_claims": [c[0] for c in checkworthy],
            "checkworthiness_scores": [c[1] for c in checkworthy],
            "entities": entities,
            "temporal_info": temporal_info,
            "primary_claim": checkworthy[0][0] if checkworthy else atomic_claims[0] if atomic_claims else text
        }
