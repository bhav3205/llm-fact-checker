"""
LLM-Powered Fact Checker
A production-grade RAG system for claim verification
"""

__version__ = "1.0.0"
__author__ = "Artikate Studio Assignment"

from .claim_extractor import AdvancedClaimExtractor
from .embedding_manager import EmbeddingManager
from .vector_store_qdrant import QdrantVectorStore
from .fact_verifier_groq import GroqFactVerifier
from .data_ingestion import PIBDataIngestion
from .pipeline import FactCheckingPipeline

__all__ = [
    'AdvancedClaimExtractor',
    'EmbeddingManager',
    'QdrantVectorStore',
    'GroqFactVerifier',
    'PIBDataIngestion',
    'FactCheckingPipeline'
]
