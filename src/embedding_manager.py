"""
Embedding Manager Module
Handles text embedding generation using sentence-transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages text embeddings using sentence-transformer models.
    Supports batch processing and caching for efficiency.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager with specified model.
        
        Args:
            model_name: Sentence-transformer model name
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully (dim={self.embedding_dim})")
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of embeddings
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, 
                     show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          corpus_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and corpus embeddings.
        
        Args:
            query_embedding: Query embedding vector
            corpus_embeddings: Matrix of corpus embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        corpus_norm = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(corpus_norm, query_norm)
        
        return similarities
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim
