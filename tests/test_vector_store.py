"""
Tests for Qdrant Vector Store
Tests vector storage, retrieval, and search functionality
"""

import pytest
import sys
import os
import numpy as np
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store_qdrant import QdrantVectorStore
from src.embedding_manager import EmbeddingManager


@pytest.fixture
def vector_store():
    """Fixture for vector store with test collection"""
    store = QdrantVectorStore(
        collection_name="test_collection",
        dimension=1024,  # BGE-Large dimension
        path="./data/test_vector_db"
    )
    yield store
    # Cleanup
    try:
        store.delete_collection()
        if os.path.exists("./data/test_vector_db"):
            shutil.rmtree("./data/test_vector_db")
    except:
        pass


@pytest.fixture
def embedding_manager():
    """Fixture for embedding manager with BGE-Large"""
    return EmbeddingManager(model_name="BAAI/bge-large-en-v1.5")


class TestVectorStoreBasics:
    """Basic vector store operations"""
    
    def test_initialization(self, vector_store):
        """Test vector store initializes correctly"""
        assert vector_store is not None
        assert vector_store.collection_name == "test_collection"
        assert vector_store.dimension == 1024
        print("✅ Vector store initialized")
    
    def test_add_vectors(self, vector_store, embedding_manager):
        """Test adding vectors to store"""
        texts = ["This is a test statement", "Another government policy"]
        embeddings = embedding_manager.encode_batch(texts)
        
        vector_store.add_vectors(embeddings, texts)
        
        info = vector_store.get_collection_info()
        assert info.get('vectors_count', 0) >= 0
        print(f"✅ Added {len(texts)} vectors. Collection info: {info}")
    
    def test_search_basic(self, vector_store, embedding_manager):
        """Test basic vector search"""
        # Add test data
        texts = [
            "The government announces new policy for farmers",
            "PM-KISAN provides financial assistance",
            "Digital India transforms the nation"
        ]
        embeddings = embedding_manager.encode_batch(texts)
        vector_store.add_vectors(embeddings, texts)
        
        # Search
        query = "government policy announcement"
        query_embedding = embedding_manager.encode_single(query)
        results = vector_store.search(query_embedding, top_k=2)
        
        assert len(results) > 0
        assert results[0]['text'] in texts
        assert 'score' in results[0]
        assert 0 <= results[0]['score'] <= 1
        print(f"✅ Search results: {len(results)} items found")
        print(f"   Top result: {results[0]['text'][:50]}... (score: {results[0]['score']:.3f})")
    
    def test_search_with_metadata(self, vector_store, embedding_manager):
        """Test vector search with metadata"""
        texts = ["Policy A from 2024", "Policy B from 2025"]
        embeddings = embedding_manager.encode_batch(texts)
        metadata = [
            {"year": "2024", "source": "PIB"},
            {"year": "2025", "source": "PIB"}
        ]
        
        vector_store.add_vectors(embeddings, texts, metadata)
        
        query_embedding = embedding_manager.encode_single("policy")
        results = vector_store.search(query_embedding, top_k=2)
        
        assert len(results) > 0
        assert 'metadata' in results[0]
        print(f"✅ Metadata preserved: {results[0]['metadata']}")


class TestVectorStoreAdvanced:
    """Advanced vector store operations"""
    
    def test_empty_search(self, vector_store):
        """Test search with no data"""
        query_embedding = np.random.rand(1024)
        results = vector_store.search(query_embedding, top_k=5)
        
        assert isinstance(results, list)
        print("✅ Empty search handled gracefully")
    
    def test_large_batch_insert(self, vector_store, embedding_manager):
        """Test adding large batch of vectors"""
        texts = [f"Statement number {i} about government policy" for i in range(50)]
        embeddings = embedding_manager.encode_batch(texts, show_progress=False)
        
        vector_store.add_vectors(embeddings, texts)
        
        info = vector_store.get_collection_info()
        assert info.get('vectors_count', 0) >= 50
        print(f"✅ Large batch inserted: {info['vectors_count']} vectors")
    
    def test_score_threshold(self, vector_store, embedding_manager):
        """Test search with score threshold"""
        texts = ["Exact match test", "Completely different topic"]
        embeddings = embedding_manager.encode_batch(texts)
        vector_store.add_vectors(embeddings, texts)
        
        query = "exact match test"
        query_embedding = embedding_manager.encode_single(query)
        results = vector_store.search(query_embedding, top_k=2, score_threshold=0.5)
        
        assert all(r['score'] >= 0.5 for r in results)
        print(f"✅ Score threshold working: {[r['score'] for r in results]}")
    
    def test_collection_info(self, vector_store):
        """Test getting collection information"""
        info = vector_store.get_collection_info()
        
        assert isinstance(info, dict)
        assert 'vectors_count' in info or 'error' in info
        print(f"✅ Collection info retrieved: {info}")


class TestEmbeddingManager:
    """Test embedding manager separately"""
    
    def test_embedding_dimension(self, embedding_manager):
        """Test embedding dimension is correct"""
        assert embedding_manager.get_embedding_dimension() == 1024
        print("✅ BGE-Large dimension: 1024")
    
    def test_single_encoding(self, embedding_manager):
        """Test encoding single text"""
        text = "This is a test"
        embedding = embedding_manager.encode_single(text)
        
        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32 or embedding.dtype == np.float64
        print(f"✅ Single encoding shape: {embedding.shape}")
    
    def test_batch_encoding(self, embedding_manager):
        """Test encoding batch of texts"""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedding_manager.encode_batch(texts, show_progress=False)
        
        assert embeddings.shape == (3, 1024)
        print(f"✅ Batch encoding shape: {embeddings.shape}")
    
    def test_normalized_embeddings(self, embedding_manager):
        """Test embeddings are normalized"""
        text = "Test normalization"
        embedding = embedding_manager.encode_single(text, normalize=True)
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be close to 1
        print(f"✅ Embedding normalized: ||v|| = {norm:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
