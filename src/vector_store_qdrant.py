# """
# Qdrant Vector Store - Open-source, production-ready, FREE
# Better than FAISS for production with built-in features
# """

# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from qdrant_client.http.models import Distance, VectorParams, PointStruct
# import numpy as np
# from typing import List, Dict, Optional
# import logging
# from uuid import uuid4
# import os

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class QdrantVectorStore:
#     """
#     Qdrant vector store with advanced features:
#     - Filtered search with metadata
#     - Automatic persistence
#     - Better scaling than FAISS
#     """
    
#     def __init__(
#         self, 
#         collection_name: str = "fact_checker",
#         dimension: int = 384,
#         path: str = "./data/vector_db",
#         distance_metric: str = "Cosine"
#     ):
#         """
#         Initialize Qdrant client (runs locally, FREE).
        
#         Args:
#             collection_name: Name of the collection
#             dimension: Embedding dimension
#             path: Path for local storage
#             distance_metric: Cosine, Dot, or Euclidean
#         """
#         self.collection_name = collection_name
#         self.dimension = dimension
        
#         # Create directory if it doesn't exist
#         os.makedirs(path, exist_ok=True)
        
#         # Initialize local Qdrant client (no server needed!)
#         self.client = QdrantClient(path=path)
        
#         # Map distance metrics
#         distance_map = {
#             "Cosine": Distance.COSINE,
#             "Dot": Distance.DOT,
#             "Euclidean": Distance.EUCLID
#         }
        
#         # Create collection if it doesn't exist
#         try:
#             self.client.get_collection(collection_name)
#             logger.info(f"Loaded existing collection: {collection_name}")
#         except Exception:
#             self.client.create_collection(
#                 collection_name=collection_name,
#                 vectors_config=VectorParams(
#                     size=dimension,
#                     distance=distance_map.get(distance_metric, Distance.COSINE)
#                 )
#             )
#             logger.info(f"Created new collection: {collection_name}")
    
#     def add_vectors(
#         self, 
#         embeddings: np.ndarray, 
#         texts: List[str], 
#         metadata: Optional[List[Dict]] = None
#     ):
#         """
#         Add vectors with automatic ID generation.
        
#         Args:
#             embeddings: Numpy array of embeddings
#             texts: List of texts
#             metadata: Optional metadata for filtering
#         """
#         if len(embeddings) == 0 or len(texts) == 0:
#             logger.warning("No vectors to add")
#             return
        
#         points = []
        
#         for i, (embedding, text) in enumerate(zip(embeddings, texts)):
#             # Prepare payload
#             payload = {"text": text}
#             if metadata and i < len(metadata):
#                 payload.update(metadata[i])
            
#             # Create point
#             point = PointStruct(
#                 id=str(uuid4()),  # Unique ID
#                 vector=embedding.tolist(),
#                 payload=payload
#             )
#             points.append(point)
        
#         # Batch upsert
#         self.client.upsert(
#             collection_name=self.collection_name,
#             points=points
#         )
        
#         logger.info(f"Added {len(points)} vectors to Qdrant")
    
#     def search(
#         self, 
#         query_embedding: np.ndarray, 
#         top_k: int = 5,
#         score_threshold: Optional[float] = None,
#         filter_conditions: Optional[Dict] = None
#     ) -> List[Dict]:
#         """
#         Search with optional filtering and score threshold.
        
#         Args:
#             query_embedding: Query vector
#             top_k: Number of results
#             score_threshold: Minimum similarity score
#             filter_conditions: Metadata filters
            
#         Returns:
#             List of results with text, score, and metadata
#         """
#         # Prepare search
#         search_params = {
#             "collection_name": self.collection_name,
#             "query_vector": query_embedding.tolist(),
#             "limit": top_k
#         }
        
#         # Add score threshold if provided
#         if score_threshold:
#             search_params["score_threshold"] = score_threshold
        
#         # Execute search
#         try:
#             results = self.client.search(**search_params)
#         except Exception as e:
#             logger.error(f"Search error: {e}")
#             return []
        
#         # Format results
#         formatted_results = []
#         for result in results:
#             formatted_results.append({
#                 "text": result.payload.get("text", ""),
#                 "score": float(result.score),
#                 "metadata": {k: v for k, v in result.payload.items() if k != "text"},
#                 "id": result.id
#             })
        
#         logger.info(f"Found {len(formatted_results)} results")
#         return formatted_results
    
#     def get_collection_info(self) -> Dict:
#         """Get statistics about the collection"""
#         try:
#             info = self.client.get_collection(self.collection_name)
#             return {
#                 "vectors_count": info.vectors_count,
#                 "points_count": info.points_count,
#                 "status": info.status
#             }
#         except Exception as e:
#             logger.error(f"Error getting collection info: {e}")
#             return {"error": str(e)}
    
#     def delete_collection(self):
#         """Delete the entire collection"""
#         try:
#             self.client.delete_collection(self.collection_name)
#             logger.info(f"Deleted collection: {self.collection_name}")
#         except Exception as e:
#             logger.error(f"Error deleting collection: {e}")

"""
Qdrant Vector Store - Open-source, production-ready, FREE
Better than FAISS for production with built-in features
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import numpy as np
from typing import List, Dict, Optional
import logging
from uuid import uuid4
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant vector store with advanced features:
    - Filtered search with metadata
    - Automatic persistence
    - Better scaling than FAISS
    """
    
    def __init__(
        self, 
        collection_name: str = "fact_checker",
        dimension: int = 384,
        path: str = "./data/vector_db",
        distance_metric: str = "Cosine"
    ):
        """
        Initialize Qdrant client (runs locally, FREE).
        """
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Initialize local Qdrant client (no server needed!)
        self.client = QdrantClient(path=path)
        
        # Map distance metrics
        distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclidean": Distance.EUCLID
        }
        
        # Create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=distance_map.get(distance_metric, Distance.COSINE)
                )
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_vectors(
        self, 
        embeddings: np.ndarray, 
        texts: List[str], 
        metadata: Optional[List[Dict]] = None
    ):
        """Add vectors with automatic ID generation."""
        if len(embeddings) == 0 or len(texts) == 0:
            logger.warning("No vectors to add")
            return
        
        points = []
        
        for i, (embedding, text) in enumerate(zip(embeddings, texts)):
            # Prepare payload
            payload = {"text": text}
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            
            # Create point
            point = PointStruct(
                id=str(uuid4()),
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        # Batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Added {len(points)} vectors to Qdrant")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search with optional filtering and score threshold.
        FIXED: Uses correct Qdrant client API.
        """
        try:
            # Check if collection has any points first
            collection_info = self.client.get_collection(self.collection_name)
            if hasattr(collection_info, 'points_count') and collection_info.points_count == 0:
                logger.warning("Collection is empty - no vectors to search")
                return []
            
            # FIXED: Use query_points which is the correct method
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=top_k,
                score_threshold=score_threshold if score_threshold else None
            )
            
            # Format results - handle response object
            formatted_results = []
            
            # Access points from response
            points = results.points if hasattr(results, 'points') else results
            
            for result in points:
                formatted_results.append({
                    "text": result.payload.get("text", ""),
                    "score": float(result.score) if hasattr(result, 'score') else 1.0,
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                    "id": result.id
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_collection_info(self) -> Dict:
        """Get statistics about the collection - FIXED"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count if hasattr(info, 'points_count') else 0,
                "vectors_count": info.points_count if hasattr(info, 'points_count') else 0,
                "status": str(info.status) if hasattr(info, 'status') else "unknown"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
