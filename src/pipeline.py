# """
# Main Pipeline Module
# Orchestrates the entire fact-checking workflow
# """

# from typing import Dict, List, Optional
# import logging
# from .claim_extractor import AdvancedClaimExtractor
# from .embedding_manager import EmbeddingManager
# from .vector_store_qdrant import QdrantVectorStore
# from .fact_verifier_groq import GroqFactVerifier
# from .data_ingestion import PIBDataIngestion
# import time
# import yaml
# import os

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class FactCheckingPipeline:
#     """
#     End-to-end fact-checking pipeline orchestrating all components.
#     """
    
#     def __init__(self, config: Optional[Dict] = None):
#         """
#         Initialize pipeline with configuration.
        
#         Args:
#             config: Configuration dictionary
#         """
#         if config is None:
#             config = self._load_default_config()
        
#         self.config = config
        
#         # Initialize all components
#         logger.info("Initializing fact-checking pipeline...")
        
#         try:
#             self.claim_extractor = AdvancedClaimExtractor(
#                 spacy_model=config.get('claim_extractor_model', 'en_core_web_sm')
#             )
            
#             self.embedding_manager = EmbeddingManager(
#                 model_name=config.get('embedding_model', 'all-MiniLM-L6-v2')
#             )
            
#             self.vector_store = QdrantVectorStore(
#                 collection_name=config.get('collection_name', 'fact_checker'),
#                 dimension=self.embedding_manager.get_embedding_dimension(),
#                 path=config.get('vector_db_path', './data/vector_db')
#             )
            
#             self.fact_verifier = GroqFactVerifier(
#                 model=config.get('llm_model', 'llama-3.1-8b')
#             )
            
#             self.data_ingestion = PIBDataIngestion()
            
#             logger.info("Pipeline initialization complete")
            
#         except Exception as e:
#             logger.error(f"Pipeline initialization error: {e}")
#             raise
    
#     def _load_default_config(self) -> Dict:
#         """Load default configuration"""
#         return {
#             'claim_extractor_model': 'en_core_web_sm',
#             'embedding_model': 'all-MiniLM-L6-v2',
#             'vector_store': 'qdrant',
#             'collection_name': 'fact_checker',
#             'vector_db_path': './data/vector_db',
#             'llm_model': 'llama-3.1-8b'
#         }
    
#     def build_fact_base(self, max_statements: int = 50):
#         """
#         Build the fact base by ingesting PIB data and creating embeddings.
        
#         Args:
#             max_statements: Maximum number of statements to ingest
#         """
#         logger.info("Building fact base...")
        
#         # Fetch PIB statements
#         statements = self.data_ingestion.fetch_pib_statements(max_items=max_statements)
        
#         # Prepare facts
#         facts = self.data_ingestion.prepare_fact_base(statements)
        
#         # Generate embeddings
#         texts = [fact['text'] for fact in facts]
#         embeddings = self.embedding_manager.encode_batch(texts, show_progress=True)
        
#         # Store in vector database
#         metadata = [{k: v for k, v in fact.items() if k != 'text'} for fact in facts]
#         self.vector_store.add_vectors(embeddings, texts, metadata)
        
#         logger.info(f"Fact base built with {len(facts)} statements")
        
#         # Log collection info
#         info = self.vector_store.get_collection_info()
#         logger.info(f"Collection info: {info}")
    
#     def check_claim(self, input_text: str, top_k: int = 5) -> Dict:
#         """
#         Check a claim through the full pipeline.
        
#         Args:
#             input_text: Input text containing claim(s)
#             top_k: Number of similar facts to retrieve
            
#         Returns:
#             Verification result dictionary
#         """
#         start_time = time.time()
        
#         # Step 1: Extract claims
#         logger.info("Step 1: Extracting claims...")
#         claim_info = self.claim_extractor.process_input(input_text)
#         claims = claim_info['checkworthy_claims']
        
#         if not claims:
#             claims = claim_info['atomic_claims']
        
#         if not claims:
#             return {
#                 "error": "No verifiable claims found in input",
#                 "input": input_text
#             }
        
#         # For simplicity, verify the first (primary) claim
#         primary_claim = claims[0]
#         logger.info(f"Primary claim: {primary_claim}")
        
#         # Step 2: Generate query embedding
#         logger.info("Step 2: Generating embeddings...")
#         query_embedding = self.embedding_manager.encode_single(primary_claim)
        
#         # Step 3: Retrieve similar facts
#         logger.info("Step 3: Retrieving similar facts...")
#         evidence = self.vector_store.search(query_embedding, top_k=top_k)
        
#         if not evidence:
#             logger.warning("No evidence found in vector store")
#             return {
#                 "error": "No relevant evidence found",
#                 "claim": primary_claim,
#                 "input": input_text
#             }
        
#         # Step 4: Verify with LLM
#         logger.info("Step 4: Verifying with LLM...")
#         verification_result = self.fact_verifier.verify_claim(primary_claim, evidence)
        
#         # Add metadata
#         verification_result['input_text'] = input_text
#         verification_result['extracted_claims'] = claims
#         verification_result['all_claims'] = claim_info['atomic_claims']
#         verification_result['entities'] = claim_info['entities']
#         verification_result['processing_time'] = round(time.time() - start_time, 2)
        
#         logger.info(f"Fact-checking complete in {verification_result['processing_time']}s")
        
#         return verification_result
    
#     def batch_check(self, claims: List[str], top_k: int = 5) -> List[Dict]:
#         """
#         Check multiple claims in batch.
        
#         Args:
#             claims: List of claims to check
#             top_k: Number of evidence pieces per claim
            
#         Returns:
#             List of verification results
#         """
#         results = []
        
#         for claim in claims:
#             result = self.check_claim(claim, top_k)
#             results.append(result)
        
#         return results
"""
Main Pipeline Module
Orchestrates the entire fact-checking workflow
Now loads configuration from .env file
"""

from typing import Dict, List, Optional
import logging
from .claim_extractor import AdvancedClaimExtractor
from .embedding_manager import EmbeddingManager
from .vector_store_qdrant import QdrantVectorStore
from .fact_verifier_groq import GroqFactVerifier
from .data_ingestion import PIBDataIngestion
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactCheckingPipeline:
    """
    End-to-end fact-checking pipeline with BEST free models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pipeline with configuration from .env or provided config.
        
        Args:
            config: Configuration dictionary (optional, uses .env by default)
        """
        if config is None:
            config = self._load_config_from_env()
        
        self.config = config
        
        # Initialize all components
        logger.info("🚀 Initializing fact-checking pipeline with BEST models...")
        
        try:
            self.claim_extractor = AdvancedClaimExtractor(
                spacy_model=config.get('spacy_model', 'en_core_web_sm')
            )
            
            embedding_model = config.get('embedding_model', 'BAAI/bge-large-en-v1.5')
            logger.info(f"📊 Using embedding model: {embedding_model}")
            self.embedding_manager = EmbeddingManager(model_name=embedding_model)
            
            self.vector_store = QdrantVectorStore(
                collection_name=config.get('collection_name', 'fact_checker_v1'),
                dimension=self.embedding_manager.get_embedding_dimension(),
                path=config.get('vector_db_path', './data/vector_db')
            )
            
            llm_model = config.get('llm_model', 'qwen-2.5-32b')
            logger.info(f"🤖 Using LLM: {llm_model}")
            self.fact_verifier = GroqFactVerifier(model=llm_model)
            
            self.data_ingestion = PIBDataIngestion()
            
            logger.info("✅ Pipeline initialization complete with BEST models!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization error: {e}")
            raise
    
    def _load_config_from_env(self) -> Dict:
        """Load configuration from .env file"""
        return {
            'spacy_model': os.getenv('SPACY_MODEL', 'en_core_web_sm'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5'),
            'llm_model': os.getenv('LLM_MODEL', 'qwen-2.5-32b'),
            'vector_db_path': os.getenv('VECTOR_DB_PATH', './data/vector_db'),
            'collection_name': os.getenv('COLLECTION_NAME', 'fact_checker_v1')
        }
    
    def build_fact_base(self, max_statements: int = 50):
        """Build the fact base by ingesting PIB data and creating embeddings."""
        logger.info("📚 Building fact base...")
        
        # Fetch PIB statements
        statements = self.data_ingestion.fetch_pib_statements(max_items=max_statements)
        
        # Prepare facts
        facts = self.data_ingestion.prepare_fact_base(statements)
        
        if not facts:
            logger.error("No facts to process!")
            return
        
        # Generate embeddings
        texts = [fact['text'] for fact in facts]
        embeddings = self.embedding_manager.encode_batch(texts, show_progress=True)
        
        # Store in vector database
        metadata = [{k: v for k, v in fact.items() if k != 'text'} for fact in facts]
        self.vector_store.add_vectors(embeddings, texts, metadata)
        
        logger.info(f"✅ Fact base built with {len(facts)} statements")
        
        # Log collection info
        info = self.vector_store.get_collection_info()
        logger.info(f"📊 Collection info: {info}")
    
    def check_claim(self, input_text: str, top_k: int = 5) -> Dict:
        """
        Check a claim through the full pipeline.
        
        Args:
            input_text: Input text containing claim(s)
            top_k: Number of similar facts to retrieve
            
        Returns:
            Verification result dictionary
        """
        start_time = time.time()
        
        # Step 1: Extract claims
        logger.info("Step 1: Extracting claims...")
        claim_info = self.claim_extractor.process_input(input_text)
        claims = claim_info['checkworthy_claims']
        
        if not claims:
            claims = claim_info['atomic_claims']
        
        if not claims:
            return {
                "error": "No verifiable claims found in input",
                "input": input_text
            }
        
        # For simplicity, verify the first (primary) claim
        primary_claim = claims[0]
        logger.info(f"Primary claim: {primary_claim}")
        
        # Step 2: Generate query embedding
        logger.info("Step 2: Generating embeddings...")
        query_embedding = self.embedding_manager.encode_single(primary_claim)
        
        # Step 3: Retrieve similar facts
        logger.info("Step 3: Retrieving similar facts from Qdrant...")
        evidence = self.vector_store.search(query_embedding, top_k=top_k)
        
        if not evidence:
            logger.warning("No evidence found in vector store")
            return {
                "error": "No relevant evidence found",
                "claim": primary_claim,
                "input": input_text
            }
        
        # Step 4: Verify with LLM
        logger.info("Step 4: Verifying with LLM...")
        verification_result = self.fact_verifier.verify_claim(primary_claim, evidence)
        
        # Add metadata
        verification_result['input_text'] = input_text
        verification_result['extracted_claims'] = claims
        verification_result['all_claims'] = claim_info['atomic_claims']
        verification_result['entities'] = claim_info['entities']
        verification_result['processing_time'] = round(time.time() - start_time, 2)
        
        logger.info(f"✅ Fact-checking complete in {verification_result['processing_time']}s")
        
        return verification_result
    
    def batch_check(self, claims: List[str], top_k: int = 5) -> List[Dict]:
        """Check multiple claims in batch."""
        results = []
        
        for claim in claims:
            result = self.check_claim(claim, top_k)
            results.append(result)
        
        return results
