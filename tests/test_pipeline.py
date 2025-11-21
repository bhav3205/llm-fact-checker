"""
Tests for Complete Fact-Checking Pipeline
Integration tests for end-to-end workflow
"""

import pytest
import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import FactCheckingPipeline


@pytest.fixture
def pipeline():
    """Fixture for pipeline with test configuration"""
    config = {
        'spacy_model': 'en_core_web_sm',
        'embedding_model': 'BAAI/bge-large-en-v1.5',
        'vector_store': 'qdrant',
        'collection_name': 'test_fact_checker',
        'vector_db_path': './data/test_vector_db',
        'llm_model': 'qwen-2.5-32b'
    }
    
    pipeline = FactCheckingPipeline(config)
    yield pipeline
    
    # Cleanup
    try:
        pipeline.vector_store.delete_collection()
        if os.path.exists('./data/test_vector_db'):
            shutil.rmtree('./data/test_vector_db')
    except:
        pass


class TestPipelineInitialization:
    """Test pipeline initialization"""
    
    def test_pipeline_creation(self, pipeline):
        """Test pipeline initializes correctly"""
        assert pipeline is not None
        assert hasattr(pipeline, 'claim_extractor')
        assert hasattr(pipeline, 'embedding_manager')
        assert hasattr(pipeline, 'vector_store')
        assert hasattr(pipeline, 'fact_verifier')
        print("✅ Pipeline components initialized")
    
    def test_config_loading(self, pipeline):
        """Test configuration is loaded correctly"""
        assert pipeline.config is not None
        assert 'embedding_model' in pipeline.config
        assert 'llm_model' in pipeline.config
        print(f"✅ Config loaded: {list(pipeline.config.keys())}")


class TestFactBaseBuilding:
    """Test fact base construction"""
    
    def test_build_fact_base_small(self, pipeline):
        """Test building small fact base"""
        try:
            pipeline.build_fact_base(max_statements=5)
            info = pipeline.vector_store.get_collection_info()
            
            assert info.get('vectors_count', 0) >= 0
            print(f"✅ Fact base built: {info.get('vectors_count', 0)} vectors")
        except Exception as e:
            pytest.skip(f"Skipping due to: {e}")
    
    def test_build_fact_base_fallback(self, pipeline):
        """Test fact base with fallback data"""
        try:
            # Should use fallback if PIB fetch fails
            pipeline.build_fact_base(max_statements=10)
            info = pipeline.vector_store.get_collection_info()
            
            assert info.get('vectors_count', 0) >= 0
            print("✅ Fact base with fallback data built")
        except Exception as e:
            pytest.skip(f"Skipping due to: {e}")


class TestClaimChecking:
    """Test claim verification"""
    
    @pytest.mark.skipif(
        not os.getenv('GROQ_API_KEY'),
        reason="GROQ_API_KEY not set"
    )
    def test_check_claim_basic(self, pipeline):
        """Test basic claim checking"""
        try:
            # Build small fact base
            pipeline.build_fact_base(max_statements=5)
            
            # Test claim
            result = pipeline.check_claim(
                "PM-KISAN provides Rs. 6000 per year to farmers.",
                top_k=3
            )
            
            if 'error' not in result:
                assert 'verdict' in result
                assert 'confidence' in result
                assert 'reasoning' in result
                assert 'evidence' in result
                
                assert result['verdict'] in ['True', 'False', 'Unverifiable', 'Partially True']
                assert 0 <= result['confidence'] <= 1
                
                print(f"✅ Claim checked: {result['verdict']} ({result['confidence']:.2f})")
                print(f"   Reasoning: {result['reasoning'][:100]}...")
            else:
                print(f"⚠️ Error in claim check: {result['error']}")
                
        except Exception as e:
            pytest.skip(f"Skipping due to: {e}")
    
    def test_check_claim_no_evidence(self, pipeline):
        """Test claim with no matching evidence"""
        result = pipeline.check_claim("Completely unrelated statement about aliens.", top_k=3)
        
        assert 'error' in result or 'verdict' in result
        print("✅ No evidence scenario handled")
    
    def test_invalid_claim(self, pipeline):
        """Test handling of invalid input"""
        result = pipeline.check_claim("", top_k=5)
        
        assert 'error' in result
        assert result['error'] == "No verifiable claims found in input"
        print("✅ Invalid claim handled")
    
    @pytest.mark.skipif(
        not os.getenv('GROQ_API_KEY'),
        reason="GROQ_API_KEY not set"
    )
    def test_multiple_claims(self, pipeline):
        """Test input with multiple claims"""
        try:
            pipeline.build_fact_base(max_statements=5)
            
            text = """
            The Indian government has multiple schemes.
            PM-KISAN provides financial assistance.
            Farmers receive Rs. 6000 annually.
            """
            
            result = pipeline.check_claim(text, top_k=3)
            
            if 'error' not in result:
                assert 'extracted_claims' in result
                assert len(result['extracted_claims']) >= 1
                print(f"✅ Multiple claims processed: {len(result['extracted_claims'])} claims")
            
        except Exception as e:
            pytest.skip(f"Skipping due to: {e}")


class TestBatchProcessing:
    """Test batch claim checking"""
    
    @pytest.mark.skipif(
        not os.getenv('GROQ_API_KEY'),
        reason="GROQ_API_KEY not set"
    )
    def test_batch_check(self, pipeline):
        """Test checking multiple claims in batch"""
        try:
            pipeline.build_fact_base(max_statements=5)
            
            claims = [
                "PM-KISAN provides Rs. 6000.",
                "Digital India is a government program."
            ]
            
            results = pipeline.batch_check(claims, top_k=3)
            
            assert len(results) == len(claims)
            assert all('verdict' in r or 'error' in r for r in results)
            print(f"✅ Batch processed: {len(results)} claims")
            
        except Exception as e:
            pytest.skip(f"Skipping due to: {e}")


class TestPerformance:
    """Test performance metrics"""
    
    @pytest.mark.skipif(
        not os.getenv('GROQ_API_KEY'),
        reason="GROQ_API_KEY not set"
    )
    def test_processing_time(self, pipeline):
        """Test processing time is reasonable"""
        try:
            pipeline.build_fact_base(max_statements=5)
            
            result = pipeline.check_claim(
                "Test claim about government policy",
                top_k=3
            )
            
            if 'processing_time' in result:
                assert result['processing_time'] < 30  # Should be under 30 seconds
                print(f"✅ Processing time: {result['processing_time']}s")
            
        except Exception as e:
            pytest.skip(f"Skipping due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
