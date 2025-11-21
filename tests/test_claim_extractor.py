"""
Tests for Advanced Claim Extractor
Tests NER, claim decomposition, and checkworthiness assessment
"""

import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.claim_extractor import AdvancedClaimExtractor


@pytest.fixture
def extractor():
    """Fixture for claim extractor"""
    return AdvancedClaimExtractor(spacy_model="en_core_web_sm")


class TestClaimExtraction:
    """Test suite for claim extraction"""
    
    def test_initialization(self, extractor):
        """Test extractor initializes correctly"""
        assert extractor is not None
        assert extractor.nlp is not None
        assert extractor.device in ["cuda", "cpu"]
    
    def test_extract_atomic_claims(self, extractor):
        """Test atomic claim extraction"""
        text = "The Indian government announced a new policy. Farmers will receive Rs. 6000 per year."
        claims = extractor.extract_atomic_claims(text)
        
        assert len(claims) > 0
        assert isinstance(claims, list)
        assert all(isinstance(claim, str) for claim in claims)
        print(f"✅ Extracted {len(claims)} claims: {claims}")
    
    def test_extract_entities(self, extractor):
        """Test named entity extraction"""
        text = "The Indian government announced Rs. 6000 for farmers in July 2025."
        entities = extractor.extract_entities(text)
        
        assert len(entities) > 0
        assert all('text' in e and 'label' in e for e in entities)
        
        # Check for expected entity types
        entity_labels = [e['label'] for e in entities]
        assert any(label in ['ORG', 'GPE', 'DATE', 'MONEY'] for label in entity_labels)
        print(f"✅ Extracted {len(entities)} entities: {entities}")
    
    def test_process_input_complete(self, extractor):
        """Test complete processing pipeline"""
        text = "PM-KISAN provides Rs. 6000 per year to farmer families in three equal installments."
        result = extractor.process_input(text)
        
        assert 'original_text' in result
        assert 'atomic_claims' in result
        assert 'entities' in result
        assert 'primary_claim' in result
        assert 'checkworthy_claims' in result
        
        assert result['original_text'] == text
        assert len(result['atomic_claims']) > 0
        print(f"✅ Complete extraction result: {result.keys()}")
    
    def test_empty_input(self, extractor):
        """Test handling of empty input"""
        result = extractor.process_input("")
        
        assert 'atomic_claims' in result
        assert isinstance(result['atomic_claims'], list)
        print("✅ Empty input handled gracefully")
    
    def test_temporal_extraction(self, extractor):
        """Test temporal information extraction"""
        text = "The policy will be implemented in July 2025 and continue until December 2030."
        temporal = extractor.extract_temporal_info(text)
        
        assert isinstance(temporal, list)
        if temporal:
            assert all('date' in t for t in temporal)
        print(f"✅ Extracted temporal info: {temporal}")
    
    def test_multiple_sentences(self, extractor):
        """Test claim extraction from multiple sentences"""
        text = """
        The Indian government has announced a new scheme for farmers.
        Under PM-KISAN, Rs. 6000 is provided annually.
        This scheme benefits over 10 crore farmers.
        """
        claims = extractor.extract_atomic_claims(text)
        
        assert len(claims) >= 2
        print(f"✅ Extracted {len(claims)} claims from multiple sentences")
    
    def test_checkworthiness_assessment(self, extractor):
        """Test checkworthiness scoring"""
        claims = [
            "PM-KISAN provides Rs. 6000 per year.",  # Should be checkworthy
            "I think the policy is good.",  # Should not be checkworthy
            "The government exists."  # Vague
        ]
        
        checkworthy = extractor.assess_checkworthiness(claims)
        
        assert isinstance(checkworthy, list)
        if checkworthy:
            assert all(isinstance(item, tuple) and len(item) == 2 for item in checkworthy)
            # First claim should have highest score
            print(f"✅ Checkworthiness scores: {checkworthy}")


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_long_text(self, extractor):
        """Test with very long input"""
        text = "The government announced a new policy. " * 100
        result = extractor.process_input(text)
        
        assert 'atomic_claims' in result
        assert len(result['atomic_claims']) > 0
        print("✅ Long text handled")
    
    def test_special_characters(self, extractor):
        """Test with special characters"""
        text = "PM-KISAN provides ₹6,000/year (Rs. 6000) to farmers! #policy"
        result = extractor.process_input(text)
        
        assert 'entities' in result
        print("✅ Special characters handled")
    
    def test_non_english_mixed(self, extractor):
        """Test with mixed language (should handle gracefully)"""
        text = "The सरकार announced Rs. 6000 for किसान families."
        result = extractor.process_input(text)
        
        assert 'atomic_claims' in result
        print("✅ Mixed language handled")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
