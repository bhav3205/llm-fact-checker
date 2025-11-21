"""
Fact Verifier using Groq API - 100% FREE with ultra-fast inference
Groq provides fastest LLM inference (up to 800 tokens/sec!)
"""

from groq import Groq
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroqFactVerifier:
    """
    Uses Groq's FREE ultra-fast LLM API for fact verification.
    
    FREE Tier Limits (per day):
    - Llama-3.1-8B: 14,400 tokens/min, 30 requests/min
    - Llama-3.1-70B: 6,000 tokens/min, 30 requests/min
    - Mixtral-8x7B: 5,000 tokens/min, 30 requests/min
    
    Speed: Up to 800 tokens/second (fastest in the market!)
    """
    
    AVAILABLE_MODELS = {
        "llama-3.1-8b": "llama-3.1-8b-instant",
        "llama-3.1-70b": "llama-3.1-70b-versatile",
        "mixtral": "mixtral-8x7b-32768",
        "llama-3.2-90b": "llama-3.2-90b-text-preview"
    }
    
    def __init__(self, model: str = "llama-3.1-8b", api_key: Optional[str] = None):
        """
        Initialize Groq client.
        
        Args:
            model: Model shortname from AVAILABLE_MODELS
            api_key: Groq API key (get free at https://console.groq.com)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Groq API key required! Get FREE key at: https://console.groq.com\n"
                "Then set GROQ_API_KEY in .env file"
            )
        
        self.client = Groq(api_key=self.api_key)
        self.model = self.AVAILABLE_MODELS.get(model, self.AVAILABLE_MODELS["llama-3.1-8b"])
        
        logger.info(f"Initialized Groq with model: {self.model} (FREE tier)")
    
    def verify_claim(
        self, 
        claim: str, 
        evidence: List[Dict],
        include_reasoning: bool = True
    ) -> Dict:
        """
        Verify claim using Groq's ultra-fast LLM.
        
        Args:
            claim: Claim to verify
            evidence: Retrieved evidence
            include_reasoning: Include chain-of-thought
            
        Returns:
            Verification result with verdict, confidence, reasoning
        """
        # Construct prompt
        prompt = self._build_verification_prompt(claim, evidence)
        
        try:
            # Call Groq API (typically responds in <1 second!)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1024,
                top_p=0.95
            )
            
            result_text = response.choices[0].message.content
            
            # Parse structured output
            verdict_info = self._parse_response(result_text)
            
            # Add metadata
            verdict_info.update({
                "claim": claim,
                "evidence": [e['text'] for e in evidence],
                "evidence_scores": [e['score'] for e in evidence],
                "model": self.model,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            })
            
            logger.info(f"Verification complete: {verdict_info['verdict']} "
                       f"(confidence: {verdict_info['confidence']:.2f})")
            
            return verdict_info
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return {
                "verdict": "Error",
                "confidence": 0.0,
                "reasoning": f"Error during verification: {str(e)}",
                "claim": claim,
                "evidence": [e['text'] for e in evidence]
            }
    
    def _get_system_prompt(self) -> str:
        """Optimized system prompt for fact-checking"""
        return """You are an expert fact-checker specialized in verifying claims against provided evidence.

Your task: Analyze claims with precision and determine truthfulness based ONLY on provided evidence.

Guidelines:
1. Compare claim against evidence point-by-point
2. Check for temporal accuracy (dates, timelines)
3. Verify scope (e.g., "all" vs "some")
4. Identify contradictions or partial matches
5. Flag unverifiable claims

Output Format (strictly follow):
VERDICT: [True/False/Unverifiable/Partially True]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Detailed explanation with evidence references]

Be precise, objective, and evidence-based."""
    
    def _build_verification_prompt(self, claim: str, evidence: List[Dict]) -> str:
        """Build structured prompt for verification"""
        # Format evidence with scores
        evidence_text = "\n\n".join([
            f"Evidence {i+1} (Relevance: {e['score']:.2f}):\n{e['text']}"
            for i, e in enumerate(evidence)
        ])
        
        prompt = f"""Verify this claim against the provided evidence:

CLAIM:
"{claim}"

RETRIEVED EVIDENCE:
{evidence_text}

Analysis Steps:
1. Does any evidence directly support or contradict the claim?
2. Are there temporal mismatches (dates, timeframes)?
3. Are there scope differences (scale, coverage)?
4. Is the evidence sufficient for a determination?

Provide your verdict with confidence score and detailed reasoning."""
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict:
        """Parse structured LLM response"""
        lines = response.strip().split('\n')
        
        verdict = "Unverifiable"
        confidence = 0.5
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("VERDICT:"):
                verdict_text = line.replace("VERDICT:", "").strip().lower()
                if "true" in verdict_text and "partially" not in verdict_text and "false" not in verdict_text:
                    verdict = "True"
                elif "false" in verdict_text:
                    verdict = "False"
                elif "partially" in verdict_text or "partial" in verdict_text:
                    verdict = "Partially True"
                else:
                    verdict = "Unverifiable"
            
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_text = line.replace("CONFIDENCE:", "").strip()
                    confidence = float(conf_text)
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    confidence = 0.5
            
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            
            elif reasoning and not line.startswith(("VERDICT:", "CONFIDENCE:")):
                reasoning += " " + line
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning.strip() if reasoning else "No reasoning provided"
        }
