"""
Gradio Application for Fact Checker
Alternative interface with simpler deployment
"""

import gradio as gr
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import FactCheckingPipeline
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pipeline
config = {
    'claim_extractor_model': 'en_core_web_sm',
    'embedding_model': 'all-MiniLM-L6-v2',
    'vector_store': 'qdrant',
    'llm_model': 'llama-3.1-8b',
    'collection_name': 'fact_checker',
    'vector_db_path': './data/vector_db'
}

print("Initializing Fact-Checking Pipeline...")
pipeline = FactCheckingPipeline(config)
pipeline.build_fact_base(max_statements=50)
print("Pipeline ready!")


def check_fact(claim_text, top_k):
    """
    Check a claim and return formatted results.
    
    Args:
        claim_text: Input claim
        top_k: Number of evidence pieces
        
    Returns:
        Formatted results for Gradio interface
    """
    if not claim_text.strip():
        return "❌ Please enter a claim to verify.", "", "", ""
    
    try:
        result = pipeline.check_claim(claim_text, top_k=int(top_k))
        
        if 'error' in result:
            return f"❌ Error: {result['error']}", "", "", ""
        
        # Format verdict
        verdict = result.get('verdict', 'Unknown')
        confidence = result.get('confidence', 0.0)
        
        verdict_emoji = {
            "True": "✅",
            "False": "❌",
            "Unverifiable": "🤷‍♂️",
            "Partially True": "⚖️"
        }.get(verdict, "❓")
        
        verdict_output = f"""
# {verdict_emoji} Verdict: {verdict}

**Confidence Score:** {confidence:.1%}  
**Processing Time:** {result.get('processing_time', 0)}s  
**Model:** {result.get('model', 'N/A')}  
**Tokens Used:** {result.get('tokens_used', 0)} (FREE)
"""
        
        # Format reasoning
        reasoning = result.get('reasoning', 'No reasoning available')
        reasoning_output = f"""
## 🧠 Chain-of-Thought Reasoning

{reasoning}
"""
        
        # Format evidence
        evidence_list = result.get('evidence', [])
        evidence_scores = result.get('evidence_scores', [])
        
        evidence_output = "## 📚 Supporting Evidence\n\n"
        evidence_output += "*Retrieved from PIB Government Data*\n\n"
        
        for i, (evidence, score) in enumerate(zip(evidence_list, evidence_scores), 1):
            evidence_output += f"### Evidence {i} (Relevance: {score:.1%})\n"
            evidence_output += f"{evidence}\n\n"
            evidence_output += "---\n\n"
        
        # Format JSON
        json_output = json.dumps(result, indent=2, default=str)
        
        return verdict_output, reasoning_output, evidence_output, json_output
        
    except Exception as e:
        logger.error(f"Error checking claim: {e}")
        return f"❌ Error: {str(e)}", "", "", ""


# Example claims
examples = [
    ["The Indian government has announced free electricity to all farmers starting July 2025.", 5],
    ["PM-KISAN provides Rs. 6000 per year to farmer families in three equal installments.", 5],
    ["India launched its first manned space mission in 2024.", 5],
    ["Ayushman Bharat provides health coverage of Rs. 5 lakh per family per year.", 5],
]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="LLM Fact Checker") as demo:
    gr.Markdown("""
    # ✅ LLM-Powered Fact Checker
    ### Advanced RAG System for Claim Verification | Artikate Studio Assignment
    
    Built with **Qdrant** (vector store) + **Groq** (LLM) + **Sentence-Transformers** (embeddings)
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            claim_input = gr.Textbox(
                label="Enter Claim to Verify",
                placeholder="Type a claim or select an example below...",
                lines=3
            )
            
            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Evidence Pieces"
                )
            
            with gr.Row():
                check_btn = gr.Button("🚀 Check Fact", variant="primary", size="lg")
                clear_btn = gr.ClearButton([claim_input], value="🔄 Clear")
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 💡 Quick Info
            
            **Tech Stack:**
            - 🧠 spaCy for NER
            - 📊 Sentence-Transformers
            - 🗄️ Qdrant Vector DB
            - 🤖 Groq Llama-3.1 (FREE)
            
            **Data Source:**
            PIB India Official RSS Feed
            
            **Cost:** $0 (100% FREE)
            """)
    
    gr.Markdown("### 📋 Results")
    
    with gr.Tab("Verdict"):
        verdict_output = gr.Markdown(label="Verdict")
    
    with gr.Tab("Reasoning"):
        reasoning_output = gr.Markdown(label="Reasoning")
    
    with gr.Tab("Evidence"):
        evidence_output = gr.Markdown(label="Evidence")
    
    with gr.Tab("JSON Export"):
        json_output = gr.Code(label="Full Result (JSON)", language="json")
    
    gr.Markdown("### 💡 Try These Examples")
    gr.Examples(
        examples=examples,
        inputs=[claim_input, top_k_slider],
        label="Sample Claims"
    )
    
    # Event handlers
    check_btn.click(
        fn=check_fact,
        inputs=[claim_input, top_k_slider],
        outputs=[verdict_output, reasoning_output, evidence_output, json_output]
    )
    
    gr.Markdown("""
    ---
    <div style="text-align: center;">
        <p><strong>LLM-Powered Fact Checker</strong> | Artikate Studio Assignment</p>
        <p>Built with ❤️ using FREE open-source tools</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
