"""
Streamlit Application for Fact Checker
Professional UI with feedback mechanism
FIXED: Button handler now works correctly
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ⚡ CRITICAL: Load .env BEFORE importing anything else
from dotenv import load_dotenv
load_dotenv()

from src.pipeline import FactCheckingPipeline
from src.vector_store_qdrant import QdrantVectorStore
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LLM Fact Checker | Artikate Studio",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .verdict-true {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 20px 0;
    }
    .verdict-false {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 20px 0;
    }
    .verdict-unverifiable {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 20px 0;
    }
    .verdict-partial {
        background-color: #d1ecf1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 20px 0;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with .env configuration
if 'pipeline' not in st.session_state:
    # Load from environment variables
    config = {
        'spacy_model': os.getenv('SPACY_MODEL', 'en_core_web_sm'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5'),
        'llm_model': os.getenv('LLM_MODEL', 'qwen-2.5-32b'),
        'collection_name': os.getenv('COLLECTION_NAME', 'fact_checker'),
        'vector_db_path': os.getenv('VECTOR_DB_PATH', './data/vector_db')
    }
    
    with st.spinner("🚀 Initializing Fact-Checking System... This may take a minute on first run."):
        try:
            st.info(f"📊 Loading: {config['embedding_model']} + {config['llm_model']}")
            
            st.session_state.pipeline = FactCheckingPipeline(config)
            st.session_state.pipeline.build_fact_base(max_statements=50)
            st.success("✅ System initialized successfully!")
        except Exception as e:
            st.error(f"❌ Initialization error: {str(e)}")
            st.info("💡 Make sure you have set GROQ_API_KEY in your .env file")
            st.stop()
    
    st.session_state.history = []
    st.session_state.feedback = {}

# Header
st.markdown('<p class="main-header">✅ LLM-Powered Fact Checker</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced RAG System for Claim Verification | Artikate Studio Assignment</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### Retrieval Settings")
    top_k = st.slider("Number of evidence pieces", 3, 10, 5, 
                     help="More evidence = more context but slower")
    
    st.markdown("### Model Information")
    st.info("""
    **Tech Stack:**
    - 🧠 Claim Extraction: spaCy NER
    - 📊 Embeddings: BGE-Large (1024 dim)
    - 🗄️ Vector Store: Qdrant (local)
    - 🤖 LLM: Qwen 2.5 32B (FREE)
    """)
    
    st.markdown("---")
    st.header("📊 Session Statistics")
    
    total_checks = len(st.session_state.history)
    st.metric("Total Checks", total_checks)
    
    if st.session_state.history:
        verdicts = [h.get('verdict', 'Unknown') for h in st.session_state.history]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ True", verdicts.count("True"))
            st.metric("❌ False", verdicts.count("False"))
        with col2:
            st.metric("🤷 Unverifiable", verdicts.count("Unverifiable"))
            st.metric("⚖️ Partial", verdicts.count("Partially True"))
        
        avg_confidence = sum([h.get('confidence', 0) for h in st.session_state.history]) / len(st.session_state.history)
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.feedback = {}
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("Built for Artikate Studio")
    st.markdown("Powered by Groq (FREE API)")

# Main content
col_main, col_history = st.columns([2, 1])

with col_main:
    st.header("🔍 Enter Claim to Verify")
    
    # Sample claims
    st.markdown("### 💡 Try These Examples")
    examples = [
        "The Indian government has announced free electricity to all farmers starting July 2025.",
        "PM-KISAN provides Rs. 6000 per year to farmer families in three equal installments.",
        "India launched its first manned space mission in 2024.",
    ]
    
    # Initialize session state for input text
    if 'input_claim' not in st.session_state:
        st.session_state.input_claim = ""
    
    example_cols = st.columns(3)
    
    for idx, example in enumerate(examples):
        with example_cols[idx]:
            if st.button(f"Example {idx+1}", key=f"ex_{idx}", use_container_width=True):
                st.session_state.input_claim = example
                st.rerun()
    
    st.markdown("### ✍️ Or Enter Your Own Claim")
    input_text = st.text_area(
        "Claim or statement:",
        value=st.session_state.input_claim,
        height=120,
        placeholder="Enter a claim to fact-check...",
        help="Enter any statement you want to verify against PIB government data",
        key="claim_input_box"
    )
    
    col_btn1, col_btn2 = st.columns([3, 1])
    
    with col_btn1:
        check_button = st.button("🚀 Check Fact", type="primary", use_container_width=True)
    
    with col_btn2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.input_claim = ""
            if 'current_result' in st.session_state:
                del st.session_state.current_result
            st.rerun()
    
    # Process claim - FIXED HANDLER
    if check_button:
        if not input_text or not input_text.strip():
            st.warning("⚠️ Please enter a claim to verify!")
        else:
            with st.spinner("🔍 Analyzing claim... This may take a few seconds."):
                try:
                    result = st.session_state.pipeline.check_claim(input_text, top_k=top_k)
                    
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                        st.info("💡 The system couldn't find relevant evidence. Try a claim related to government schemes.")
                    else:
                        st.session_state.history.insert(0, result)
                        st.session_state.current_result = result
                        st.success("✅ Analysis complete!")
                        st.rerun()  # Refresh to show results
                        
                except Exception as e:
                    st.error(f"❌ Error during fact-checking: {str(e)}")
                    with st.expander("🐛 Debug Info"):
                        st.exception(e)
                    logger.error(f"Check claim error: {e}", exc_info=True)

# Display results
if hasattr(st.session_state, 'current_result') and st.session_state.current_result:
    result = st.session_state.current_result
    
    st.markdown("---")
    st.header("📋 Verification Results")
    
    # Verdict display
    verdict = result.get('verdict', 'Unknown')
    confidence = result.get('confidence', 0.0)
    
    verdict_map = {
        "True": ("verdict-true", "✅"),
        "False": ("verdict-false", "❌"),
        "Unverifiable": ("verdict-unverifiable", "🤷‍♂️"),
        "Partially True": ("verdict-partial", "⚖️")
    }
    
    verdict_class, emoji = verdict_map.get(verdict, ("verdict-unverifiable", "❓"))
    
    st.markdown(f"""
    <div class="{verdict_class}">
        <h2>{emoji} Verdict: {verdict}</h2>
        <p><strong>Confidence Score:</strong> {confidence:.1%}</p>
        <p><strong>Processing Time:</strong> {result.get('processing_time', 0)}s</p>
        <p><strong>Model:</strong> {result.get('model', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Reasoning
    st.subheader("🧠 Chain-of-Thought Reasoning")
    reasoning = result.get('reasoning', 'No reasoning available')
    st.markdown(f"> {reasoning}")
    
    # Evidence
    st.subheader("📚 Supporting Evidence")
    st.markdown("*Retrieved from PIB (Press Information Bureau) Government Data*")
    
    evidence_list = result.get('evidence', [])
    evidence_scores = result.get('evidence_scores', [])
    
    if evidence_list and evidence_scores:
        for i, (evidence, score) in enumerate(zip(evidence_list, evidence_scores)):
            with st.expander(f"📄 Evidence {i+1} (Relevance: {score:.1%})", expanded=(i==0)):
                st.write(evidence)
                st.progress(score)
    else:
        st.info("No evidence retrieved. This claim may be unverifiable with current data.")
    
    # Extracted information
    with st.expander("🔎 Detailed Extraction Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Extracted Claims:**")
            claims = result.get('extracted_claims', [])
            for idx, claim in enumerate(claims, 1):
                st.markdown(f"{idx}. {claim}")
        
        with col2:
            st.markdown("**Named Entities:**")
            entities = result.get('entities', [])
            if entities:
                entity_text = ", ".join([f"{e['text']} ({e['label']})" for e in entities[:10]])
                st.markdown(entity_text)
            else:
                st.markdown("*No entities found*")
    
    # Token usage
    tokens = result.get('tokens_used', 0)
    if tokens > 0:
        st.caption(f"🔢 Tokens Used: {tokens} (FREE with Groq API)")
    
    # Feedback section
    st.markdown("---")
    st.subheader("💭 Was this helpful?")
    
    feedback_cols = st.columns(4)
    
    with feedback_cols[0]:
        if st.button("👍 Helpful", key="fb_helpful"):
            st.session_state.feedback[result.get('claim', '')] = "helpful"
            st.success("Thanks for your feedback!")
    
    with feedback_cols[1]:
        if st.button("👎 Not Helpful", key="fb_not_helpful"):
            st.session_state.feedback[result.get('claim', '')] = "not_helpful"
            st.info("We'll improve our system!")
    
    with feedback_cols[2]:
        if st.button("⚠️ Report Issue", key="fb_issue"):
            st.warning("Issue reported. Thank you!")
    
    with feedback_cols[3]:
        json_str = json.dumps(result, indent=2, default=str)
        st.download_button(
            label="📥 Export JSON",
            data=json_str,
            file_name=f"fact_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# History panel
with col_history:
    st.header("📜 Recent Checks")
    
    if st.session_state.history:
        for i, hist in enumerate(st.session_state.history[:5]):
            verdict = hist.get('verdict', 'Unknown')
            verdict_emoji = {
                "True": "✅",
                "False": "❌",
                "Unverifiable": "🤷‍♂️",
                "Partially True": "⚖️"
            }.get(verdict, "❓")
            
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{verdict_emoji} {verdict}</strong><br>
                    <small><em>{hist.get('claim', '')[:80]}...</em></small><br>
                    <small>Confidence: {hist.get('confidence', 0):.0%} | {hist.get('processing_time', 0)}s</small>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("View Details", key=f"view_{i}"):
                    st.session_state.current_result = hist
                    st.rerun()
    else:
        st.info("No fact checks yet. Enter a claim to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>LLM-Powered Fact Checker</strong> | Artikate Studio Assignment</p>
    <p>Built with Qdrant + Groq + Sentence-Transformers | 100% FREE Stack</p>
    <p>Data Source: <a href="https://www.pib.gov.in/ViewRss.aspx" target="_blank">PIB India RSS Feed</a></p>
</div>
""", unsafe_allow_html=True)
