# 🎯 LLM-Powered Fact Checker with RAG

> **Artikate Studio Assignment** - Production-grade fact-checking system using Retrieval-Augmented Generation (RAG) to verify claims against trusted government data sources.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Free-orange.svg)](https://groq.com)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech)

## 🌟 Features

✅ **Advanced Claim Extraction** - Uses spaCy NER and zero-shot classification  
✅ **Qdrant Vector Store** - Open-source, production-ready, better than FAISS  
✅ **Groq LLM (FREE)** - Ultra-fast inference (800 tokens/sec) with Llama-3.1  
✅ **Real-time PIB Data** - Fetches latest government press releases  
✅ **Dual UI** - Professional Streamlit + Gradio interfaces  
✅ **Confidence Scoring** - Quantified verification confidence  
✅ **100% FREE** - No API costs, fully open-source stack  

## 🏗️ Architecture


### Tech Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| **Claim Extraction** | spaCy + Zero-shot BART | NER + checkworthiness assessment |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Fast, accurate, 384-dim |
| **Vector Store** | Qdrant (local) | Better than FAISS, built-in filtering |
| **LLM** | Groq Llama-3.1-8B | FREE, 800 tok/sec, 14.4K tok/min |
| **Data Source** | PIB RSS Feed | Official government press releases |
| **UI** | Streamlit + Gradio | Professional, user-friendly |

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Git
- Internet connection (for initial setup)

### 2. Installation

