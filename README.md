# 🏥 RAG Health Agent

An AI-powered **Agentic RAG** system for querying patient health records at scale. Built with **FastAPI**, **LangChain**, **Pinecone**, and **Groq (Llama 3.3 70B)**.

![Dashboard](chat_history.png)

## ✨ Features

- **Agentic RAG**: Retrieves and synthesizes data from **10,000+ synthetic patient records**.
- **Rich Medical Context**: Ingests Demographics, Diagnoses, Medications, Allergies, Immunizations, and Procedures.
- **Smart Retrieval**: Uses **Hybrid Search** + **BGE Reranking** for high precision.
- **Knowledge Base**: Also answers general medical questions (e.g., "Precautions for Diabetes?").
- **Premium UI**: Glassmorphism design with patient search, detail modals, and PDF export.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Pinecone API Key](https://www.pinecone.io/)
- [Groq API Key](https://groq.com/)
- [Google Gemini API Key](https://aistudio.google.com/) (for embeddings)

### Installation

1. **Clone & Install**
   ```bash
   git clone <repo-url>
   cd rag-health-agent
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Ingest Data (First Run)**
   ```bash
   # Ingest sample data (100+ rich records)
   python -c "from core.ingestion import ingest_from_csv; ingest_from_csv('data', max_patients=100)"
   ```

4. **Run Server**
   ```bash
   python main.py
   # Visit http://localhost:8000
   ```

## 🏗️ Architecture & Scale
See [ARCHITECTURE_REPORT.md](ARCHITECTURE_REPORT.md).

### 10k Record Scaling
This system is optimized for scale. To upgrade to 10k records:
1. Place 10k Synthea CSVs in `data/`.
2. Run: `python scripts/reindex_for_10k.py`
3. The script handles **Batch Ingestion** to prevent memory overflows.

## 🌍 Deployment

### Can I deploy this?
Yes! Since we use **serverless Pinecone** and **Groq Cloud**, the local server is very lightweight.

1.  **Backend**: Deploy `main.py` to **Render** (Free tier) or **Railway**.
    *   Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
2.  **Vector DB**: Pinecone Free Tier supports 1 Index (Starter).
    *   *Limit*: ~100k vectors. For 10k patients (approx 200k vectors), you may need the **Standard Plan** (~$70/mo).
3.  **LLM**: Groq usage is currently free (beta) but will eventually be paid.

**Production Note**: For real HIPAA compliance, you must use Pinecone Enterprise and sign BAA agreements with all AI providers.

## 📸 Walkthrough
See [WALKTHROUGH.md](WALKTHROUGH.md).

## 🛡️ License

MIT
