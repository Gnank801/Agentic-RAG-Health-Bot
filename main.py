"""
RAG Health Agent — FastAPI Backend

Endpoints for data ingestion and health queries.
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from core.ingestion import ingest_patient_data, load_sample_data, ingest_from_csv
from core.rag_engine import get_rag_engine

load_dotenv()


# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    patient_id: Optional[str] = None
    chat_history: Optional[str] = ""


class QueryResponse(BaseModel):
    response: str
    sources: list
    patient_id: Optional[str]
    documents_retrieved: int


class IngestRequest(BaseModel):
    patients: list[dict]


class IngestResponse(BaseModel):
    message: str
    documents_ingested: int


class CsvIngestRequest(BaseModel):
    data_dir: str
    max_patients: int = 500


class PatientInfo(BaseModel):
    id: str
    name: str


# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: Load sample data if needed
    print("🏥 RAG Health Agent starting up...")
    
    # Check if API keys are configured
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  WARNING: GOOGLE_API_KEY not set!")
    if not os.getenv("PINECONE_API_KEY"):
        print("⚠️  WARNING: PINECONE_API_KEY not set!")
    
    yield
    
    # Shutdown
    print("👋 RAG Health Agent shutting down...")


# Create FastAPI app
app = FastAPI(
    title="RAG Health Agent",
    description="Agentic RAG chatbot for patient health summaries",
    version="1.0.0",
    lifespan=lifespan
)


# API Endpoints
@app.get("/")
async def root():
    """Serve the main dashboard."""
    return FileResponse("index.html")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rag-health-agent"}


@app.get("/api/patients", response_model=list[PatientInfo])
async def get_patients(search: str = "", limit: int = 100):
    """
    Get list of all patients in the system.
    
    - **search**: Optional search string to filter patients by name
    - **limit**: Maximum number of patients to return (default: 100)
    """
    try:
        engine = get_rag_engine()
        patients = engine.get_all_patients(search_query=search, limit=limit)
        return patients
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patients/{patient_id}")
async def get_patient_details(patient_id: str):
    """Get full details for a specific patient."""
    try:
        engine = get_rag_engine()
        details = engine.get_patient_details(patient_id)
        if "error" in details:
            raise HTTPException(status_code=404, detail=details["error"])
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_health_data(request: QueryRequest):
    """
    Query the health RAG system.
    
    - **question**: Natural language question about patient health
    - **patient_id**: Optional patient ID for filtered search
    """
    try:
        engine = get_rag_engine()
        result = engine.query(
            question=request.question,
            patient_id=request.patient_id,
            chat_history=request.chat_history
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_data(request: IngestRequest):
    """
    Ingest new patient data into the system.
    
    - **patients**: List of patient record dictionaries
    """
    try:
        count = ingest_patient_data(request.patients)
        return IngestResponse(
            message="Data ingested successfully",
            documents_ingested=count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/sample", response_model=IngestResponse)
async def ingest_sample_data():
    """Load and ingest sample patient data for demo purposes."""
    try:
        sample_data = load_sample_data()
        count = ingest_patient_data(sample_data)
        return IngestResponse(
            message="Sample data ingested successfully",
            documents_ingested=count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/csv", response_model=IngestResponse)
async def ingest_csv_data(request: CsvIngestRequest):
    """
    Load Synthea CSV data and ingest it.
    
    - **data_dir**: Path to directory containing Synthea CSV files
    - **max_patients**: Maximum number of patients to load (default: 500)
    """
    try:
        count = ingest_from_csv(request.data_dir, request.max_patients)
        return IngestResponse(
            message=f"CSV data ingested from {request.data_dir}",
            documents_ingested=count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files (for CSS/JS)
app.mount("/static", StaticFiles(directory="."), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
