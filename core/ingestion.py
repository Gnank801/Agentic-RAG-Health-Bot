"""
Data Ingestion Pipeline for RAG Health Agent

Handles loading, chunking, embedding, and upserting health data to Pinecone.
Supports both sample data and large CSV datasets (e.g., Synthea).
"""

import os
import csv
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initialize Google Generative AI Embeddings."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def init_pinecone_index(index_name: str) -> None:
    """Create Pinecone index if it doesn't exist."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,  # text-embedding-004 dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created new index: {index_name}")
    else:
        print(f"Index '{index_name}' already exists")


def reset_index(index_name: Optional[str] = None) -> None:
    """
    DANGER: Delete all vectors in the index to start fresh.
    Required when replacing data (e.g. upgrading to 10k records).
    """
    if index_name is None:
        index_name = os.getenv("PINECONE_INDEX_NAME", "health-agent-index")
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    try:
        print(f"Deleting all vectors in '{index_name}'...")
        index.delete(delete_all=True)
        print("Index cleared successfully.")
    except Exception as e:
        print(f"Error clearing index: {e}")


def create_patient_documents(patient_data: dict) -> List[Document]:
    """
    Convert patient data dict to LangChain Documents with metadata.
    
    Each patient record becomes a document with patient_id metadata
    for safe filtered retrieval.
    """
    documents = []
    
    patient_id = patient_data.get("patient_id", "unknown")
    patient_name = patient_data.get("name", "Unknown Patient")
    
    # Create comprehensive patient summary document
    content_parts = []
    
    if "demographics" in patient_data:
        demo = patient_data["demographics"]
        content_parts.append(
            f"Patient Demographics: {patient_name}, "
            f"Age: {demo.get('age', 'N/A')}, "
            f"Gender: {demo.get('gender', 'N/A')}, "
            f"Blood Type: {demo.get('blood_type', 'N/A')}"
        )
    
    # CRITICAL: Diagnoses/Conditions
    if "diagnoses" in patient_data and patient_data["diagnoses"]:
        diagnoses = ", ".join(patient_data["diagnoses"][:10])  # Top 10
        content_parts.append(f"Current Diagnoses: {diagnoses}")
    
    # CRITICAL: Medications
    if "medications" in patient_data and patient_data["medications"]:
        medications = ", ".join(patient_data["medications"][:10])  # Top 10
        content_parts.append(f"Current Medications: {medications}")
    
    if "allergies" in patient_data and patient_data["allergies"]:
        allergies = ", ".join(patient_data["allergies"])
        content_parts.append(f"Known Allergies: {allergies}")
    
    if "procedures" in patient_data and patient_data["procedures"]:
        procedures = ", ".join(patient_data["procedures"][:5])  # Top 5
        content_parts.append(f"Recent Procedures: {procedures}")
    
    if "immunizations" in patient_data and patient_data["immunizations"]:
        immunizations = ", ".join(patient_data["immunizations"][:5])  # Top 5
        content_parts.append(f"Immunizations: {immunizations}")
    
    if "observations" in patient_data and patient_data["observations"]:
        observations = patient_data["observations"]
        # Get top 10 observations
        obs_items = list(observations.items())[:10]
        obs_str = "; ".join([f"{k}: {v}" for k, v in obs_items])
        content_parts.append(f"Recent Vitals/Labs: {obs_str}")

    if "notes" in patient_data and patient_data["notes"]:
        content_parts.append(f"Clinical Notes: {patient_data['notes']}")
    
    # Create the main document
    full_content = "\n\n".join(content_parts)
    
    doc = Document(
        page_content=full_content,
        metadata={
            "patient_id": patient_id,
            "patient_name": patient_name,
            "type": "health_summary",
            "source": "ingestion_pipeline"
        }
    )
    documents.append(doc)
    
    return documents


def create_knowledge_documents(data_dir: str) -> List[Document]:
    """
    Ingest general medical knowledge from generic CSVs.
    Target file: disease_knowledge.csv (simulated Kaggle dataset).
    """
    file_path = Path(data_dir) / "disease_knowledge.csv"
    if not file_path.exists():
        print(f"No knowledge base file found at {file_path}")
        return []

    print(f"Loading knowledge base from {file_path}...")
    documents = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = row.get("Disease", "Unknown Disease")
            if not disease or disease == "Unknown Disease":
                continue
            
            # Collect symptoms
            symptoms = []
            precautions = []
            
            for k, v in row.items():
                if not k or not v:
                    continue
                if "Symptom" in k:
                    symptoms.append(v.strip().replace("_", " "))
                elif k == "Precautions" or "Precaution" in k:
                    # Precautions column contains comma-separated values
                    for p in v.split(","):
                        if p.strip():
                            precautions.append(p.strip())
            
            # Build rich content
            content = (
                f"Disease: {disease}\n\n"
                f"Common Symptoms:\n- " + "\n- ".join(symptoms) + "\n\n"
                f"Recommended Precautions:\n- " + "\n- ".join(precautions) + "\n\n"
                "Source: General Medical Knowledge Base (Kaggle Disease-Symptom Dataset)"
            )
            
            doc = Document(
                page_content=content,
                metadata={
                    "type": "medical_knowledge",
                    "disease": disease,
                    "patient_name": f"[Knowledge] {disease}",  # For consistent source display
                    "source": "kaggle_disease_symptom_dataset"
                }
            )
            documents.append(doc)
            
    print(f"Created {len(documents)} knowledge base documents")
    return documents


def ingest_patient_data(
    patient_records: List[dict],
    index_name: Optional[str] = None
) -> int:
    """
    Ingest a list of patient records into Pinecone.
    
    Args:
        patient_records: List of patient data dictionaries
        index_name: Pinecone index name (defaults to env var)
    
    Returns:
        Number of documents ingested
    """
    if index_name is None:
        index_name = os.getenv("PINECONE_INDEX_NAME", "health-agent-index")
    
    # Initialize index
    init_pinecone_index(index_name)
    
    # Get embeddings model
    embeddings = get_embeddings()
    
    # Create vector store
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    
    # Process Knowledge Base FIRST (it's small)
    data_dir = "data" 
    kb_docs = create_knowledge_documents(data_dir)
    if kb_docs:
        print(f"Upserting {len(kb_docs)} knowledge base documents...")
        vectorstore.add_documents(kb_docs)
    
    # Process Patient Records in Batches
    BATCH_SIZE = 100
    total_ingested = len(kb_docs)
    
    if patient_records:
        print(f"Ingesting {len(patient_records)} patients in batches of {BATCH_SIZE}...")
        batch_docs = []
        
        for i, patient in enumerate(patient_records):
            docs = create_patient_documents(patient)
            batch_docs.extend(docs)
            
            # Upsert batch
            if len(batch_docs) >= BATCH_SIZE:
                vectorstore.add_documents(batch_docs)
                total_ingested += len(batch_docs)
                print(f"  Processed {i+1}/{len(patient_records)} patients ({total_ingested} docs total)...")
                batch_docs = []  # Reset batch
        
        # Upsert remaining
        if batch_docs:
            vectorstore.add_documents(batch_docs)
            total_ingested += len(batch_docs)
    
    print(f"Ingestion complete. Total documents: {total_ingested}")
    return total_ingested


def load_sample_data() -> List[dict]:
    """
    Load sample synthetic patient data.
    
    Returns sample patients for demo purposes.
    """
    return [
        {
            "patient_id": "P001",
            "name": "John Doe",
            "demographics": {
                "age": 45,
                "gender": "Male",
                "blood_type": "O+"
            },
            "diagnoses": ["Type 2 Diabetes", "Hypertension"],
            "medications": ["Metformin 500mg", "Lisinopril 10mg"],
            "vitals": {
                "blood_pressure": "130/85 mmHg",
                "heart_rate": "72 bpm",
                "temperature": "98.6°F"
            },
            "lab_results": {
                "HbA1c": "7.2%",
                "fasting_glucose": "126 mg/dL",
                "cholesterol": "210 mg/dL"
            },
            "notes": "Patient reports improved diet adherence. Follow-up in 3 months."
        },
        {
            "patient_id": "P002",
            "name": "Jane Smith",
            "demographics": {
                "age": 62,
                "gender": "Female",
                "blood_type": "A-"
            },
            "diagnoses": ["Atrial Fibrillation", "Osteoarthritis"],
            "medications": ["Warfarin 5mg", "Ibuprofen 400mg PRN"],
            "vitals": {
                "blood_pressure": "142/88 mmHg",
                "heart_rate": "88 bpm (irregular)",
                "temperature": "98.4°F"
            },
            "lab_results": {
                "INR": "2.5",
                "creatinine": "1.1 mg/dL",
                "potassium": "4.2 mEq/L"
            },
            "notes": "INR within therapeutic range. Continue current Warfarin dose."
        },
        {
            "patient_id": "P003",
            "name": "Robert Johnson",
            "demographics": {
                "age": 35,
                "gender": "Male",
                "blood_type": "B+"
            },
            "diagnoses": ["Asthma", "Seasonal Allergies"],
            "medications": ["Albuterol inhaler PRN", "Cetirizine 10mg"],
            "vitals": {
                "blood_pressure": "118/76 mmHg",
                "heart_rate": "68 bpm",
                "oxygen_saturation": "98%"
            },
            "lab_results": {
                "peak_flow": "450 L/min",
                "IgE": "180 IU/mL"
            },
            "notes": "Asthma well-controlled. Patient using inhaler 1-2x per week."
        }
    ]


def load_synthea_csv(data_dir: str, max_patients: int = 1000) -> List[dict]:
    """
    Load Synthea synthetic patient data from CSV files.
    
    Download Synthea data from: https://synthea.mitre.org/downloads
    Extract the CSV files to a 'data' folder.
    
    Expected files:
    - patients.csv (required)
    - conditions.csv (optional)
    - medications.csv (optional)
    - observations.csv (optional)
    - allergies.csv (optional)
    - procedures.csv (optional)
    - immunizations.csv (optional)
    
    Args:
        data_dir: Path to directory containing Synthea CSV files
        max_patients: Maximum number of patients to load
    
    Returns:
        List of patient dictionaries ready for ingestion
    """
    data_path = Path(data_dir)
    patients_file = data_path / "patients.csv"
    
    if not patients_file.exists():
        raise FileNotFoundError(
            f"patients.csv not found in {data_dir}. "
            "Download Synthea data from https://synthea.mitre.org/downloads"
        )
    
    # Load patients
    patients_data = {}
    print(f"Loading patients from {patients_file}...")
    
    with open(patients_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_patients:
                break
            
            patient_id = row.get('Id', row.get('PATIENT', f'P{i}'))
            first = row.get('FIRST', row.get('first', ''))
            last = row.get('LAST', row.get('last', ''))
            
            patients_data[patient_id] = {
                "patient_id": patient_id,
                "name": f"{first} {last}".strip() or f"Patient {patient_id[:8]}",
                "demographics": {
                    "gender": row.get('GENDER', row.get('gender', 'N/A')),
                    "birthdate": row.get('BIRTHDATE', row.get('birthdate', 'N/A')),
                    "city": row.get('CITY', row.get('city', 'N/A')),
                    "state": row.get('STATE', row.get('state', 'N/A')),
                    "race": row.get('RACE', row.get('race', 'N/A')),
                    "ethnicity": row.get('ETHNICITY', row.get('ethnicity', 'N/A'))
                },
                "diagnoses": [],
                "medications": [],
                "allergies": [],
                "procedures": [],
                "immunizations": [],
                "observations": {},
                "notes": ""
            }
    
    print(f"Loaded {len(patients_data)} patients")
    
    # Load conditions if available
    conditions_file = data_path / "conditions.csv"
    if conditions_file.exists():
        print(f"Loading conditions from {conditions_file}...")
        with open(conditions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row.get('PATIENT', row.get('patient', ''))
                if patient_id in patients_data:
                    condition = row.get('DESCRIPTION', row.get('description', ''))
                    if condition and condition not in patients_data[patient_id]["diagnoses"]:
                        patients_data[patient_id]["diagnoses"].append(condition)
    
    # Load medications if available
    medications_file = data_path / "medications.csv"
    if medications_file.exists():
        print(f"Loading medications from {medications_file}...")
        with open(medications_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row.get('PATIENT', row.get('patient', ''))
                if patient_id in patients_data:
                    med = row.get('DESCRIPTION', row.get('description', ''))
                    if med and med not in patients_data[patient_id]["medications"]:
                        patients_data[patient_id]["medications"].append(med)

    # Load allergies if available
    allergies_file = data_path / "allergies.csv"
    if allergies_file.exists():
        print(f"Loading allergies from {allergies_file}...")
        with open(allergies_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row.get('PATIENT', row.get('patient', ''))
                if patient_id in patients_data:
                    allergy = row.get('DESCRIPTION', row.get('description', ''))
                    if allergy and allergy not in patients_data[patient_id]["allergies"]:
                        patients_data[patient_id]["allergies"].append(allergy)

    # Load procedures if available
    procedures_file = data_path / "procedures.csv"
    if procedures_file.exists():
        print(f"Loading procedures from {procedures_file}...")
        with open(procedures_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row.get('PATIENT', row.get('patient', ''))
                if patient_id in patients_data:
                    proc = row.get('DESCRIPTION', row.get('description', ''))
                    if proc and proc not in patients_data[patient_id]["procedures"]:
                        patients_data[patient_id]["procedures"].append(proc)

    # Load immunizations if available
    immunizations_file = data_path / "immunizations.csv"
    if immunizations_file.exists():
        print(f"Loading immunizations from {immunizations_file}...")
        with open(immunizations_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row.get('PATIENT', row.get('patient', ''))
                if patient_id in patients_data:
                    imm = row.get('DESCRIPTION', row.get('description', ''))
                    if imm and imm not in patients_data[patient_id]["immunizations"]:
                        patients_data[patient_id]["immunizations"].append(imm)

    # Load observations if available
    observations_file = data_path / "observations.csv"
    if observations_file.exists():
        print(f"Loading observations from {observations_file}...")
        with open(observations_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row.get('PATIENT', row.get('patient', ''))
                if patient_id in patients_data:
                    desc = row.get('DESCRIPTION', row.get('description', ''))
                    val = row.get('VALUE', row.get('value', ''))
                    unit = row.get('UNITS', row.get('units', ''))
                    if desc and val:
                        patients_data[patient_id]["observations"][desc] = f"{val} {unit}".strip()
    
    # Convert to list
    result = list(patients_data.values())
    print(f"Prepared {len(result)} patient records with conditions and medications")
    
    return result


def ingest_from_csv(data_dir: str, max_patients: int = 500) -> int:
    """
    Convenience function to load Synthea CSV data and ingest it.
    
    Args:
        data_dir: Path to directory containing Synthea CSV files
        max_patients: Maximum number of patients to load
    
    Returns:
        Number of documents ingested
    """
    patients = load_synthea_csv(data_dir, max_patients)
    return ingest_patient_data(patients)
