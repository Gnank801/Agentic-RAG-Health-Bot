"""
RAG Engine for Health Agent

Implements hybrid search, reranking, and Gemini-powered generation
with strict medical safety guardrails.
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore, PineconeRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()


# Rate limit retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds to wait between retries


# Structured output schema for health summaries
class HealthSummary(BaseModel):
    """Structured health summary response."""
    patient_name: str = Field(description="Name of the patient")
    summary: str = Field(description="Comprehensive health summary")
    diagnoses: list[str] = Field(description="List of current diagnoses")
    medications: list[str] = Field(description="List of current medications")
    recent_vitals: dict = Field(description="Most recent vital signs")
    recommendations: str = Field(description="Clinical recommendations or notes")
    sources: list[str] = Field(description="Source documents used for this summary")
    confidence: str = Field(description="Confidence level: HIGH, MEDIUM, or LOW")


# Medical safety prompt template
MEDICAL_PROMPT_TEMPLATE = """You are a Medical Research Assistant helping healthcare professionals review patient records.

**STRICT SAFETY RULES:**
1. ONLY use information from the provided CONTEXT. Never invent or assume medical information.
2. If information is NOT in the context, explicitly state: "Not found in available records."
3. NEVER provide medical advice or treatment recommendations beyond what's in the records.
4. Always cite which source document contains each piece of information.
5. Flag any concerning findings that may need immediate attention.

**PRIVACY RULES:**
- Do not expose raw addresses, SSNs, or insurance IDs in your response.
- Focus on clinically relevant information only.

**RESPONSE FORMAT:**
Organize your response as:
1. **Patient Overview**: Name, age, basic demographics
2. **Current Diagnoses**: List all known conditions
3. **Medications**: Current prescriptions with dosages
4. **Recent Vitals**: Latest vital signs
5. **Lab Results**: Relevant test results
6. **Clinical Notes**: Important observations
7. **Data Confidence**: State if the records appear complete or if data is missing

---

**CONVERSATION HISTORY:**
{chat_history}

---

**CONTEXT (Retrieved Patient Records):**
{context}

---

**QUESTION:** {question}

**RESPONSE:**"""


class RAGEngine:
    """
    Agentic RAG Engine with hybrid search and reranking.
    """
    
    def __init__(self, index_name: Optional[str] = None):
        """Initialize the RAG engine."""
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "health-agent-index")
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize LLM (Groq Llama 3.3 70B - much higher rate limits!)
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,  # Deterministic for medical accuracy
            max_tokens=2048
        )
        
        # Initialize vector store
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        # Initialize reranker (BGE model for precision)
        self.reranker = PineconeRerank(
            model="bge-reranker-v2-m3",
            top_n=5  # Return top 5 most relevant after reranking
        )
        
        # Build the prompt
        self.prompt = ChatPromptTemplate.from_template(MEDICAL_PROMPT_TEMPLATE)
    
    def get_retriever(self, patient_id: Optional[str] = None, k: int = 20):
        """
        Build a contextual compression retriever with optional patient filtering.
        
        Args:
            patient_id: If provided, only retrieve docs for this patient OR general knowledge
            k: Number of initial documents to retrieve before reranking
        """
        search_kwargs = {"k": k}
        
        # Add metadata filter
        if patient_id:
            # Retrieve Patient Data OR General Medical Knowledge
            # Note: This assumes newer Pinecone SDK supporting $or
            search_kwargs["filter"] = {
                "$or": [
                    {"patient_id": {"$eq": patient_id}},
                    {"type": {"$eq": "medical_knowledge"}}
                ]
            }
        
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )
        
        # Wrap with reranker for precision
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=base_retriever
        )
        
        return compression_retriever
    
    def format_docs(self, docs) -> str:
        """Format retrieved documents for the prompt."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            patient = doc.metadata.get("patient_name", "Unknown")
            formatted.append(
                f"[Source {i}: {patient} - {source}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)
    
    def query(
        self,
        question: str,
        patient_id: Optional[str] = None,
        chat_history: str = ""
    ) -> dict:
        """
        Query the RAG engine for a health summary.
        
        Args:
            question: The user's question
            patient_id: Optional patient ID for filtered retrieval
            chat_history: Formatted string of previous conversation
        
        Returns:
            Dict with response, sources, and metadata
        """
        # Get appropriate retriever
        retriever = self.get_retriever(patient_id=patient_id)
        
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(question)
        
        if not retrieved_docs:
            return {
                "response": "No matching patient records found. Please verify the patient name or ID.",
                "sources": [],
                "patient_id": patient_id,
                "documents_retrieved": 0
            }
        
        # Format context
        context = self.format_docs(retrieved_docs)
        
        # Build and run the chain with retry logic for rate limits
        chain = self.prompt | self.llm | StrOutputParser()
        
        response = None
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                response = chain.invoke({
                    "context": context,
                    "question": question,
                    "chat_history": chat_history or "None."
                })
                break  # Success, exit retry loop
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (attempt + 1)  # Exponential backoff
                        print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 2}/{MAX_RETRIES}...")
                        time.sleep(wait_time)
                        continue
                # For other errors, don't retry
                break
        
        if response is None:
            return {
                "response": f"Error: API rate limit exceeded. Please wait a minute and try again. Details: {str(last_error)[:200]}",
                "sources": [],
                "patient_id": patient_id,
                "documents_retrieved": len(retrieved_docs)
            }
        
        # Extract source metadata
        sources = [
            {
                "patient_name": doc.metadata.get("patient_name"),
                "patient_id": doc.metadata.get("patient_id"),
                "type": doc.metadata.get("type"),
                "source": doc.metadata.get("source")
            }
            for doc in retrieved_docs
        ]
        
        return {
            "response": response,
            "sources": sources,
            "patient_id": patient_id,
            "documents_retrieved": len(retrieved_docs)
        }
    
    def get_all_patients(self, search_query: str = "", limit: int = 200) -> list[dict]:
        """
        Get a list of all patient names/IDs in the index.
        
        Args:
            search_query: Optional search string to filter patients
            limit: Maximum number of patients to return
        """
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(self.index_name)
        
        try:
            # Query with a generic patient query to retrieve diverse results
            query_text = search_query if search_query else "patient medical record demographics"
            query_embedding = self.embeddings.embed_query(query_text)
            
            # Fetch more to get diverse patients
            results = index.query(
                vector=query_embedding,
                top_k=min(limit * 3, 1000),  # Fetch more to filter
                include_metadata=True,
                filter={"type": {"$eq": "health_summary"}}  # Only patient records, not knowledge base
            )
            
            # Extract unique patients
            patients = {}
            for match in results.get('matches', []):
                metadata = match.get('metadata', {})
                patient_id = metadata.get('patient_id')
                patient_name = metadata.get('patient_name', 'Unknown')
                
                # Skip knowledge base entries
                if metadata.get('type') == 'medical_knowledge':
                    continue
                
                if patient_id and patient_id not in patients:
                    # Apply search filter if provided
                    if search_query:
                        if search_query.lower() not in patient_name.lower():
                            continue
                    
                    patients[patient_id] = {
                        "id": patient_id,
                        "name": patient_name
                    }
                    
                    if len(patients) >= limit:
                        break
            
            return list(patients.values())
            
        except Exception as e:
            print(f"Error fetching patients: {e}")
            # Fallback to sample data
            return [
                {"id": "P001", "name": "John Doe"},
                {"id": "P002", "name": "Jane Smith"},
                {"id": "P003", "name": "Robert Johnson"}
            ]

    def get_patient_details(self, patient_id: str) -> dict:
        """
        Fetch all documents for a specific patient and combine them.
        """
        # Retrieve docs without reranking for speed
        docs = self.vectorstore.similarity_search(
            query="patient history",
            k=10,
            filter={"patient_id": patient_id}
        )
        
        if not docs:
            return {"error": "No records found for this patient."}
            
        return {
            "patient_id": patient_id,
            "patient_name": docs[0].metadata.get("patient_name", "Unknown"),
            "records": [d.page_content for d in docs],
            "metadata": docs[0].metadata
        }


# Singleton instance
_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create the RAG engine singleton."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
