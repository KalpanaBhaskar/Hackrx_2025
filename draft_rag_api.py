# final_api.py

import os
import numpy as np
import faiss
import re
import fitz  # PyMuPDF
import requests
import asyncio
from fastapi import FastAPI, HTTPException, Body, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List
import httpx

# --- API Security and Configuration ---
# This sets up the security check for the "Bearer" token.
security = HTTPBearer()
# This is the required token for your hackathon.
REQUIRED_TOKEN = "2b44085e4132acd0163a6273bbaf087f1775c7a67d0b33632884ddb989983828"

# --- Pydantic Models for the Hackathon API Contract ---
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document.")
    questions: List[str] = Field(..., description="A list of questions to answer based on the document.")

class RunResponse(BaseModel):
    answers: List[str]

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Hackathon Q&A Engine API",
    description="Handles the full process from PDF URL to final answers for a list of questions.",
    version="1.0.0"
)

# --- Load AI Model on Startup (Done Once) ---
try:
    retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    # In a real server, this would be logged to a file.
    # For the hackathon, if this fails, the server won't be able to process requests.
    retrieval_model = None

# --- Helper Functions (Internal Logic) ---

def download_pdf(url: str) -> bytes:
    """Downloads the PDF from the provided URL."""
    try:
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {e}")

def extract_and_chunk_text(pdf_bytes: bytes) -> List[str]:
    """Extracts text and splits it into meaningful paragraphs/clauses."""
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "".join(page.get_text() for page in document)
        document.close()
        paragraphs = re.split(r'\n\s*\n', full_text)
        chunks = [re.sub(r'\s+', ' ', para).strip() for para in paragraphs if len(para.strip()) > 30]
        return chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF content: {e}")

# --- Core Logic Functions (The RAG Pipeline) ---

def retrieve_relevant_clauses(text_chunks: List[str], question: str, model: SentenceTransformer, index: faiss.Index, top_k: int = 5) -> List[str]:
    """Finds the most relevant clauses for a SINGLE question."""
    query_embedding = model.encode([question])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [text_chunks[i] for i in indices[0]]

async def get_final_answer_from_gemini(question: str, evidence: List[str], api_key: str) -> str:
    """Gets a definitive answer for a SINGLE question using Gemini."""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    context_clauses = "\n\n".join(evidence)
    
    prompt = f"""
    Based *only* on the following text from a policy document, answer the user's question.
    Be direct and concise. If the answer isn't in the text, state that the information is not available.

    **Policy Text:**
    ---
    {context_clauses}
    ---

    **Question:**
    {question}

    **Answer:**
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        except (httpx.RequestError, httpx.HTTPStatusError, KeyError, IndexError):
            return "Error: Could not get a valid response from the AI model."

# --- Security Dependency ---
def verify_token(auth: HTTPAuthorizationCredentials = Security(security)):
    """This function checks if the provided token is correct."""
    if auth.scheme != "Bearer" or auth.credentials != REQUIRED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token")
    return auth.credentials

# --- The Main API Endpoint for the Hackathon ---
@app.post("/hackrx/run", response_model=RunResponse)
async def run_submission(
    request: RunRequest = Body(...),
    token: str = Depends(verify_token) # This line enforces the security check.
):
    if not retrieval_model:
        raise HTTPException(status_code=503, detail="Service Unavailable: Core AI model not loaded.")
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Server Configuration Error: GEMINI_API_KEY not set.")

    # Step 1: Prepare the Knowledge Base
    pdf_content = download_pdf(request.documents)
    all_document_chunks = extract_and_chunk_text(pdf_content)
    if not all_document_chunks:
        raise HTTPException(status_code=500, detail="Failed to extract any usable text from the document.")

    corpus_embeddings = retrieval_model.encode(all_document_chunks)
    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    index.add(np.array(corpus_embeddings, dtype=np.float32))

    # Step 2: Answer Each Question in Parallel for maximum speed
    tasks = []
    for question in request.questions:
        evidence = retrieve_relevant_clauses(all_document_chunks, question, retrieval_model, index)
        task = get_final_answer_from_gemini(question, evidence, gemini_api_key)
        tasks.append(task)

    final_answers = await asyncio.gather(*tasks)

    return RunResponse(answers=final_answers)