import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request
from schemas.rag_schemas import RunRequest, RunResponse
from api.dependencies import verify_access_token
from services.document import DocumentProcessor

router = APIRouter(prefix="/hackrx", tags=["RAG Execution"])

@router.post("/run", response_model=RunResponse)
async def run_submission(request: RunRequest, token: str = Depends(verify_access_token), req_app: Request = None):
    # Fetch vector and generation engine states bound globally on active application memory space
    vector_service = req_app.app.state.vector_service
    llm_service = req_app.app.state.llm_service

    if not vector_service:
        raise HTTPException(status_code=503, detail="Vector embedding framework state down or uninitialized.")

    # Phase 1: Download & Chunk Document
    pdf_data = await DocumentProcessor.download_pdf(str(request.documents))
    chunks = DocumentProcessor.extract_and_chunk_text(pdf_data)
    
    if not chunks:
        raise HTTPException(status_code=422, detail="Parsed file text is barren or too brief to build reference index states.")

    # Phase 2: Compute Embeddings & Construct In-Memory Vector Store
    # Note: Running synchronous blocking code (like model processing) in async routes should be handled with care. 
    # For small workloads, it is okay, but thread pools can be utilized here for scale.
    index, _ = vector_service.generate_index(chunks)

    # Phase 3: Parallelized Neighborhood Mapping & Generation Orchestration
    tasks = []
    for question in request.questions:
        evidence = vector_service.query_index(question=question, text_chunks=chunks, index=index)
        tasks.append(llm_service.get_answer(question, evidence))

    final_answers = await asyncio.gather(*tasks)
    return RunResponse(answers=final_answers)