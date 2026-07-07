import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from config import settings
from services.embedding import VectorStorageService
from services.llm import GenerationService
from api.routes import router

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Safely manages application lifetime setup and teardown stages."""
    print("Initializing RAG Infrastructure systems...", flush=True)
    try:
        # Load heavy model on startup once; attach directly to application shared context state
        app_instance.state.vector_service = VectorStorageService(settings.EMBEDDING_MODEL_NAME)
        app_instance.state.llm_service = GenerationService()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to prepare state pipelines. Exiting. Detail: {e}", file=sys.stderr)
        app_instance.state.vector_service = None
        app_instance.state.llm_service = None

    yield
    # Cleanup operations go here if needed
    print("Shutting down RAG Infrastructure systems...", flush=True)

app = FastAPI(
    title="Hackathon Q&A Engine API",
    description="Production-grade decoupled architectural pipeline for handling on-demand document RAG requests.",
    version="2.0.0",
    lifespan=lifespan
)

# Root health check
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Hackathon Q&A Engine is running!"}

# Attach routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)