import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    REQUIRED_TOKEN: str = "2b44085e4132acd0163a6273bbaf087f1775c7a67d0b33632884ddb989983828"
    
    # --- GROQ CONFIG ---
    GROQ_API_KEY: str = ""  # Pydantic automatically finds GROQ_API_KEY in your .env!
    GROQ_MODEL_NAME: str = "llama-3.1-8b-instant" 
    
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # HTTP Client Configurations
    DOWNLOAD_TIMEOUT_SECS: float = 45.0
    LLM_TIMEOUT_SECS: float = 90.0

    class Config:
        env_file = ".env"

settings = Settings()
