import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    REQUIRED_TOKEN: str = "2b44085e4132acd0163a6273bbaf087f1775c7a67d0b33632884ddb989983828"
    
    # --- GROK CONFIG ---
    GROK_API_KEY: str = os.getenv("GROK_API_KEY", "")
    GROK_MODEL_NAME: str = "grok-beta" # You can also use "grok-2-latest"
    
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # HTTP Client Configurations
    DOWNLOAD_TIMEOUT_SECS: float = 45.0
    LLM_TIMEOUT_SECS: float = 90.0

    class Config:
        env_file = ".env"

settings = Settings()
