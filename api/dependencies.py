from fastapi import Security, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from config import settings

security = HTTPBearer()

def verify_access_token(auth: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verifies standard platform execution access headers."""
    if auth.scheme != "Bearer" or auth.credentials != settings.REQUIRED_TOKEN:
        raise HTTPException(
            status_code=401, 
            detail="Invalid authorization token signature or bad payload type."
        )
    return auth.credentials