import re
import fitz  # PyMuPDF
import httpx
from fastapi import HTTPException
from typing import List
from config import settings

class DocumentProcessor:
    @staticmethod
    async def download_pdf(url: str) -> bytes:
        """Asynchronously fetches document binary stream to maximize throughput."""
        async with httpx.AsyncClient(timeout=settings.DOWNLOAD_TIMEOUT_SECS) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=f"Remote host returned an error while downloading file: {e}")
            except httpx.RequestError as e:
                raise HTTPException(status_code=400, detail=f"Network transport failure trying to reach document source: {e}")

    @staticmethod
    def extract_and_chunk_text(pdf_bytes: bytes, min_chunk_size: int = 30) -> List[str]:
        """Parses structural layout of document and normalizes text chunks."""
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                full_text = "".join(page.get_text() for page in document)
            
            # Smart paragraph separation splitting on double spacing
            paragraphs = re.split(r'\n\s*\n', full_text)
            chunks = [
                re.sub(r'\s+', ' ', para).strip() 
                for para in paragraphs 
                if len(para.strip()) > min_chunk_size
            ]
            return chunks
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Structural analysis or text processing of the PDF failed: {e}")