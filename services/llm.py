import httpx
from typing import List
from config import settings

class GenerationService:
    def __init__(self):
        # groq's chat completion endpoint
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    async def get_answer(self, question: str, evidence_context: List[str]) -> str:
        """Dispatches an async generation prompt containing text extracts to Groq."""
        if not settings.GROQ_API_KEY:
            return "Configuration Error: GROQ_API_KEY missing from environment."

        context_payload = "\n\n".join(evidence_context)
        
        # Split into System and User messages for better instruction following
        system_prompt = (
            "You are a precise AI assistant extracting information from policy documents. "
            "Answer the user's question based *only* on the provided text. "
            "Be direct and concise. If the answer isn't in the text, strictly reply that the information is not available."
        )
        
        user_prompt = f"**Policy Text:**\n---\n{context_payload}\n---\n\n**Question:** {question}"

        payload = {
            "model": settings.GROQ_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0 # Keep this at 0 for RAG so it doesn't hallucinate
        }
        
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECS) as client:
            try:
                response = await client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                # Groq returns data in the standard OpenAI format
                return data['choices'][0]['message']['content'].strip()
            except Exception as e:
                return f"Error: Could not retrieve a synchronized context extraction answer. Details: {str(e)}"