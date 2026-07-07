import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class VectorStorageService:
    def __init__(self, model_name: str):
        # Heavy model loading encapsulated in service initialization
        self.model = SentenceTransformer(model_name)

    def generate_index(self, text_chunks: List[str]) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
        """Vectorizes textual database elements and registers them inside an index."""
        embeddings = self.model.encode(text_chunks, show_progress_bar=False)
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
        return index, embeddings_np

    def query_index(self, question: str, text_chunks: List[str], index: faiss.IndexFlatL2, top_k: int = 5) -> List[str]:
        """Finds closest neighborhood vector indices mapped back to text data chunks."""
        query_vector = self.model.encode([question])
        query_np = np.array(query_vector, dtype=np.float32)
        
        _, indices = index.search(query_np, top_k)
        # Avoid index errors if fewer chunks exist than top_k elements requested
        valid_indices = [i for i in indices[0] if i < len(text_chunks)]
        return [text_chunks[i] for i in valid_indices]
    
