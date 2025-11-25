from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            print("Loading sentence transformer model...")
            self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("Model loaded successfully!")

    def encode(self, text: str) -> np.ndarray:
        embedding = self._model.encode([text])
        return embedding[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self._model.encode(texts)
        return embeddings
