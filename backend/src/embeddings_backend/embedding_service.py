from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingService:
    _instance = None
    _model = None
    _model_name = None

    def __new__(cls, model_name=None):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name=None):
        if self._model is None and model_name is not None:
            print(f"Loading sentence transformer model: {model_name}...")
            self._model = SentenceTransformer(model_name)
            self._model_name = model_name
            print("Model loaded successfully!")

    def encode(self, text: str) -> np.ndarray:
        embedding = self._model.encode([text])
        return embedding[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self._model.encode(texts)
        return embeddings
