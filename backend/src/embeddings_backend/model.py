from typing import List
from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    id: str
    text: str
    embedding: list

class EmbeddingPoint(BaseModel):
    id: str
    text: str
    x: float
    y: float
    embedding: list

class EmbeddingsResponse(BaseModel):
    count: int
    embeddings: List[EmbeddingPoint]

class SimilarityResponse(BaseModel):
    word1: str
    word2: str
    cosine_similarity: float
    id1: str
    id2: str