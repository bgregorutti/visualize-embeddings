from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel
from typing import List

import numpy as np

from .embedding_service import EmbeddingService
from .embedding_store import EmbeddingStore
from .dimensionality_reduction import DimensionalityReduction

app = FastAPI(title="Word Embeddings Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_service = EmbeddingService(model_name=os.environ["MODEL_NAME"])
embedding_store = EmbeddingStore()
dim_reduction = DimensionalityReduction(method=os.environ.get("DIMENSIONALITY_REDUCTION", "umap"))


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


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}


@app.post("/embed", response_model=EmbedResponse)
async def create_embedding(input_data: TextInput):
    if not input_data.text or not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    text = input_data.text.strip()
    embedding = embedding_service.encode(text)
    entry_id = embedding_store.add(text, embedding)

    return EmbedResponse(
        id=entry_id,
        text=text,
        embedding=embedding.tolist()
    )


@app.get("/embeddings", response_model=EmbeddingsResponse)
async def get_all_embeddings():
    entries = embedding_store.get_all()

    if len(entries) == 0:
        return EmbeddingsResponse(count=0, embeddings=[])

    embeddings_matrix = np.array([entry.embedding for entry in entries])
    coords_2d = dim_reduction.reduce(embeddings_matrix, n_components=2)

    embedding_points = []
    for entry, coords in zip(entries, coords_2d):
        embedding_points.append(EmbeddingPoint(
            id=entry.id,
            text=entry.text,
            x=float(coords[0]),
            y=float(coords[1]),
            embedding=entry.embedding.tolist()
        ))

    return EmbeddingsResponse(
        count=len(entries),
        embeddings=embedding_points
    )


@app.delete("/embeddings")
async def clear_embeddings():
    embedding_store.clear()
    return {"status": "success", "message": "All embeddings cleared"}
