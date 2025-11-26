from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel
from typing import List

import numpy as np
from loguru import logger

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
logger.info(f"Using '{dim_reduction.method}' dimensionality reduction")


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

    # Split by semicolon and clean up whitespace
    texts = [t.strip() for t in input_data.text.strip().split(";") if t.strip()]

    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="No valid text found")

    # Compute embeddings (batch if multiple, single if one)
    if len(texts) == 1:
        embeddings = [embedding_service.encode(texts[0])]
    else:
        embeddings = embedding_service.encode_batch(texts)

    # Add each word as a separate entry
    entry_ids = []
    for text, embedding in zip(texts, embeddings):
        entry_id = embedding_store.add(text, embedding)
        entry_ids.append(entry_id)

    # Return info about the first word (for UI display)
    # but include count of total words added in the id field
    response_text = texts[0] if len(texts) == 1 else f"{texts[0]} (+ {len(texts)-1} more)"

    return EmbedResponse(
        id=entry_ids[0],
        text=response_text,
        embedding=embeddings[0].tolist()
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
