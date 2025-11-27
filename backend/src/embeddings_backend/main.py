from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
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

# Initialize services
embedding_service = EmbeddingService(model_name=os.environ["MODEL_NAME"])
embedding_store = EmbeddingStore()
dim_reduction = DimensionalityReduction(method=os.environ.get("DIMENSIONALITY_REDUCTION", "umap"))
logger.info(f"Using '{dim_reduction.method}' dimensionality reduction")


def load_training_words(file_path: str) -> List[str]:
    """Load training words from text file (semicolon-separated)."""
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"Training file not found: {file_path}")
        return []

    with open(path, 'r') as f:
        content = f.read().strip()

    # Parse semicolon-separated words
    words = [w.strip() for w in content.split(';') if w.strip()]
    logger.info(f"Loaded {len(words)} training words from {file_path}")
    return words


def pretrain_dimensionality_reduction():
    """
    Pre-train the dimensionality reduction model using words from text.txt.
    This ensures that all subsequently added words are projected into a fixed 2D space.
    """
    # Load training words from text.txt
    training_file = Path(__file__).parent.parent.parent / "text.txt"
    training_words = load_training_words(str(training_file))

    if len(training_words) < 3:
        logger.warning("Not enough training words to pre-train reducer. Need at least 3 words.")
        return

    # Compute embeddings for training words
    logger.info(f"Computing embeddings for {len(training_words)} training words...")
    training_embeddings = embedding_service.encode_batch(training_words)

    # Fit the dimensionality reduction model
    logger.info(f"Fitting {dim_reduction.method.upper()} on training embeddings...")
    dim_reduction.fit(training_embeddings, n_components=2)
    logger.info(f"Dimensionality reduction pre-trained successfully!")

    # OPTIONAL: Uncomment the following lines to add training words to the plot
    # This will pre-populate the visualization with the training words
    # for word, embedding in zip(training_words, training_embeddings):
    #     embedding_store.add(word, embedding)
    # logger.info(f"Added {len(training_words)} training words to the embedding store")


# Pre-train the reducer at startup
pretrain_dimensionality_reduction()
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
    coords_2d = dim_reduction.transform(embeddings_matrix)

    embedding_points = []
    for entry, coords in zip(entries, coords_2d):
        # Get text in embedding_points
        current = [item.text for item in embedding_points]
        if entry.text not in current:
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
