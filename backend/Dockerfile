FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

RUN uv sync --frozen --no-cache

EXPOSE 8000

ENV MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

CMD ["uv", "run", "uvicorn", "embeddings_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
