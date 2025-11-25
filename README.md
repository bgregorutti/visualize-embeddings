# Word Embeddings Visualizer

An interactive web application for computing and visualizing word embeddings using transformer models.

## Features

- Compute embeddings for words and sentences using `sentence-transformers/all-MiniLM-L6-v2`
- Display raw embedding vectors (384 dimensions)
- Visualize embeddings in 2D using UMAP dimensionality reduction
- Interactive scatter plot showing semantic relationships
- Real-time updates as you add more inputs
- Clean, minimal UI with responsive design

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   Frontend      │         │    Backend       │
│   (nginx)       │────────▶│   (FastAPI)      │
│   - HTML/JS     │  HTTP   │   - Transformers │
│   - Chart.js    │         │   - UMAP         │
└─────────────────┘         └──────────────────┘
      :80                         :8000
```

### Backend
- **Framework**: FastAPI
- **Package Manager**: UV (modern Python package manager)
- **Model**: sentence-transformers/all-MiniLM-L6-v2 (384-dim embeddings)
- **Dimensionality Reduction**: UMAP
- **Storage**: In-memory (no database required)

### Frontend
- **Tech**: Vanilla JavaScript, Chart.js
- **Server**: nginx
- **Features**: Interactive form, real-time plot updates, embedding display

## Quick Start

### Prerequisites
- Docker
- Docker Compose

### Running the Application

1. Clone this repository
2. Navigate to the project directory
3. Start the application:

```bash
docker-compose up --build
```

4. Open your browser and go to `http://localhost`
5. Enter words or sentences to see their embeddings visualized!

The first startup will take a few minutes as it downloads the transformer model (~90MB).

### Stopping the Application

```bash
docker-compose down
```

## Usage

1. **Enter Text**: Type a word or sentence in the input box
2. **Compute**: Click "Compute Embedding" or press Enter
3. **View Results**:
   - See the embedding vector (first 10 dimensions shown)
   - Watch the 2D plot update with the new point
4. **Add More**: Keep adding words/sentences to see relationships

### Example Inputs to Try

Try these to see semantic clustering:
- "cat", "dog", "puppy", "kitten"
- "king", "queen", "prince", "princess"
- "happy", "sad", "joyful", "depressed"
- "Paris", "France", "London", "England"

## API Endpoints

### GET /health
Health check endpoint
```bash
curl http://localhost:8000/health
```

### POST /embed
Compute embedding for input text
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world"}'
```

Response:
```json
{
  "id": "uuid",
  "text": "hello world",
  "embedding": [0.123, -0.456, ...]
}
```

### GET /embeddings
Get all stored embeddings with 2D coordinates
```bash
curl http://localhost:8000/embeddings
```

Response:
```json
{
  "count": 3,
  "embeddings": [
    {
      "id": "uuid",
      "text": "hello world",
      "x": 1.23,
      "y": -0.45,
      "embedding": [...]
    }
  ]
}
```

## Development

### Running Backend Locally

The backend is now a UV-managed Python package:

```bash
cd backend
uv sync  # Install dependencies
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" uv run uvicorn embeddings_backend.main:app --reload
```

Backend will be available at `http://localhost:8000`

### Running Frontend Locally

Simply open `frontend/index.html` in a browser, or use a local server:

```bash
cd frontend
python -m http.server 8080
```

Frontend will be available at `http://localhost:8080`

**Note**: Update `API_BASE_URL` in `app.js` if running locally without Docker.

## Project Structure

```
visualize-embeddings/
├── backend/                     # UV-managed Python package
│   ├── src/
│   │   └── embeddings_backend/
│   │       ├── __init__.py
│   │       ├── main.py          # FastAPI app and endpoints
│   │       ├── embedding_service.py  # Transformer model wrapper
│   │       └── embedding_store.py    # In-memory storage
│   ├── pyproject.toml           # UV package configuration
│   ├── uv.lock                  # Locked dependencies
│   └── Dockerfile
├── frontend/
│   ├── index.html               # UI layout and styling
│   ├── app.js                   # Frontend logic and API calls
│   └── Dockerfile
├── docker-compose.yml           # Container orchestration
└── README.md
```

## Technical Details

### UMAP Dimensionality Reduction
- Uses UMAP for projecting 384-dim embeddings to 2D
- Preserves local and global structure
- Random state fixed for reproducibility
- Handles edge cases (1 or 2 points)

### In-Memory Storage
- Embeddings stored as numpy arrays
- Singleton pattern for service instances
- No persistence (data lost on restart)
- Suitable for demo/exploration purposes

### CORS Configuration
- Configured to allow all origins
- Suitable for development
- Consider restricting in production

## Limitations

- Data is not persisted (in-memory only)
- No authentication or user management
- Single-user experience
- Model cannot be changed without code modification

## Future Enhancements

- Add database for persistence
- Support multiple models
- Download embeddings as CSV
- Similarity search functionality
- Clustering visualization
- Multi-user support with sessions

## License

MIT

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Embeddings from [sentence-transformers](https://www.sbert.net/)
- Visualization with [Chart.js](https://www.chartjs.org/)
- Dimensionality reduction using [UMAP](https://umap-learn.readthedocs.io/)
