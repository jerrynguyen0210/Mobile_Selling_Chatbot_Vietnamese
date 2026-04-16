# Mobile Selling Chatbot (Vietnamese)

AI-powered chatbot for mobile phone sales in the Vietnamese market.

Users can:
- Browse phones by brand or price range
- Ask feature-based questions (battery, camera, performance)
- Get Vietnamese recommendations backed by a RAG pipeline

The system retrieves relevant products from Qdrant, then generates final responses with Claude.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser)                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────┐
│               Frontend (Streamlit · port 8501)                  │
│  • Chat UI with user/bot bubbles                                │
│  • Sidebar with brand categories + price slider                 │
│  • Quick-reply suggestions                                      │
│  • Local mock fallback if backend is offline                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ POST /chat (JSON)
┌──────────────────────────────▼──────────────────────────────────┐
│               Backend API (FastAPI · port 8000)                 │
│  1) Embed user query (sentence-transformers)                    │
│  2) Retrieve top-k products from Qdrant                         │
│  3) Build prompt with retrieved context                         │
│  4) Generate response via Claude API                            │
│  5) Cache session state in Redis                                │
└──────────────┬───────────────┬───────────────┬─────────────────┘
               │               │               │
    ┌──────────────────┐   ┌───────────────┐
    │   Qdrant Cloud   │   │     Redis     │
    │ (managed remote) │   │ (port 6379)   │
    │ Product vectors  │   │ Session cache │
    └──────────────────┘   └───────────────┘
```

### Key Design Decisions

| Concern | Choice | Rationale |
|---|---|---|
| LLM | Claude (Anthropic) | Strong Vietnamese language quality |
| Vector store | Qdrant | Fast ANN search, easy local + cloud usage |
| Embeddings | `intfloat/multilingual-e5-base` | High multilingual retrieval quality (768-d) |
| Cache | Redis | Session storage with TTL support |
| Frontend | Streamlit | Rapid prototyping without frontend JS |
| API | FastAPI | Async-first with built-in OpenAPI docs |

## Quick Start

### Prerequisites

- Docker + Docker Compose v2
- [Anthropic API key](https://console.anthropic.com/)

### 1. Clone and configure

```bash
git clone <repo-url>
cd Mobile_Selling_Chatbot_Vietnamese

make env
# Then edit .env and set ANTHROPIC_API_KEY (and other secrets as needed)
```

### 2. Start the stack

```bash
make run
# or:
make run-detach
```

Services started: Redis, Qdrant, FastAPI backend, and Streamlit frontend.

### 3. Ingest product data

```bash
make ingest
# Reads PRODUCT_DATA_PATH (default: data/products.json)
# and upserts embeddings into Qdrant
```

### 4. Open the app

| Service | URL |
|---|---|
| Chatbot UI | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Qdrant dashboard | Your Qdrant Cloud console URL |

## Local Development (Without Docker App Services)

```bash
# Python 3.11+
make install

# Start infrastructure only (e.g., Redis/Qdrant)
make run-infra

# In separate terminals:
make run-backend   # FastAPI on :8000
make run-frontend  # Streamlit on :8501
```

## API Reference

### `POST /chat`

Sends a user message and returns a chatbot response.

Request body:

```json
{
  "message": "Cho tôi xem điện thoại dưới 10 triệu pin trâu",
  "session_id": "session_1718000000"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | User message in Vietnamese |
| `session_id` | string | yes | Unique conversation session ID |

Response (`200`):

```json
{
  "reply": "🔋 Dưới 10 triệu với pin trâu, bạn có thể cân nhắc:\n1. Samsung Galaxy A35 – pin 5.000mAh, giá 7.990.000đ\n2. Xiaomi Redmi Note 13 – pin 5.000mAh, giá 5.990.000đ\n\nBạn muốn biết thêm thông tin về mẫu nào?",
  "session_id": "session_1718000000"
}
```

| Field | Type | Description |
|---|---|---|
| `reply` | string | Vietnamese bot reply (may include Markdown) |
| `session_id` | string | Echoed session ID |

Error responses:

| Status | Meaning |
|---|---|
| `422` | Validation error in request body |
| `500` | Internal server error (check backend logs) |

### `GET /health`

Returns health status:

```json
{ "status": "ok" }
```

## Project Structure

```text
Mobile_Selling_Chatbot_Vietnamese/
├── back-end/
│   ├── app/
│   │   ├── main.py           # FastAPI app + route registration
│   │   ├── routers/
│   │   │   └── chat.py       # POST /chat endpoint
│   │   ├── services/
│   │   │   ├── rag.py        # Retrieval-Augmented Generation pipeline
│   │   │   ├── embedder.py   # sentence-transformers wrapper
│   │   │   └── llm.py        # Anthropic Claude client
│   │   ├── db/
│   │   │   ├── session.py    # SQLAlchemy async engine
│   │   │   └── models.py     # ORM models (Conversation, Message, Product)
│   │   └── core/
│   │       ├── config.py     # pydantic-settings configuration
│   │       └── deps.py       # FastAPI dependency injection
│   ├── migrations/           # Alembic migration scripts
│   ├── scripts/
│   │   └── ingest.py         # Product catalogue -> Qdrant ingestion script
│   ├── Dockerfile
│   └── tests/
│       ├── unit/
│       └── integration/
├── front-end/
│   ├── chatting_bot.py       # Streamlit chat application
│   └── Dockerfile
├── data/
│   └── products.json         # Source-of-truth product catalogue
├── tests/                    # Root-level test entry point
├── pyproject.toml            # Dependencies + ruff + pytest config
├── docker-compose.yml        # Full local dev stack
├── Makefile                  # Developer commands
├── .env.example              # Environment variable template
└── README.md
```

## Common Commands

```bash
make help              # list all commands

# Development
make run               # start full stack via docker-compose
make run-backend       # backend only (local)
make run-frontend      # frontend only (local)
make stop              # stop all containers

# Data
make ingest            # ingest products into Qdrant

# Quality
make test              # run pytest with coverage
make lint              # ruff check
make lint-fix          # ruff check --fix
make format            # ruff format
make check             # lint + typecheck

# Utilities
make logs              # tail container logs
make clean             # remove build/cache artifacts
```

## Environment Variables

See [.env.example](.env.example) for the full list and descriptions.

Most important before startup:

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key used by Claude |
| `REDIS_PASSWORD` | Redis password (also update `REDIS_URL`) |
| `QDRANT_API_KEY` | Qdrant service API key |
| `SECRET_KEY` | 64-character hex string for token signing |

## Contributing

1. Create a feature branch from `main`.
2. Run `make install`.
3. Implement your changes and add tests.
4. Run `make check && make test`.
5. Open a pull request.

## Docker Image Workflow (Optional)

Tag and push images:

```bash
docker tag mobile_selling_chatbot_vietnamese-backend jerrynguyen0210/chatbot-backend:latest
docker tag mobile_selling_chatbot_vietnamese-frontend jerrynguyen0210/chatbot-frontend:latest

docker push jerrynguyen0210/chatbot-backend:latest
docker push jerrynguyen0210/chatbot-frontend:latest
```

Pull images:

```bash
docker pull jerrynguyen0210/chatbot-backend:latest
docker pull jerrynguyen0210/chatbot-frontend:latest
```

Run with host-based backend URL:

```bash
docker run -d --name chatbot_backend \
  -p 8000:8000 \
  --env-file ./.env \
  jerrynguyen0210/chatbot-backend:latest

docker run -d --name chatbot_frontend \
  --network chatbot_net \
  -p 8601:8501 \
  -e BACKEND_URL=http://localhost:8000 \
  jerrynguyen0210/chatbot-frontend:latest
```

Run both containers on a shared Docker network:

```bash
docker network create chatbot_net

docker run -d --name chatbot_backend \
  --network chatbot_net \
  -p 8000:8000 \
  --env-file .env \
  jerrynguyen0210/chatbot-backend:latest

docker run -d --name chatbot_frontend \
  --network chatbot_net \
  -p 8501:8501 \
  -e BACKEND_URL=http://chatbot_backend:8000 \
  jerrynguyen0210/chatbot-frontend:latest
```

## License

MIT
