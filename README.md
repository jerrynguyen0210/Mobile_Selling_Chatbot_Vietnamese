# 📱 Mobile Selling Chatbot Vietnamese

An AI-powered mobile phone selling chatbot for the Vietnamese market. Customers can browse phones by brand or price range, ask feature-specific questions (battery, camera, performance), and receive product recommendations in Vietnamese — backed by a RAG pipeline that retrieves relevant products from a vector store before generating responses with Claude.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser)                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────┐
│               Frontend  (Streamlit · port 8501)                 │
│  • Chat UI with user/bot message bubbles                        │
│  • Sidebar: brand categories + price-range slider               │
│  • Quick-reply suggestions                                      │
│  • Falls back to local mock responses when backend is offline   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ POST /chat  (JSON)
┌──────────────────────────────▼──────────────────────────────────┐
│               Backend API  (FastAPI · port 8000)                │
│                                                                 │
│  POST /chat ──► 1. Embed user query (sentence-transformers)     │
│                 2. Retrieve top-k products  ──► Qdrant          │
│                 3. Build prompt with context                    │
│                 4. Generate reply  ──────────► Claude API       │
│                 5. Persist conversation ─────► PostgreSQL       │
│                 6. Cache session ────────────► Redis            │
└──────────────┬───────────────┬───────────────┬─────────────────┘
               │               │               │
    ┌──────────▼──┐   ┌────────▼──────┐  ┌────▼──────────┐
    │  PostgreSQL  │   │    Qdrant     │  │     Redis     │
    │  (port 5432) │   │  (port 6333)  │  │  (port 6379)  │
    │  Conversations│  │  Product      │  │  Session      │
    │  & products  │   │  embeddings   │  │  cache & TTL  │
    └─────────────┘   └───────────────┘  └───────────────┘
```

### Key design decisions

| Concern | Choice | Why |
|---|---|---|
| LLM | Claude (Anthropic) | Best-in-class Vietnamese language quality |
| Vector store | Qdrant | Fast ANN search, easy Docker deployment |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Good Vietnamese coverage, runs on CPU |
| Relational DB | PostgreSQL | Conversation history, product catalogue |
| Cache | Redis | Session state, TTL-based expiry |
| Frontend | Streamlit | Rapid iteration, no JS required |
| API | FastAPI | Async, auto-generated OpenAPI docs |

---

## Quick start

### Prerequisites

- Docker & Docker Compose v2
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone & configure

```bash
git clone <repo-url>
cd Mobile_Selling_Chatbot_Vietnamese

# Creates .env from the template
make env
# → open .env and set ANTHROPIC_API_KEY (and any other secrets you want to change)
```

### 2. Start all services

```bash
make run
# or in detached mode:
make run-detach
```

This starts: **PostgreSQL · Redis · Qdrant · Backend (FastAPI) · Frontend (Streamlit)**

### 3. Apply database migrations

```bash
make migrate
```

### 4. Ingest product catalogue

```bash
make ingest
# Reads PRODUCT_DATA_PATH (default: data/products.json) and upserts
# product embeddings into Qdrant.
```

### 5. Open the app

| Service | URL |
|---|---|
| Chatbot UI | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |

---

## Local development (without Docker)

```bash
# Install Python dependencies (requires Python 3.11+)
make install

# Start only the infrastructure containers
make run-infra

# Run backend + frontend in separate terminals
make run-backend   # terminal 1 — FastAPI on :8000
make run-frontend  # terminal 2 — Streamlit on :8501
```

---

## API reference

### `POST /chat`

Send a user message and receive a bot reply.

**Request body**

```json
{
  "message": "Cho tôi xem điện thoại dưới 10 triệu pin trâu",
  "session_id": "session_1718000000"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | User message in Vietnamese |
| `session_id` | string | yes | Unique identifier for the conversation session |

**Response (200)**

```json
{
  "reply": "🔋 Dưới 10 triệu với pin trâu, bạn có thể cân nhắc:\n1. Samsung Galaxy A35 – pin 5.000mAh, giá 7.990.000đ\n2. Xiaomi Redmi Note 13 – pin 5.000mAh, giá 5.990.000đ\n\nBạn muốn biết thêm thông tin về mẫu nào?",
  "session_id": "session_1718000000"
}
```

| Field | Type | Description |
|---|---|---|
| `reply` | string | Bot response in Vietnamese (may include Markdown) |
| `session_id` | string | Echo of the request session ID |

**Error responses**

| Status | Meaning |
|---|---|
| 422 | Validation error — check request body |
| 500 | Internal server error (check backend logs) |

---

### `GET /health`

Returns service health status.

```json
{ "status": "ok" }
```

---

## Project structure

```
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
│   │   └── ingest.py         # Product catalogue → Qdrant ingestion script
│   ├── Dockerfile
│   └── tests/
│       ├── unit/
│       └── integration/
├── front-end/
│   ├── chatting_bot.py       # Streamlit chat application
│   └── Dockerfile
├── data/
│   └── products.json         # Product catalogue (source of truth for ingestion)
├── tests/                    # Root-level test entry point
├── pyproject.toml            # Dependencies + ruff + pytest config
├── docker-compose.yml        # Full local dev stack
├── alembic.ini               # Alembic migration config
├── Makefile                  # Developer commands
├── .env.example              # Environment variable template
└── README.md
```

---

## Common commands

```bash
make help              # list all available commands

# Development
make run               # start everything via docker-compose
make run-backend       # FastAPI only (local)
make run-frontend      # Streamlit only (local)
make stop              # stop all containers

# Database
make migrate                          # apply pending migrations
make migrate-create MSG="add table"   # generate new migration
make migrate-down                     # roll back last migration

# Data
make ingest            # ingest products into Qdrant

# Quality
make test              # pytest with coverage
make lint              # ruff check
make lint-fix          # ruff check --fix
make format            # ruff format
make check             # lint + typecheck

# Utilities
make logs              # tail all container logs
make shell-db          # psql into Postgres
make clean             # remove cache / build artefacts
```

---

## Environment variables

See [.env.example](.env.example) for the full list with descriptions.

The most important ones to set before starting:

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `POSTGRES_PASSWORD` | Postgres password (also update `DATABASE_URL`) |
| `REDIS_PASSWORD` | Redis password (also update `REDIS_URL`) |
| `QDRANT_API_KEY` | Qdrant service API key |
| `SECRET_KEY` | 64-character hex string for token signing |

---

## Contributing

1. Create a feature branch from `main`
2. Run `make install` to set up the dev environment
3. Make your changes, add tests
4. Run `make check && make test` — all checks must pass
5. Open a pull request

---

## License

MIT
