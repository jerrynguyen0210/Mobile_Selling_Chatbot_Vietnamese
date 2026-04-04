"""One-time ingestion script: hoanghamobile.csv → Qdrant cloud.

Usage (from the back-end/ directory):
    python -m app.database.init_data_once_time

Requirements (add to your virtualenv if missing):
    pip install qdrant-client sentence-transformers python-dotenv
"""

import csv
import html
import logging
import os
import re
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")
COLLECTION_NAME   = os.getenv("QDRANT_COLLECTION", "mobile_products")
EMBEDDING_MODEL   = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
BATCH_SIZE        = int(os.getenv("INGEST_BATCH_SIZE", "50"))

CSV_PATH = Path(__file__).parent / "hoanghamobile.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    """Strip HTML tags and decode entities."""
    return html.unescape(_HTML_TAG.sub(" ", text or "")).strip()


def build_text(row: dict) -> str:
    """Concatenate all meaningful fields into a single string for embedding."""
    parts = [
        row.get("title", ""),
        clean_html(row.get("product_promotion", "")),
        clean_html(row.get("product_specs", "")),
        row.get("current_price", ""),
        row.get("color_options", ""),
    ]
    return " | ".join(p for p in parts if p)


def mongo_id_to_uuid(mongo_id: str) -> str:
    """Derive a deterministic UUID from a MongoDB ObjectId hex string."""
    padded = mongo_id.ljust(32, "0")[:32]
    return str(uuid.UUID(padded))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    log.info("Reading CSV: %s", CSV_PATH)
    with CSV_PATH.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    log.info("Loaded %d rows", len(rows))

    if not rows:
        log.error("CSV is empty — nothing to ingest.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load embedding model
    # ------------------------------------------------------------------
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = model.get_sentence_embedding_dimension()
    log.info("Embedding dimension: %d", vector_size)

    # ------------------------------------------------------------------
    # 3. Connect to Qdrant
    # ------------------------------------------------------------------
    log.info("Connecting to Qdrant: %s", QDRANT_URL)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=60)

    # ------------------------------------------------------------------
    # 4. Recreate collection (required when changing embedding models)
    # ------------------------------------------------------------------
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        log.info("Deleting existing collection '%s' (model change requires rebuild)", COLLECTION_NAME)
        client.delete_collection(collection_name=COLLECTION_NAME)

    log.info("Creating collection '%s' (dim=%d)", COLLECTION_NAME, vector_size)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # ------------------------------------------------------------------
    # 5. Embed and upsert in batches
    # ------------------------------------------------------------------
    total_upserted = 0

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]

        texts = [build_text(r) for r in batch]
        vectors = model.encode(texts, show_progress_bar=False).tolist()

        points = []
        for row, vector in zip(batch, vectors):
            mongo_id = row.get("_id", "")
            point_id = mongo_id_to_uuid(mongo_id) if mongo_id else str(uuid.uuid4())

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "mongo_id":          mongo_id,
                        "url":               row.get("url", ""),
                        "title":             row.get("title", ""),
                        "product_promotion": clean_html(row.get("product_promotion", "")),
                        "product_specs":     clean_html(row.get("product_specs", "")),
                        "current_price":     row.get("current_price", ""),
                        "color_options":     row.get("color_options", ""),
                    },
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_upserted += len(points)
        log.info(
            "Upserted batch %d–%d  (%d total)",
            batch_start + 1,
            batch_start + len(batch),
            total_upserted,
        )

    log.info("Done. %d points upserted into '%s'.", total_upserted, COLLECTION_NAME)


if __name__ == "__main__":
    main()
