import os
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from app.rag.retriever import ProductRetriever
from app.config import Settings
import asyncio

# --- CONFIGURATION ---
# Load back-end/.env before any Settings object is instantiated so that
# QDRANT_URL, QDRANT_API_KEY, etc. are present in the environment.
load_dotenv(Path(__file__).parents[1] / "back-end" / ".env", override=True)

QDRANT_URL = os.getenv("QDRANT_URL")
API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")

# Define a sample query vector (dimensionality must match your collection)
# Replace this with your actual embedding logic
DEBUG_VECTOR = [0.1] * 1536  # Example for OpenAI embeddings (1536 dims)

def debug_qdrant_context():
    # 1. Initialize Client
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)
    
    print(f"--- Starting Debug for Collection: {COLLECTION_NAME} ---")
    
    try:
        # 2. Check Connection & Collection Info
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"[SUCCESS] Connected to Cloud.")
        print(f"  - Status: {collection_info.status}")
        print(f"  - Points Count: {collection_info.points_count}")
        print(f"  - Vectors Config: {collection_info.config.params.vectors}")
        
        # 3. Check Payload Indexes
        # Searches are slow/incorrect if the fields you filter on aren't indexed.
        payload_schema = collection_info.payload_schema
        print(f"  - Indexed Fields: {list(payload_schema.keys())}")

        # 4. Test Retrieval (Get a random point to see payload structure)
        # This helps verify if your search 'context' matches actual data keys.
        random_points = client.scroll(collection_name=COLLECTION_NAME, limit=1)[0]
        if random_points:
            print(f"[INFO] Sample Payload Schema from ID {random_points[0].id}:")
            print(f"  {random_points[0].payload}")
        else:
            print("[WARNING] Collection is empty. Search will return no results.")

        # 5. Perform a Dry-Run Search with Debugging Params
        print("\n--- Running Debug Search ---")
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=DEBUG_VECTOR,
            limit=3,
            with_payload=True,
            # Common Pitfall: Ensure your Filter keys match the Payload keys found in Step 4
            query_filter=None, 
            search_params=models.SearchParams(
                hnsw_ef=128,  # Increase for better accuracy during debug
                exact=False   # Set to True if you want to bypass HNSW for "Ground Truth"
            )
        )

        if not search_result:
            print("[ALERT] Search returned 0 results. Check your filters or vector dimensions.")
        else:
            for i, hit in enumerate(search_result):
                print(f"Hit {i+1}: ID={hit.id}, Score={hit.score:.4f}")

    except UnexpectedResponse as e:
        print(f"[ERROR] API Response Error: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected Error: {e}")

# Replace this with your actual embedding (e.g., from OpenAI or SentenceTransformers)
# If your collection uses "Named Vectors", specify the name here.
EMBEDDING_MODEL   = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
model = SentenceTransformer(EMBEDDING_MODEL)  # or whatever model you used at indexing time
# QUERY_VECTOR = model.encode("Samsung Galaxy S24").tolist()
VECTOR_NAME = None  # Change to "text" or "default" if applicable
# QUERY_VECTOR = [0.01, 0.05, -0.12] # ... actual embedding values ...
QUERY_VECTOR = model.encode("iphone pro max").tolist()

def debug_vector_search(text_query: str):
    QUERY_VECTOR = model.encode(text_query).tolist()
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)
    
    print(f"--- Vector Search Debug: {COLLECTION_NAME} ---")
    
    try:
        # 1. Inspect Vector Configuration
        col = client.get_collection(COLLECTION_NAME)
        v_config = col.config.params.vectors
        
        # Determine expected dimension and distance metric
        if isinstance(v_config, dict):
            target_config = v_config.get(VECTOR_NAME) if VECTOR_NAME else next(iter(v_config.values()))
        else:
            target_config = v_config

        exp_dim = target_config.size
        distance = target_config.distance
        
        print(f"[CONFIG] Expected Dimension: {exp_dim}")
        print(f"[CONFIG] Distance Metric: {distance}")

        # 2. Validate Input Vector
        input_dim = len(QUERY_VECTOR)
        if input_dim != exp_dim:
            print(f"[ERROR] Dimension Mismatch! Input is {input_dim}, but Collection expects {exp_dim}.")
            return

        # 3. Check for "Zero Vectors" (Common embedding failure)
        vector_norm = np.linalg.norm(QUERY_VECTOR)
        if vector_norm == 0:
            print("[ERROR] Input vector is a 'Zero Vector'. Cosine similarity will fail.")
        
        # 4. Execute Search with Score Threshold
        print(f"\n[ACTION] Executing search (Limit: 5)...")
        
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=QUERY_VECTOR,
            limit=5,
            with_payload=True,
            search_params=models.SearchParams(
                exact=True,
                hnsw_ef=128
            )
        ).points

        # 5. Analyze Results
        if not results:
            print("[ALERT] No matches found. Possible reasons:")
            print("  - Score threshold too high (if used)")
            print("  - Data not yet indexed or collection is empty")
        else:
            print(f"[SUCCESS] Found {len(results)} matches:")
            for hit in results:
                # Log score to see if matches are 'tight' or 'loose'
                status = "Strong Match" if hit.score > 0.8 else "Weak Match"
                print(f" -> ID: {hit.id} | Score: {hit.score:.4f} ({status})")
                print(hit.payload.get("title"))

    except Exception as e:
        print(f"[CRITICAL] Debug failed: {str(e)}")
    
async def debug_retriever(text_query: str):
    # This function can be used to test the retriever logic in isolation
    # by simulating a search with known inputs and expected outputs.
    settings = Settings(retrieval_score_threshold=0.3)
    debug_retriever = ProductRetriever(settings)
    docs = await debug_retriever.search(text_query)
    for doc in docs:
        print(f"Product ID: {doc.product_id}, Name: {doc.product_name}, Score: {doc.score:.4f}")
        print(f"Snippet: {doc.snippet}\n")
      

if __name__ == "__main__":
    asyncio.run(debug_retriever("iphone pro max"))
    # print("\n-------------\n")
    # debug_vector_search("nokia 3210 4g")
