import logging
import os
from timescale_vector import client
from typing import Optional
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)

# Database connection configuration
postgres_password = os.getenv("POSTGRES_PASSWORD")
if not postgres_password:
    raise ValueError("POSTGRES_PASSWORD environment variable is not set")

postgres_db = os.getenv("POSTGRES_DB")
if not postgres_db:
    raise ValueError("POSTGRES_DB environment variable is not set")

# Use service name for Docker Compose, or 127.0.0.1 for local development
postgres_host = os.getenv("POSTGRES_HOST", "timescaledb")
postgres_port = os.getenv("POSTGRES_PORT", "5432")

service_url = f"postgresql://postgres:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

# Vector client configuration
# all-MiniLM-L6-v2 produces 384-dimensional embeddings
embedding_dimensions = 384
time_partition_interval = timedelta(days=7)

# Lazy-loaded model cache - only loaded on first use
_model: Optional[object] = None

# Cache for vector clients per level
_vec_clients: dict[str, client.Sync] = {}


def _get_model():
    """Lazy load the embedding model only when needed (not on module import)."""
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2' (first use)")
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer model loaded successfully")
    else:
        logger.debug("Using cached SentenceTransformer model")
    return _model


def get_embedding(query_text: str):
    """
    Generate embedding for the query text.

    Args:
        query_text: The text to generate embedding for

    Returns:
        numpy array of embedding vector
    """
    model = _get_model()
    # Generate embedding - convert to numpy array
    embedding = model.encode(query_text, convert_to_numpy=True)
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    return embedding


def _get_vector_client(level: str) -> client.Sync:
    """
    Get or create a vector client for the specified level.

    Args:
        level: Language level ("A1", "A2", "B1", or "B2")

    Returns:
        Vector client instance for the level
    """
    if level not in {"A1", "A2", "B1", "B2"}:
        raise ValueError("Invalid level. Must be 'A1', 'A2', 'B1', or 'B2'.")

    # Check cache first
    if level in _vec_clients:
        return _vec_clients[level]

    # Table name follows pattern: "a1_minimal.csv", "a2_minimal.csv", etc.
    table_name = f"{level.lower()}_minimal.csv"

    logger.info(f"Initializing vector client for level {level}, table: {table_name}")

    # Initialize vector client
    vec_client = client.Sync(
        service_url,
        table_name,
        embedding_dimensions,
        time_partition_interval=time_partition_interval,
    )

    # Cache the client
    _vec_clients[level] = vec_client

    logger.info(f"Vector client initialized for level {level}")
    return vec_client


def fetch_vocab_from_vector_db(query: str, level: str = "A1", n: int = 10) -> list[str]:
    """
    Fetch vocabulary words from the vector database filtered by 'level'.

    Args:
        query (str): Search query for vocab.
        level (str): Language level to filter by ("A1", "A2", "B1", or "B2").
        n (int): Maximum number of vocab items to return.

    Returns:
        list[str]: List of vocabulary words matching the query and level.
    """
    logger.info(
        f"Fetching vocabulary from vector DB: query='{query}', level={level}, n={n}"
    )

    if level not in {"A1", "A2", "B1", "B2"}:
        logger.error(f"Invalid level provided: {level}")
        raise ValueError("Invalid level. Must be 'A1', 'A2', 'B1', or 'B2'.")

    # Get vector client for this level
    vec_client = _get_vector_client(level)

    # Generate embedding for the query
    logger.debug("Encoding query using embedding model")
    query_embedding = get_embedding(query)
    logger.debug(f"Query encoded, vector shape: {query_embedding.shape}")

    # Perform the search
    logger.debug(f"Searching table with limit={n}")
    try:
        results = vec_client.search(query_embedding, limit=n)
        logger.info(f"Search returned {len(results)} results")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise

    # Extract vocabulary words from results
    # Results format: (id, metadata, content, embedding, distance)
    vocab_list = []
    for result in results:
        result_id, metadata, content, embedding, distance = result

        # Extract vocabulary from content or metadata
        # The content field should contain the vocabulary word/term
        if content:
            # If content is a string, use it directly
            if isinstance(content, str):
                vocab_list.append(content)
            # If content is a dict, try to extract relevant fields
            elif isinstance(content, dict):
                # Try common field names
                for field in [
                    "german_term",
                    "word",
                    "vocab",
                    "text",
                    "term",
                    "vocabulary",
                    "content",
                ]:
                    if field in content:
                        vocab_list.append(str(content[field]))
                        break
                else:
                    # If no standard field found, use the first value
                    if content:
                        vocab_list.append(str(list(content.values())[0]))
        elif metadata:
            # Fallback to metadata if content is not available
            if isinstance(metadata, dict):
                for field in ["german_term", "word", "vocab", "text", "term"]:
                    if field in metadata:
                        vocab_list.append(str(metadata[field]))
                        break

    logger.info(f"Extracted {len(vocab_list)} vocabulary items")
    logger.debug(f"Vocabulary items: {vocab_list}")

    return vocab_list
