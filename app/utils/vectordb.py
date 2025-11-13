import logging
from lancedb import connect
from typing import Optional
from pathlib import Path

# Get the absolute path to the database directory
# This file is in app/utils/, so we go up one level to app/, then to utils/lancedb_data
_current_dir = Path(__file__).parent.resolve()  # Use resolve() to get absolute path
db_path = str(_current_dir / "lancedb_data")

# Set up logger
logger = logging.getLogger(__name__)

# Lazy-loaded model cache - only loaded on first use
_model: Optional[object] = None


def _get_model():
    """Lazy load the embedding model only when needed (not on module import)."""
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2' (first use)")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SentenceTransformer model loaded successfully")
    else:
        logger.debug("Using cached SentenceTransformer model")
    return _model


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
    logger.info(f"Fetching vocabulary from vector DB: query='{query}', level={level}, n={n}")
    
    if level not in {"A1", "A2", "B1", "B2"}:
        logger.error(f"Invalid level provided: {level}")
        raise ValueError("Invalid level. Must be 'A1', 'A2', 'B1', or 'B2'.")
    
    dbName = level + "_MINIMAL_vocabulary"
    
    # Verify the database directory exists
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        error_msg = f"LanceDB directory not found at: {db_path} (absolute: {db_path_obj.absolute()})"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # List available tables for debugging
    available_tables = [d.name for d in db_path_obj.iterdir() if d.is_dir() and d.name.endswith('.lance')]
    logger.info(f"Available LanceDB table directories: {available_tables}")
    
    logger.debug(f"Connecting to LanceDB at path: {db_path} (absolute: {db_path_obj.absolute()})")
    db = connect(db_path)
    
    # Try to list tables using LanceDB API if available
    lancedb_table_names = []
    try:
        # Try different methods to list tables
        if hasattr(db, 'table_names'):
            lancedb_table_names = db.table_names()
            logger.info(f"LanceDB reports available tables: {lancedb_table_names}")
        elif hasattr(db, 'list_tables'):
            lancedb_table_names = db.list_tables()
            logger.info(f"LanceDB reports available tables: {lancedb_table_names}")
    except Exception as e:
        logger.debug(f"Could not list tables via LanceDB API: {e}")
    
    # If LanceDB API provided table names, use those; otherwise use directory names
    if lancedb_table_names:
        logger.info(f"Using LanceDB API table names: {lancedb_table_names}")
    else:
        logger.info(f"LanceDB API did not provide table names, using directory scan: {available_tables}")
    
    logger.info(f"Attempting to open table for dbName: {dbName}")
    table = None
    last_error = None
    
    # Build list of table name variants to try
    # Priority: 1) Exact match from LanceDB API (authoritative), 2) dbName without extension, 3) dbName with .lance extension
    table_name_variants = []
    
    # If LanceDB API provided table names, prioritize exact matches from the API
    if lancedb_table_names:
        # Check for exact match in API table names (these are authoritative)
        if dbName in lancedb_table_names:
            table_name_variants.append(dbName)
            logger.info(f"Found exact match in LanceDB API: '{dbName}' - will try this first")
        else:
            # Try to find a match (case-insensitive or partial)
            for lancedb_name in lancedb_table_names:
                if lancedb_name.lower() == dbName.lower() or dbName in lancedb_name:
                    table_name_variants.append(lancedb_name)
                    logger.info(f"Found match in LanceDB API: '{lancedb_name}'")
                    break
    
    # Add standard variants as fallbacks (try these if API names didn't work or weren't available)
    # Note: LanceDB API typically reports names without .lance extension
    if dbName not in table_name_variants:
        table_name_variants.append(dbName)  # Without .lance extension (most common case)
    
    # Add .lance variant as last resort (for directory-based access)
    table_name_with_extension = dbName + ".lance"
    if table_name_with_extension not in table_name_variants:
        table_name_variants.append(table_name_with_extension)
    
    # Remove duplicates while preserving order
    seen = set()
    table_name_variants = [v for v in table_name_variants if v not in seen and not seen.add(v)]
    
    logger.info(f"Will try table name variants in order: {table_name_variants}")
    
    for variant in table_name_variants:
        try:
            logger.debug(f"Trying table name variant: '{variant}'")
            table = db.open_table(variant)
            logger.info(f"Successfully opened table: '{variant}'")
            break
        except (ValueError, FileNotFoundError, Exception) as e:
            last_error = e
            logger.debug(f"Failed to open table '{variant}': {type(e).__name__}: {e}")
            continue
    
    if table is None:
        # Provide helpful error message with available tables
        error_msg_parts = [
            f"Could not open table '{dbName}' or '{table_name_with_extension}'.",
            f"Last error: {last_error}.",
            f"Available table directories: {available_tables}.",
        ]
        if lancedb_table_names:
            error_msg_parts.append(f"LanceDB API reports tables: {lancedb_table_names}.")
        error_msg_parts.append(f"Database path: {db_path} (absolute: {db_path_obj.absolute()})")
        error_msg = " ".join(error_msg_parts)
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Lazy load model and encode query (model is cached after first use)
    logger.debug("Encoding query using embedding model")
    model = _get_model()
    # Use convert_to_numpy=True for faster encoding and direct compatibility with LanceDB
    query_vector = model.encode(query, convert_to_numpy=True)
    logger.debug(f"Query encoded, vector shape: {query_vector.shape}")
    
    logger.debug(f"Searching table with limit={n}")
    results = table.search(query_vector).limit(n).to_pandas()
    logger.info(f"Search returned {len(results)} results")

    # only take german_term & english_translation
    results = results[['german_term', 'english_translation']]
    logger.debug(f"Results columns: {results.columns.tolist()}")
    logger.debug(f"Results preview:\n{results}")
    
    # Extract vocabulary words from results (adjust column name based on your schema)
    # Assuming results have a column with vocabulary words
    if not results.empty:
        # Try common column names for vocabulary
        vocab_column = None
        for col in ['word', 'vocab', 'text', 'term', 'vocabulary']:
            if col in results.columns:
                vocab_column = col
                break
        
        if vocab_column:
            vocab_list = results[vocab_column].head(n).tolist()
            logger.info(f"Extracted {len(vocab_list)} vocabulary items from column '{vocab_column}'")
            return vocab_list
        else:
            # Return first column if no standard name found
            vocab_list = results.iloc[:, 0].head(n).tolist()
            logger.warning(f"No standard vocab column found, using first column. Extracted {len(vocab_list)} items")
            return vocab_list
    
    logger.warning("No results returned from vector database search")
    return []


# if __name__ == "__main__":
#     results = fetch_vocab_from_vector_db("food", "B2")
#     for result in results:
#         print(result)