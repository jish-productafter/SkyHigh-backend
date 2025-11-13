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
    # Ensure we use absolute path for connection
    abs_db_path = str(db_path_obj.absolute())
    db = connect(abs_db_path)
    
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
    
    # Determine the table name to use
    # Priority: Use exact match from LanceDB API if available (most reliable)
    table_name_to_use = None
    
    if lancedb_table_names:
        # Check for exact match in API table names (these are authoritative)
        if dbName in lancedb_table_names:
            table_name_to_use = dbName
            logger.info(f"Using exact match from LanceDB API: '{table_name_to_use}'")
        else:
            # Try to find a case-insensitive match
            for lancedb_name in lancedb_table_names:
                if lancedb_name.lower() == dbName.lower():
                    table_name_to_use = lancedb_name
                    logger.info(f"Using case-insensitive match from LanceDB API: '{table_name_to_use}'")
                    break
    
    # If no match found in API, use dbName as fallback
    if table_name_to_use is None:
        table_name_to_use = dbName
        logger.info(f"Using table name '{table_name_to_use}' (not found in LanceDB API, using directory-based name)")
    
    # Try to open the table using multiple methods
    logger.info(f"Attempting to open table: '{table_name_to_use}'")
    table = None
    last_error = None
    tried_names = []
    
    # Method 1: Try open_table with the determined name
    try:
        table = db.open_table(table_name_to_use)
        logger.info(f"Successfully opened table using open_table('{table_name_to_use}')")
        tried_names.append(table_name_to_use)
    except Exception as e:
        last_error = e
        tried_names.append(table_name_to_use)
        logger.debug(f"Method 1 (open_table) failed: {type(e).__name__}: {e}")
    
    # Method 2: Try bracket notation (db[table_name])
    if table is None:
        try:
            table = db[table_name_to_use]
            logger.info(f"Successfully opened table using db['{table_name_to_use}']")
        except Exception as e:
            last_error = e
            logger.debug(f"Method 2 (bracket notation) failed: {type(e).__name__}: {e}")
    
    # Method 2b: Try db.table() method (alternative API)
    if table is None:
        try:
            if hasattr(db, 'table'):
                table = db.table(table_name_to_use)
                logger.info(f"Successfully opened table using db.table('{table_name_to_use}')")
        except Exception as e:
            last_error = e
            logger.debug(f"Method 2b (db.table) failed: {type(e).__name__}: {e}")
    
    # Method 3: If API reported the table exists but open failed, try the original dbName
    if table is None and lancedb_table_names and table_name_to_use in lancedb_table_names:
        logger.warning(f"LanceDB API reported table '{table_name_to_use}' exists but open failed. Trying original dbName '{dbName}'")
        try:
            table = db.open_table(dbName)
            logger.info(f"Successfully opened table using open_table('{dbName}')")
            tried_names.append(dbName)
            last_error = None
        except Exception as e:
            last_error = e
            tried_names.append(dbName)
            logger.debug(f"Method 3 (fallback dbName) failed: {type(e).__name__}: {e}")
    
    # Method 4: Try bracket notation with dbName
    if table is None:
        try:
            table = db[dbName]
            logger.info(f"Successfully opened table using db['{dbName}']")
        except Exception as e:
            last_error = e
            logger.debug(f"Method 4 (bracket notation with dbName) failed: {type(e).__name__}: {e}")
    
    # Method 4b: Try db.table() with dbName
    if table is None:
        try:
            if hasattr(db, 'table'):
                table = db.table(dbName)
                logger.info(f"Successfully opened table using db.table('{dbName}')")
        except Exception as e:
            last_error = e
            logger.debug(f"Method 4b (db.table with dbName) failed: {type(e).__name__}: {e}")
    
    # Method 5: Try opening by direct path if table directory exists
    if table is None:
        table_dir_path = db_path_obj / (dbName + ".lance")
        if table_dir_path.exists() and table_dir_path.is_dir():
            logger.warning(f"Trying to open table using direct path: {table_dir_path}")
            try:
                # Try connecting directly to the table directory
                # Note: connect is already imported at the top of the file
                table_db = connect(str(table_dir_path.parent))
                table = table_db.open_table(dbName)
                logger.info(f"Successfully opened table using direct path method")
            except Exception as e:
                last_error = e
                logger.debug(f"Method 5 (direct path) failed: {type(e).__name__}: {e}")
    
    if table is None:
        # Verify table directory structure
        table_dir_path = db_path_obj / (dbName + ".lance")
        if table_dir_path.exists() and table_dir_path.is_dir():
            logger.info(f"Table directory exists: {table_dir_path}")
            # Check for required LanceDB table structure files
            manifest_path = table_dir_path / "_versions" / "1.manifest"
            if manifest_path.exists():
                logger.info(f"Table manifest found: {manifest_path}")
            else:
                logger.warning(f"Table manifest not found at: {manifest_path}")
                # List what's actually in the directory
                try:
                    dir_contents = list(table_dir_path.iterdir())
                    logger.info(f"Table directory contents: {[d.name for d in dir_contents]}")
                except Exception as e:
                    logger.warning(f"Could not list table directory contents: {e}")
        
        # Provide helpful error message with available tables
        all_tried = list(set([table_name_to_use, dbName] + (tried_names if tried_names else [])))
        error_msg_parts = [
            f"Could not open table '{table_name_to_use}' (tried methods: open_table, bracket notation, table() method).",
            f"Tried names: {all_tried}.",
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