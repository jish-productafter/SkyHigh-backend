from lancedb import connect
from typing import Optional
from pathlib import Path

# Get the absolute path to the database directory
# This file is in app/utils/, so we go up one level to app/, then to utils/lancedb_data
_current_dir = Path(__file__).parent
db_path = str(_current_dir / "lancedb_data")

# Lazy-loaded model cache - only loaded on first use
_model: Optional[object] = None


def _get_model():
    """Lazy load the embedding model only when needed (not on module import)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
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
    if level not in {"A1", "A2", "B1", "B2"}:
        raise ValueError("Invalid level. Must be 'A1', 'A2', 'B1', or 'B2'.")
    
    dbName = level + "_MINIMAL_vocabulary"
    db = connect(db_path)
    # LanceDB table names match directory names, which include .lance extension
    table_name = dbName + ".lance"
    try:
        table = db.open_table(table_name)
    except ValueError:
        # Fallback: try without .lance extension in case LanceDB handles it automatically
        table = db.open_table(dbName)
    
    # Lazy load model and encode query (model is cached after first use)
    model = _get_model()
    # Use convert_to_numpy=True for faster encoding and direct compatibility with LanceDB
    query_vector = model.encode(query, convert_to_numpy=True)
    results = table.search(query_vector).limit(n).to_pandas()

    # only take german_term & english_translation
    results = results[['german_term', 'english_translation']]
    print(results)
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
            return results[vocab_column].head(n).tolist()
        else:
            # Return first column if no standard name found
            return results.iloc[:, 0].head(n).tolist()
    
    return []


if __name__ == "__main__":
    results = fetch_vocab_from_vector_db("food", "B2")
    for result in results:
        print(result)