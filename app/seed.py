import logging
from timescale_vector import client
from psycopg2.errors import DuplicateTable
import os
import json
import time
from datetime import timedelta
from dotenv import load_dotenv
import psycopg2

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)

postgres_password = os.getenv("POSTGRES_PASSWORD")
if not postgres_password:
    raise ValueError("POSTGRES_PASSWORD environment variable is not set")

postgres_db = os.getenv("POSTGRES_DB")
if not postgres_db:
    raise ValueError("POSTGRES_DB environment variable is not set")

# Use service name for Docker Compose, or 127.0.0.1 for local development
postgres_host = os.getenv("POSTGRES_HOST", "timescaledb")
postgres_port = os.getenv("POSTGRES_PORT", "5432")

# TIMESCALE_SERVICE_URL=postgresql://postgres:password@127.0.0.1:5432/postgres

TIMESCALE_SERVICE_URL = f"postgresql://postgres:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"


def seed(json_filename: str, table_name: str = None):
    """
    Load processed records from JSON file and upsert them directly to the database.
    This skips the CSV processing and embedding generation step.

    Args:
        json_filename: Path to the JSON file containing processed records
        table_name: Name of the table (if None, inferred from JSON filename)
    """
    import numpy as np

    # Load JSON data
    with open(json_filename, "r", encoding="utf-8") as f:
        records_list = json.load(f)

    print(f"Loaded {len(records_list)} records from {json_filename}")

    # Infer table name from filename if not provided
    if table_name is None:
        # Extract table name from filename (e.g., "records_a1_minimal.json" -> "a1_minimal.csv")
        table_name = json_filename.replace("records_", "").replace(".json", ".csv")

    # Convert back to records format for upsert
    # Create structured numpy array compatible with timescale_vector
    records = []
    for record_dict in records_list:
        # Convert embedding list back to numpy array
        embedding = np.array(record_dict["embedding"], dtype=np.float32)
        record = (
            record_dict["id"],
            record_dict["metadata"],
            record_dict["contents"],
            embedding,
        )
        records.append(record)

    # Initialize vector client
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    embedding_dimensions = 384
    time_partition_interval = timedelta(days=7)

    vec_client = client.Sync(
        TIMESCALE_SERVICE_URL,
        table_name,
        embedding_dimensions,
        time_partition_interval=time_partition_interval,
    )

    # Create tables if they don't exist
    try:
        vec_client.create_tables()
        print(f"  Created tables for {table_name}")
    except DuplicateTable:
        print(f"  Tables for {table_name} already exist, skipping...")

    # Create index if it doesn't exist
    try:
        vec_client.create_embedding_index(client.DiskAnnIndex())
        print(f"  Created embedding index for {table_name}")
    except DuplicateTable:
        print(f"  Embedding index for {table_name} already exists, skipping...")

    # Upsert records
    vec_client.upsert(records)
    print(f"  Upserted {len(records)} records to {table_name}\n")

    return vec_client


def wait_for_database(max_retries=30, retry_delay=2):
    """Wait for the database to be ready to accept connections."""
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(TIMESCALE_SERVICE_URL)
            conn.close()
            print("Database is ready!")
            return True
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                print(
                    f"Waiting for database to be ready... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)
            else:
                raise Exception(f"Database not ready after {max_retries} attempts: {e}")
    return False


def seedall():
    """Seed the database with all the necessary data."""
    logger.info("Seeding the database with all the necessary data...")

    # Wait for database to be ready
    wait_for_database()

    # list of json files to seed
    # check dataset folder for existing files
    dataset_folder = "dataset"
    json_files = [f for f in os.listdir(dataset_folder) if f.endswith(".json")]
    print(json_files)

    for json_file in json_files:
        seed(
            f"{dataset_folder}/{json_file}",
            json_file.replace("records_", "").replace(".json", ".csv"),
        )
