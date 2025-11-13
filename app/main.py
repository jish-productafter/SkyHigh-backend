import os

# Set TOKENIZERS_PARALLELISM to avoid warnings when uvicorn forks processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from router import router as process_router

app = FastAPI()
app.include_router(process_router)
