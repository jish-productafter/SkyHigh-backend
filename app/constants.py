import os
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL_NAME")
if not model_name:
    raise ValueError("MODEL_NAME environment variable is not set")
