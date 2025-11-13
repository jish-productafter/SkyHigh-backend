import json
import os
from http import HTTPStatus

from fastapi import APIRouter, HTTPException
from utils.prompts import get_listening_prompt
from openai import OpenAI
from starlette.responses import Response

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

router = APIRouter()

@router.post("/listening")
def generate_listening(topic: str, level: str = "A1"):
    try:
        prompt = get_listening_prompt(topic, level)

        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-8b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
        )

        # Error handling: Check if response has choices
        if not response.choices:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No response choices returned from the model"
            )

        # Error handling: Check if message exists
        if not response.choices[0].message:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No message in response choices"
            )

        content = response.choices[0].message.content

        # Error handling: Check if content exists
        if content is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Empty content in response message"
            )

        print(content)

        # Try to parse as JSON
        try:
            parsed_json = json.loads(content)
            # If successful, return parsed JSON
            return Response(
                content=json.dumps(parsed_json),
                media_type="application/json",
                status_code=HTTPStatus.OK
            )
        except json.JSONDecodeError:
            # If not JSON, return as plain text
            return Response(
                content=content,
                media_type="text/plain",
                status_code=HTTPStatus.OK
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error generating listening content: {str(e)}"
        )