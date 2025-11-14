import json
import logging
import os
from http import HTTPStatus
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from utils.prompts import (
    get_listening_prompt,
    get_reading_prompt,
    get_writing_prompt,
    get_speaking_prompt,
)
from openai import OpenAI
from starlette.responses import Response

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)

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
    logger.info(f"Received request to generate listening content: topic='{topic}', level={level}")
    try:
        logger.debug(f"Generating listening prompt for topic='{topic}', level={level}")
        prompt = get_listening_prompt(topic, level)
        logger.debug(f"Prompt generated successfully (length: {len(prompt)} characters)")

        logger.info("Sending request to OpenAI API for listening content generation")
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-8b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
        )
        logger.debug("Received response from OpenAI API")

        # Error handling: Check if response has choices
        if not response.choices:
            logger.error("No response choices returned from the model")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No response choices returned from the model"
            )

        # Error handling: Check if message exists
        if not response.choices[0].message:
            logger.error("No message in response choices")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No message in response choices"
            )

        content = response.choices[0].message.content

        # Error handling: Check if content exists
        if content is None:
            logger.error("Empty content in response message")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Empty content in response message"
            )

        logger.debug(f"Response content received (length: {len(content)} characters)")
        logger.debug(f"Response content preview: {content[:200]}...")

        # Try to parse as JSON
        try:
            parsed_json = json.loads(content)
            logger.info(f"Successfully parsed JSON response with {len(parsed_json)} items")
            # If successful, return parsed JSON
            return Response(
                content=json.dumps(parsed_json),
                media_type="application/json",
                status_code=HTTPStatus.OK
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response as JSON: {str(e)}. Returning as plain text")
            # If not JSON, return as plain text
            return Response(
                content=content,
                media_type="text/plain",
                status_code=HTTPStatus.OK
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        logger.error("HTTPException raised, re-raising")
        raise
    except Exception as e:
        # Handle any other unexpected errors
        logger.exception(f"Unexpected error generating listening content: {str(e)}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error generating listening content: {str(e)}"
        )


@router.post("/reading")
def generate_reading(topic: str, level: str = "A1", item_id_start: int = 1, prefer_type: str = "MultipleChoice"):
    logger.info(f"Received request to generate reading content: topic='{topic}', level={level}, item_id_start={item_id_start}, prefer_type={prefer_type}")
    try:
        logger.debug(f"Generating reading prompt for topic='{topic}', level={level}, item_id_start={item_id_start}, prefer_type={prefer_type}")
        prompt = get_reading_prompt(topic, level, item_id_start, prefer_type)
        logger.debug(f"Prompt generated successfully (length: {len(prompt)} characters)")

        logger.info("Sending request to OpenAI API for reading content generation")
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-8b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
        )
        logger.debug("Received response from OpenAI API")

        # Error handling: Check if response has choices
        if not response.choices:
            logger.error("No response choices returned from the model")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No response choices returned from the model"
            )

        # Error handling: Check if message exists
        if not response.choices[0].message:
            logger.error("No message in response choices")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No message in response choices"
            )

        content = response.choices[0].message.content

        # Error handling: Check if content exists
        if content is None:
            logger.error("Empty content in response message")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Empty content in response message"
            )

        logger.debug(f"Response content received (length: {len(content)} characters)")
        logger.debug(f"Response content preview: {content[:200]}...")

        # Try to parse as JSON
        try:
            parsed_json = json.loads(content)
            logger.info(f"Successfully parsed JSON response with {len(parsed_json)} items")
            # If successful, return parsed JSON
            return Response(
                content=json.dumps(parsed_json),
                media_type="application/json",
                status_code=HTTPStatus.OK
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response as JSON: {str(e)}. Returning as plain text")
            # If not JSON, return as plain text
            return Response(
                content=content,
                media_type="text/plain",
                status_code=HTTPStatus.OK
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        logger.error("HTTPException raised, re-raising")
        raise
    except Exception as e:
        # Handle any other unexpected errors
        logger.exception(f"Unexpected error generating reading content: {str(e)}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error generating reading content: {str(e)}"
        )


@router.post("/writing")
def generate_writing(topic: str, level: str = "A1", item_id_start: int = 1, task_type: str = "email"):
    logger.info(f"Received request to generate writing content: topic='{topic}', level={level}, item_id_start={item_id_start}, task_type={task_type}")
    try:
        logger.debug(f"Generating writing prompt for topic='{topic}', level={level}, item_id_start={item_id_start}, task_type={task_type}")
        prompt = get_writing_prompt(topic, level, item_id_start, task_type)
        logger.debug(f"Prompt generated successfully (length: {len(prompt)} characters)")

        logger.info("Sending request to OpenAI API for writing content generation")
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-8b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
        )
        logger.debug("Received response from OpenAI API")

        # Error handling: Check if response has choices
        if not response.choices:
            logger.error("No response choices returned from the model")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No response choices returned from the model"
            )

        # Error handling: Check if message exists
        if not response.choices[0].message:
            logger.error("No message in response choices")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No message in response choices"
            )

        content = response.choices[0].message.content

        # Error handling: Check if content exists
        if content is None:
            logger.error("Empty content in response message")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Empty content in response message"
            )

        logger.debug(f"Response content received (length: {len(content)} characters)")
        logger.debug(f"Response content preview: {content[:200]}...")

        # Try to parse as JSON
        try:
            parsed_json = json.loads(content)
            logger.info(f"Successfully parsed JSON response with {len(parsed_json)} items")
            # If successful, return parsed JSON
            return Response(
                content=json.dumps(parsed_json),
                media_type="application/json",
                status_code=HTTPStatus.OK
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response as JSON: {str(e)}. Returning as plain text")
            # If not JSON, return as plain text
            return Response(
                content=content,
                media_type="text/plain",
                status_code=HTTPStatus.OK
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        logger.error("HTTPException raised, re-raising")
        raise
    except Exception as e:
        # Handle any other unexpected errors
        logger.exception(f"Unexpected error generating writing content: {str(e)}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error generating writing content: {str(e)}"
        )


@router.post("/speaking")
def generate_speaking(topic: str, level: str = "A1", item_id_start: int = 1, interaction_type: str = "interview"):
    logger.info(f"Received request to generate speaking content: topic='{topic}', level={level}, item_id_start={item_id_start}, interaction_type={interaction_type}")
    try:
        logger.debug(f"Generating speaking prompt for topic='{topic}', level={level}, item_id_start={item_id_start}, interaction_type={interaction_type}")
        prompt = get_speaking_prompt(topic, level, item_id_start, interaction_type)
        logger.debug(f"Prompt generated successfully (length: {len(prompt)} characters)")

        logger.info("Sending request to OpenAI API for speaking content generation")
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-8b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
        )
        logger.debug("Received response from OpenAI API")

        # Error handling: Check if response has choices
        if not response.choices:
            logger.error("No response choices returned from the model")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No response choices returned from the model"
            )

        # Error handling: Check if message exists
        if not response.choices[0].message:
            logger.error("No message in response choices")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No message in response choices"
            )

        content = response.choices[0].message.content

        # Error handling: Check if content exists
        if content is None:
            logger.error("Empty content in response message")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Empty content in response message"
            )

        logger.debug(f"Response content received (length: {len(content)} characters)")
        logger.debug(f"Response content preview: {content[:200]}...")

        # Try to parse as JSON
        try:
            parsed_json = json.loads(content)
            logger.info(f"Successfully parsed JSON response with {len(parsed_json)} items")
            # If successful, return parsed JSON
            return Response(
                content=json.dumps(parsed_json),
                media_type="application/json",
                status_code=HTTPStatus.OK
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response as JSON: {str(e)}. Returning as plain text")
            # If not JSON, return as plain text
            return Response(
                content=content,
                media_type="text/plain",
                status_code=HTTPStatus.OK
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        logger.error("HTTPException raised, re-raising")
        raise
    except Exception as e:
        # Handle any other unexpected errors
        logger.exception(f"Unexpected error generating speaking content: {str(e)}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error generating speaking content: {str(e)}"
        )