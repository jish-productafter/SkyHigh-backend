import json
import logging
import os
import tempfile
from http import HTTPStatus

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from openai import OpenAI
from starlette.responses import Response

from utils.prompts import evaluate_speaking_response, evaluate_writing_response
from utils.whisper import transcribe_mp3
from constants import model_name

# Set up logger
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

base_url = os.getenv("OPENAI_BASE_URL")
if not base_url:
    raise ValueError("OPENAI_BASE_URL environment variable is not set")
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

router = APIRouter()


@router.post("/speaking")
async def validate_speaking(
    file: UploadFile = File(...),
    speaking_task: str = Form(...),
) -> Response:
    """
    Validate a speaking response by transcribing audio and evaluating it.

    Args:
        file: The MP3 audio file containing the user's speaking response
        speaking_task: JSON string of the speaking task object

    Returns:
        JSON response with the evaluation results
    """
    logger.info(
        f"Received request to validate speaking response: filename='{file.filename}', size={file.size if hasattr(file, 'size') else 'unknown'}"
    )
    tmp_file_path = None

    # Create a temporary file to save the uploaded MP3
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        try:
            # Save the uploaded file to the temporary location
            logger.debug(f"Saving uploaded file to temporary location: {tmp_file.name}")
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            logger.debug(f"File saved successfully ({len(content)} bytes)")

            # Transcribe the MP3 file
            logger.info("Starting audio transcription")
            transcribed_audio_text = transcribe_mp3(tmp_file_path)
            logger.info(
                f"Transcription completed: {len(transcribed_audio_text)} characters"
            )
            logger.debug(f"Transcribed text: {transcribed_audio_text[:200]}...")

            # Parse the speaking_task JSON string
            try:
                logger.debug("Parsing speaking_task JSON")
                speaking_task_dict = json.loads(speaking_task)
                logger.debug(
                    f"Successfully parsed speaking_task with level: {speaking_task_dict.get('metadata', {}).get('level', 'unknown')}"
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse speaking_task JSON: {str(e)}")
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Invalid JSON in speaking_task: {str(e)}",
                )

            # Generate evaluation prompt
            logger.debug("Generating evaluation prompt")
            prompt = evaluate_speaking_response(
                speaking_task_dict, transcribed_audio_text
            )
            logger.debug(
                f"Evaluation prompt generated (length: {len(prompt)} characters)"
            )

            # Get evaluation from LLM
            logger.info("Sending request to OpenAI API for evaluation")
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            logger.debug("Received response from OpenAI API")

            # Error handling: Check if response has choices
            if not response.choices:
                logger.error("No response choices returned from the model")
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail="No response choices returned from the model",
                )

            # Error handling: Check if message exists
            if not response.choices[0].message:
                logger.error("No message in response choices")
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail="No message in response choices",
                )

            content = response.choices[0].message.content

            # Error handling: Check if content exists
            if content is None:
                logger.error("Empty content in response message")
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail="Empty content in response message",
                )

            logger.debug(
                f"Evaluation response received (length: {len(content)} characters)"
            )
            logger.debug(f"Evaluation response preview: {content[:200]}...")

            # Try to parse as JSON
            try:
                parsed_json = json.loads(content)
                logger.info(f"Successfully parsed evaluation JSON response")
                logger.debug(
                    f"Evaluation result: task_completed={parsed_json.get('task_completed')}, is_acceptable={parsed_json.get('is_acceptable')}, score={parsed_json.get('score_out_of_10')}"
                )
                # If successful, return parsed JSON
                return Response(
                    content=json.dumps(parsed_json),
                    media_type="application/json",
                    status_code=HTTPStatus.OK,
                )
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse evaluation response as JSON: {str(e)}. Returning as plain text"
                )
                # If not JSON, return as plain text
                return Response(
                    content=content, media_type="text/plain", status_code=HTTPStatus.OK
                )

        except HTTPException:
            # Re-raise HTTP exceptions
            logger.error("HTTPException raised, re-raising")
            raise
        except Exception as e:
            # Handle any other unexpected errors
            logger.exception(f"Unexpected error validating speaking response: {str(e)}")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=f"Error validating speaking response: {str(e)}",
            )
        finally:
            # Clean up the temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                logger.debug(f"Cleaning up temporary file: {tmp_file_path}")
                os.unlink(tmp_file_path)
                logger.debug("Temporary file deleted successfully")


@router.post("/writing")
async def validate_writing(
    writing_task: str = Form(...),
    user_response: str = Form(...),
) -> Response:
    """
    Validate a writing response by evaluating it.

    Args:
        writing_task: JSON string of the writing task object
        user_response: JSON string of the user's response
    """
    logger.info(
        f"Received request to validate writing response: writing_task='{writing_task}', user_response='{user_response}'"
    )
    try:
        logger.debug(f"Parsing writing_task JSON: {writing_task}")
        writing_task_dict = json.loads(writing_task)
        logger.debug(
            f"Successfully parsed writing_task with level: {writing_task_dict.get('metadata', {}).get('level', 'unknown')}"
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse writing_task JSON: {str(e)}")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid JSON in writing_task: {str(e)}",
        )
    logger.debug("Generating evaluation prompt")
    prompt = evaluate_writing_response(writing_task_dict, user_response)
    logger.debug(f"Evaluation prompt generated (length: {len(prompt)} characters)")
    logger.info("Sending request to OpenAI API for evaluation")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    logger.debug("Received response from OpenAI API")
    content = response.choices[0].message.content
    logger.debug(f"Evaluation response received (length: {len(content)} characters)")
    logger.debug(f"Evaluation response preview: {content[:200]}...")
    try:
        parsed_json = json.loads(content)
        logger.info(f"Successfully parsed evaluation JSON response")
        logger.debug(
            f"Evaluation result: task_completed={parsed_json.get('task_completed')}, is_acceptable={parsed_json.get('is_acceptable')}, score={parsed_json.get('score_out_of_10')}"
        )
        # If successful, return parsed JSON
        return Response(
            content=json.dumps(parsed_json),
            media_type="application/json",
            status_code=HTTPStatus.OK,
        )
    except json.JSONDecodeError as e:
        logger.warning(
            f"Failed to parse evaluation response as JSON: {str(e)}. Returning as plain text"
        )
        # If not JSON, return as plain text
        return Response(
            content=content, media_type="text/plain", status_code=HTTPStatus.OK
        )
    except Exception as e:
        logger.exception(f"Unexpected error validating writing response: {str(e)}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error validating writing response: {str(e)}",
        )
