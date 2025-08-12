import os
import time
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_rate_limit_error(exception):
    """Check if the exception is a rate limit error"""
    if hasattr(exception, 'status_code'):
        return exception.status_code == 429
    if hasattr(exception, 'code'):
        return exception.code == 429
    error_msg = str(exception).lower()
    return any(phrase in error_msg for phrase in [
        'rate limit', 'quota exceeded', 'too many requests', 
        '429', 'resource_exhausted'
    ])

def retry_on_rate_limit(exception):
    """Retry condition for rate limiting"""
    should_retry = is_rate_limit_error(exception)
    if should_retry:
        logger.warning(f"Rate limit hit: {exception}. Retrying with exponential backoff...")
    return should_retry

@retry(
    retry=retry_on_rate_limit,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True
)
def generate_text(prompt):
    try:
        logger.info("Generating text with Gemini 2.0 Flash")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        logger.info("Text generation completed successfully")
        return response.content
    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        if is_rate_limit_error(e):
            logger.warning("Rate limit detected, will retry with exponential backoff")
        raise e

@retry(
    retry=retry_on_rate_limit,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True
)
def generate_code(prompt):
    try:
        logger.info("Generating code with Gemini 2.0 Flash")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        messages = [
            HumanMessage(content=f"You are a helpful coding assistant. Generate code for the following prompt: {prompt}"),
        ]
        response = llm.invoke(messages)
        logger.info("Code generation completed successfully")
        return response.content
    except Exception as e:
        logger.error(f"Error in generate_code: {str(e)}")
        if is_rate_limit_error(e):
            logger.warning("Rate limit detected, will retry with exponential backoff")
        raise e

@retry(
    retry=retry_on_rate_limit,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True
)
def classify_text(text, categories):
    try:
        logger.info("Classifying text with Gemini 2.0 Flash")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        prompt = f"Classify the following text: '{text}' into one of the following categories: {', '.join(categories)}. Only return the category name."
        messages = [
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        logger.info("Text classification completed successfully")
        return response.content
    except Exception as e:
        logger.error(f"Error in classify_text: {str(e)}")
        if is_rate_limit_error(e):
            logger.warning("Rate limit detected, will retry with exponential backoff")
        raise e