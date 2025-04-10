# utils.py
import logging
import json
import re
from typing import List, Dict, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False, level: int = logging.INFO, log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s') -> None:
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging (overrides level to DEBUG).
        level: The default logging level if verbose is False.
        log_format: The format string for log messages.
    """
    # Determine the final log level
    log_level = logging.DEBUG if verbose else level
    
    # If not verbose, set the default level to WARNING
    if not verbose:
        log_level = logging.WARNING
    
    # Configure root logger
    # Remove existing handlers to avoid duplicate logs if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Silence excessively noisy third-party loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING) # Added for httpx dependency
    logging.getLogger("faiss").setLevel(logging.WARNING) # Silence FAISS info logs
    logging.getLogger("nomic").setLevel(logging.WARNING) # Silence Nomic info logs (if any)

    if verbose:
        logger.info(f"Verbose logging enabled. Root logger level set to DEBUG.")
    else:
        logger.debug(f"Standard logging enabled. Root logger level set to {logging.getLevelName(log_level)}.")
        # Add a debug log to confirm non-verbose mode is active, won't show unless root is DEBUG

def process_memory_data(memory_data: List[Dict]) -> List[BaseMessage]:
    """
    Process memory data and convert to BaseMessage objects.
    
    Args:
        memory_data: A list of dictionaries containing memory data
        
    Returns:
        A list of BaseMessage objects
    """
    messages = []
    # Check if memory_data is indeed a list
    if not isinstance(memory_data, list):
        logger.error(f"Invalid memory data format: Expected list, got {type(memory_data)}")
        return []
        
    for item in memory_data:
        # Ensure item is a dictionary and has 'role' and 'content'
        if isinstance(item, dict) and 'role' in item and 'content' in item:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user" or role == "human": # Accept both 'user' and 'human'
                messages.append(HumanMessage(content=content))
            elif role == "ai" or role == "assistant": # Accept both 'ai' and 'assistant'
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            else:
                logger.warning(f"Unsupported role found in memory data: {role}")
        else:
            logger.warning(f"Skipping invalid memory item: {item}")
    return messages

def sanitize_user_input(input_text: str) -> str:
    """
    Sanitize user input to prevent injection attacks or unwanted formatting.
    
    Args:
        input_text: The raw user input
        
    Returns:
        Sanitized input text
    """
    # Remove any attempt to impersonate system or control messages
    sanitized = re.sub(r'(system|user|assistant):', '', input_text, flags=re.IGNORECASE)
    
    # Remove any markdown code block syntax that might be confused with system instructions
    sanitized = re.sub(r'```.*?```', '[code removed]', sanitized, flags=re.DOTALL)
    
    # Trim extra whitespace
    sanitized = sanitized.strip()
    
    return sanitized

def format_context_for_prompt(context_data: Dict[str, Any]) -> List[BaseMessage]:
    """
    Format context data for inclusion in prompts.
    
    Args:
        context_data: A dictionary containing context information
            with keys like 'summary', 'entities', and 'documents'
            
    Returns:
        A list of SystemMessage objects containing formatted context
    """
    messages = []
    
    # Add conversation summary if available
    if 'summary' in context_data and context_data['summary']:
        messages.append(SystemMessage(
            content=f"Conversation summary: {context_data['summary']}"
        ))
    
    # Add entity information if available
    if 'entities' in context_data and context_data['entities']:
        try:
            if isinstance(context_data['entities'], str):
                # Attempt to load if it's a JSON string
                try:
                    entities = json.loads(context_data['entities'])
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode entity data string: {context_data['entities']}")
                    entities = None # Treat as invalid
            elif isinstance(context_data['entities'], dict):
                 entities = context_data['entities']
            else:
                logger.warning(f"Unexpected type for entity data: {type(context_data['entities'])}")
                entities = None
                
            if entities:
                entities_message = "The following entities have been mentioned:"
                for entity, description in entities.items():
                    entities_message += f"\n- {entity}: {description}"
                messages.append(SystemMessage(content=entities_message))
        except Exception as e:
            # Log but continue without entities
            logger.warning(f"Error formatting entity data: {e}")
    
    # Add relevant documents if available
    if 'documents' in context_data and context_data['documents']:
        docs = context_data['documents']
        if isinstance(docs, list) and docs:
            # Check if docs are strings or have page_content
            formatted_docs = []
            for doc in docs:
                if isinstance(doc, str):
                    formatted_docs.append(doc)
                elif hasattr(doc, 'page_content'):
                    formatted_docs.append(doc.page_content)
                else:
                     logger.warning(f"Skipping document with unexpected format: {type(doc)}")
            
            if formatted_docs:
                docs_message = "Previous conversations contained the following relevant information:\n"
                docs_message += "\n---\n".join(formatted_docs)
                messages.append(SystemMessage(content=docs_message))
    
    return messages