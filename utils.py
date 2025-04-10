# utils.py
import logging
import json
import re
from typing import List, Dict, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Silence noisy loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Set up debug logger if verbose
    if verbose:
        # Create a logger for detailed debugging
        debug_logger = logging.getLogger("debug")
        debug_logger.setLevel(logging.DEBUG)
        
        # Add handler for debug logs
        debug_handler = logging.StreamHandler()
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        debug_logger.addHandler(debug_handler)

def process_memory_data(memory_data: List[Dict]) -> List[BaseMessage]:
    """
    Process memory data and convert to BaseMessage objects.
    
    Args:
        memory_data: A list of dictionaries containing memory data
        
    Returns:
        A list of BaseMessage objects
    """
    messages = []
    for item in memory_data:
        if item.get("role") == "user":
            messages.append(HumanMessage(content=item.get("content", "")))
        elif item.get("role") == "ai":
            messages.append(AIMessage(content=item.get("content", "")))
        elif item.get("role") == "system":
            messages.append(SystemMessage(content=item.get("content", "")))
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
    sanitized = re.sub(r'(system|user|assistant):', '', input_text)
    
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
                entities = json.loads(context_data['entities'])
            else:
                entities = context_data['entities']
                
            if entities:
                entities_message = "The following entities have been mentioned:"
                for entity, description in entities.items():
                    entities_message += f"\n- {entity}: {description}"
                messages.append(SystemMessage(content=entities_message))
        except Exception as e:
            # Log but continue without entities
            logging.warning(f"Error formatting entity data: {e}")
    
    # Add relevant documents if available
    if 'documents' in context_data and context_data['documents']:
        docs = context_data['documents']
        if isinstance(docs, list) and docs:
            docs_message = "Previous conversations contained the following relevant information:\n"
            docs_message += "\n---\n".join(docs)
            messages.append(SystemMessage(content=docs_message))
    
    return messages