# utils.py
import logging
from typing import List, Dict, Union, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool) -> None:
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def process_memory_data(memory_data: List[Dict[str, str]]) -> List[BaseMessage]:
    """
    Process memory data from file into Langchain message objects.
    
    Args:
        memory_data: List of dictionaries with role and content keys
        
    Returns:
        List of BaseMessage objects
    """
    # Create a ChatMessageHistory to handle the conversion
    history = ChatMessageHistory()
    
    if not memory_data:
        return history.messages
        
    # Add messages to history in order
    for message in memory_data:
        if not isinstance(message, dict):
            logger.warning(f"Unexpected message format, expecting dict: {message}")
            continue
            
        role = message.get('role')
        content = message.get('content')
        
        if not role or not content:
            logger.warning(f"Message missing role or content: {message}")
            continue
            
        if role == 'user':
            history.add_user_message(content)
        elif role == 'ai' or role == 'assistant':
            history.add_ai_message(content)
        elif role == 'system':
            # Convert system messages separately as ChatMessageHistory doesn't handle them
            history.messages.append(SystemMessage(content=content))
        else:
            logger.warning(f"Unknown role '{role}' in message: {message}")
    
    return history.messages

def sanitize_user_input(user_input: str) -> str:
    """Clean user input to prevent issues"""
    return user_input.strip()

def format_context_for_prompt(context_data: Dict[str, Any]) -> str:
    """Format context data for inclusion in a prompt"""
    sections = []
    
    # Add summary if available
    if context_data.get("conversation_summary"):
        sections.append(f"CONVERSATION SUMMARY:\n{context_data['conversation_summary']}")
    
    # Add entities if available
    if context_data.get("entity_information"):
        sections.append(f"ENTITY INFORMATION:\n{context_data['entity_information']}")
    
    # Add retrieved documents if available
    if context_data.get("retrieved_documents"):
        docs_text = "\n---\n".join(context_data["retrieved_documents"])
        sections.append(f"RELEVANT CONTEXT:\n{docs_text}")
    
    return "\n\n".join(sections)