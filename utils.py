# utils.py
import logging
from typing import List, Dict, Union, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool) -> None:
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def process_memory_data(memory_data: Union[Dict[str, Union[str, List[Dict[str, str]]]], List]) -> List[BaseMessage]:
    """Process memory data from file into message objects"""
    processed_messages = []
    
    if isinstance(memory_data, dict):
        for key, value in memory_data.items():
            if isinstance(value, list):
                for message in value:
                    if message.get('role') == 'user':
                        processed_messages.append(HumanMessage(content=message['content']))
                    elif message.get('role') == 'ai' or message.get('role') == 'assistant':
                        processed_messages.append(AIMessage(content=message['content']))
                    else:
                        logger.warning(f"Unknown role '{message.get('role')}' in message: {message}")
            elif isinstance(value, str):
                processed_messages.append(SystemMessage(content=value))
    elif isinstance(memory_data, list):
        for message in memory_data:
            if isinstance(message, dict):
                if message.get('role') == 'user':
                    processed_messages.append(HumanMessage(content=message['content']))
                elif message.get('role') == 'ai' or message.get('role') == 'assistant':
                    processed_messages.append(AIMessage(content=message['content']))
                elif message.get('role') == 'system':
                    processed_messages.append(SystemMessage(content=message['content']))
                else:
                    logger.warning(f"Unknown role '{message.get('role')}' in message: {message}")
            elif isinstance(message, str):
                processed_messages.append(SystemMessage(content=message))
            else:
                logger.warning(f"Unexpected message format: {message}")
    else:
        logger.warning(f"Unexpected memory data format: {memory_data}")
    
    return processed_messages

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