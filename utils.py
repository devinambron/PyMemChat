import logging
from typing import List, Dict
from langchain.schema.messages import AIMessage, HumanMessage

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def process_memory_data(memory_data: List[Dict[str, str]]) -> List[AIMessage | HumanMessage]:
    processed_messages = []
    for message in memory_data:
        if message['role'] == 'user':
            processed_messages.append(HumanMessage(content=message['content']))
        else:
            processed_messages.append(AIMessage(content=message['content']))
    return processed_messages

def sanitize_user_input(user_input: str) -> str:
    # Add any input sanitization logic here
    return user_input.strip()