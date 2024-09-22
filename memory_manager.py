import json
import logging
from typing import List, Dict
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from exceptions import MemoryLoadError, MemorySaveError

logger = logging.getLogger(__name__)

class AdvancedMemoryManager:
    def __init__(self, file_path: str, llm):
        self.file_path = file_path
        self.llm = llm
        self.short_term_memory = ConversationBufferMemory(return_messages=True)
        self.long_term_memory = ConversationSummaryMemory(llm=llm)
        self.session_memory = []

    def load_memory(self) -> Dict[str, any]:
        logger.debug(f"Loading memory from file: {self.file_path}")
        try:
            with open(self.file_path, 'r') as f:
                memory_data = json.load(f)
            
            # Load short-term memory
            for msg in memory_data.get('short_term', []):
                self.short_term_memory.chat_memory.add_message(self._create_message(msg))
            
            # Load long-term memory
            self.long_term_memory.buffer = memory_data.get('long_term', '')
            
            self.session_memory = memory_data.get('session', [])
            
            logger.debug("Memory loaded successfully")
            return memory_data
        except FileNotFoundError:
            logger.warning("Memory file not found, initializing empty memory.")
            return {'short_term': [], 'long_term': '', 'session': []}
        except json.JSONDecodeError as e:
            raise MemoryLoadError(f"Error decoding memory file: {e}")

    def save_memory(self) -> None:
        logger.debug(f"Saving memory to file: {self.file_path}")
        try:
            memory_data = {
                'short_term': self._serialize_messages(self.short_term_memory.chat_memory.messages),
                'long_term': self.long_term_memory.buffer,
                'session': self.session_memory
            }
            with open(self.file_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
            logger.debug("Memory saved successfully.")
        except IOError as e:
            raise MemorySaveError(f"Error saving memory to file: {e}")

    def add_to_memory(self, memory_type: str, message: Dict[str, str]) -> None:
        if memory_type == 'short_term':
            self.short_term_memory.chat_memory.add_message(self._create_message(message))
        elif memory_type == 'long_term':
            # Not adding directly to long_term_memory; it should be handled via summarization
            pass
        elif memory_type == 'session':
            self.session_memory.append(message)

    def get_memory(self, memory_type: str) -> any:
        if memory_type == 'short_term':
            return self.short_term_memory.chat_memory.messages
        elif memory_type == 'long_term':
            return self.long_term_memory.buffer
        elif memory_type == 'session':
            return self.session_memory

    def _create_message(self, message: Dict[str, str]):
        if message['role'] == 'user':
            return HumanMessage(content=message['content'])
        elif message['role'] == 'ai':
            return AIMessage(content=message['content'])
        else:
            # For system messages or any other type
            return SystemMessage(content=message['content'])

    def _serialize_messages(self, messages: List) -> List[Dict[str, str]]:
        return [{'role': 'user' if isinstance(msg, HumanMessage) else 'ai', 'content': msg.content} for msg in messages]

    def classify_memory(self, conversation: Dict[str, str]) -> str:
        prompt = f"Classify this memory from the conversation: {conversation} into short-term, long-term, or session-based storage."
        classification = self.llm.predict(prompt)
        return classification.lower()

    def evaluate_relevance(self, conversation: Dict[str, str]) -> bool:
        prompt = f"Is this conversation: {conversation} relevant for future reference? Should it be stored or discarded?"
        decision = self.llm.predict(prompt)
        return "store" in decision.lower()

    def add_to_memory_with_classification(self, conversation: Dict[str, str]) -> None:
        if self.evaluate_relevance(conversation):
            classification = self.classify_memory(conversation)
            if "short_term" in classification:
                self.add_to_memory('short_term', conversation)
            if "long_term" in classification:
                self.add_to_memory('long_term', conversation)
            if "session" in classification:
                self.add_to_memory('session', conversation)
        else:
            logger.info("Conversation deemed not relevant for storage.")

    def review_and_summarize(self):
        try:
            # Combine short-term and long-term memory
            combined_memory = self.short_term_memory.chat_memory.messages + [SystemMessage(content=self.long_term_memory.buffer)]
            
            # Create a new summary
            new_summary = self.long_term_memory.predict_new_summary(combined_memory, "")
            
            # Validate the new summary
            if self._validate_summary(new_summary):
                # Update long-term memory
                self.long_term_memory.buffer = new_summary
            else:
                logger.warning("Invalid summary generated. Keeping previous summary.")
            
            # Clear short-term memory
            self.short_term_memory.clear()
        except Exception as e:
            logger.error(f"Error during review and summarize: {e}")

    def _validate_summary(self, summary):
        # Implement validation logic here
        # For example, check if the summary contains unexpected topics
        return "math homework" not in summary.lower()