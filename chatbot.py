import logging
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import Config
from memory_manager import AdvancedMemoryManager
from utils import process_memory_data, sanitize_user_input
from exceptions import APICallError
from langchain.schema import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self):
        self.config = Config()
        self.chat = self._initialize_chat()
        self.memory_manager = AdvancedMemoryManager(self.config.MEMORY_FILE, self.chat)
        self.prompt = self._create_prompt()

    def _initialize_chat(self) -> ChatOpenAI:
        logger.debug("Initializing ChatOpenAI model.")
        return ChatOpenAI(
            model=self.config.MODEL_NAME,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
            openai_api_base=self.config.OPENAI_API_BASE,
            openai_api_key=self.config.OPENAI_API_KEY,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

    def _create_prompt(self) -> ChatPromptTemplate:
        logger.debug("Defining prompt template.")
        return ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant named {self.config.AI_NAME}. Use the following information to assist in your responses:\n\nLong-term memory: {{long_term_memory}}\n\nSession information: {{session_memory}}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

    def load_memory(self) -> None:
        self.memory_manager.load_memory()
        logger.debug("Memory loaded successfully.")

    def save_memory(self) -> None:
        self.memory_manager.save_memory()
        logger.debug("Memory saved successfully.")

    def generate_response(self, user_input: str) -> str:
        try:
            short_term_memory = self.memory_manager.get_memory('short_term')
            long_term_summary = self.memory_manager.get_memory('long_term')
            session_memory = self.memory_manager.get_memory('session')

            session_memory_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in session_memory])

            messages = [
                SystemMessage(content=f"You are a helpful assistant named {self.config.AI_NAME}. Use the following information to assist in your responses:\n\nLong-term memory: {long_term_summary}\n\nSession information: {session_memory_str}"),
                *short_term_memory,
                HumanMessage(content=user_input)
            ]

            print(f"{self.config.AI_NAME}: ", end='', flush=True)
            response = self.chat.invoke(messages)
            ai_message = response.content

            # Use the new classification method
            conversation = {'role': 'user', 'content': user_input}
            self.memory_manager.add_to_memory_with_classification(conversation)
            
            conversation = {'role': 'ai', 'content': ai_message}
            self.memory_manager.add_to_memory_with_classification(conversation)

            # Review and summarize memory after each interaction
            self.memory_manager.review_and_summarize()

            # Check if verbose logging is enabled before printing the JSON response
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Memory summary: {long_term_summary}")

            print("")
            return ai_message
        except Exception as e:
            raise APICallError(f"Error generating response: {e}")

    def run(self) -> None:
        self.load_memory()
        while True:
            try:
                user_input = input("You: ")
                user_input = sanitize_user_input(user_input)
                logger.debug(f"User input received: {user_input}")
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    logger.info("User has chosen to exit the chat.")
                    break

                self.generate_response(user_input)
                self.save_memory()  # Save memory after each interaction

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Exiting chat.")
                break

        self.save_memory()
        logger.info("Memory saved successfully. Chat ended.")