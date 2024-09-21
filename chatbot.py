import logging
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import Config
from memory_manager import MemoryManager
from utils import process_memory_data, sanitize_user_input
from exceptions import APICallError
from langchain.schema import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self):
        self.config = Config()
        self.memory_manager = MemoryManager(self.config.MEMORY_FILE)
        self.memory = ConversationBufferMemory(return_messages=True)
        self.chat = self._initialize_chat()
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
            ("system", f"You are a helpful assistant named {self.config.AI_NAME}."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

    def load_memory(self) -> None:
        memory_data = self.memory_manager.load_memory()
        processed_messages = process_memory_data(memory_data)
        for message in processed_messages:
            self.memory.chat_memory.add_message(message)
        logger.debug(f"Loaded memory data: {memory_data}")

    def save_memory(self) -> None:
        self.memory_manager.save_memory(self.memory.chat_memory.messages)

    def generate_response(self, user_input: str) -> str:
        try:
            memory_variables = self.memory.load_memory_variables({})
            logger.debug(f"Memory variables before invoking chain: {memory_variables}")

            messages = [
                SystemMessage(content=f"You are a helpful assistant named {self.config.AI_NAME}."),
                *memory_variables["history"],
                HumanMessage(content=user_input)
            ]

            print(f"{self.config.AI_NAME}: ", end='', flush=True)
            response = self.chat.invoke(messages)
            ai_message = response.content

            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(ai_message)
            logger.debug("Saved user and AI messages to memory.")

            print(f"")

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

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Exiting chat.")
                break

        self.save_memory()
        logger.info("Memory saved successfully. Chat ended.")