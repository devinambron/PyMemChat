# chatbot.py
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import Config
from memory_manager import MemoryManager
from utils import process_memory_data, sanitize_user_input
from exceptions import APICallError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self):
        self.config = Config()
        self.chat = self._initialize_chat()
        self.memory_manager = MemoryManager(self.config.MEMORY_FILE, llm=self.chat)
        self.chat_chain = self._create_chat_chain()

    def _initialize_chat(self) -> ChatOpenAI:
        """Initialize the LLM"""
        logger.debug("Initializing ChatOpenAI model.")
        return ChatOpenAI(
            model=self.config.MODEL_NAME,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
            openai_api_base=self.config.OPENAI_API_BASE,
            openai_api_key=self.config.OPENAI_API_KEY,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

    def _create_chat_chain(self) -> RunnableSequence:
        """
        Create a chat chain using LCEL.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="context"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        chain = (
            {
                "context": lambda x: self._get_context_messages(x["question"]),
                "chat_history": lambda x: x["chat_history"],
                "question": lambda x: x["question"]
            }
            | prompt
            | self.chat
            | StrOutputParser()
        )
        
        return chain
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the chatbot"""
        return f"""You are a helpful assistant named {self.config.AI_NAME}.
You have access to the following information to help you remember the conversation:
1. Previous conversation history
2. Key facts about entities mentioned in the conversation
3. A summary of the conversation so far
4. Relevant context from previous conversations

Use this information to provide helpful, contextually relevant responses.
Be concise, focused and helpful."""
    
    def _get_context_messages(self, question: str) -> List[BaseMessage]:
        """
        Get context messages based on the query.
        """
        # Skip extensive processing for simple statements
        if question.lower().startswith("my name is ") or "hi" == question.lower() or "hello" == question.lower():
            return []
        
        return self.memory_manager.get_context_from_question(question)

    def load_memory(self) -> None:
        """Load memory from file and populate vector store"""
        memory_data = self.memory_manager.load_memory()
        logger.debug(f"Raw memory data loaded: {memory_data}")
        if not memory_data:
            memory_data = []
        
        processed_messages = process_memory_data(memory_data)
        
        # Add to vector store
        self.memory_manager.add_to_vector_store(processed_messages)
        
        # Add to chat history
        self.memory_manager.chat_history = processed_messages
        
        logger.debug(f"Loaded memory data: {len(processed_messages)} messages")

    def save_memory(self) -> None:
        """Save memory to file"""
        self.memory_manager.save_memory(self.memory_manager.chat_history)
        logger.debug("Saved chat history to file")

    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to the user input using the LLM.
        
        Args:
            user_input: The user's input message
            
        Returns:
            The AI's response
        """
        
        self.logger.debug(f"User input received: {user_input}")
        
        # Create human message
        human_message = HumanMessage(content=user_input)
        
        # Get context based on the user query
        context = self._get_context_messages(user_input)
        
        # Create the messages for the LLM
        messages = [
            SystemMessage(content=self._get_system_prompt()),
        ]
        
        # Add context messages if available
        if context:
            messages.extend(context)
        
        # Add the conversation history via MessagesPlaceholder
        messages.append(
            MessagesPlaceholder(variable_name="chat_history")
        )
        
        # Add the human message
        messages.append(human_message)
        
        # Generate a response using the chat chain
        ai_response = self.chat_chain.invoke({
            "chat_history": self.memory_manager.get_chat_history()
        })
        
        # Create an AI message
        ai_message = AIMessage(content=ai_response)
        
        # Add the messages to memory
        self.memory_manager.add_message(human_message)
        self.memory_manager.add_message(ai_message)
        
        return ai_response

    def run(self) -> None:
        """Run the chatbot in an interactive loop"""
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