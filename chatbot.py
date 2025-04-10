# chatbot.py
import logging
import os
from typing import List, Optional, Dict, Any
from operator import itemgetter  # Import itemgetter
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# Ensure ConversationBufferMemory is imported
try:
    # Langchain >= 0.3.0
    from langchain_community.memory import ConversationBufferMemory
except ImportError:
    # Fallback for older versions
    from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from memory_manager import MemoryManager
from config import (
    AI_NAME, MEMORY_FILE, OPENAI_MODEL,
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import Config
from utils import process_memory_data, sanitize_user_input
from exceptions import APICallError
import re

logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self, model_name: str = OPENAI_MODEL, verbose: bool = False):
        self.logger = logging.getLogger(__name__)
        
        # Set up logging with appropriate verbosity
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            
        # Configure the language model
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            verbose=verbose
        )
        
        # Initialize memory manager and load memory immediately
        self.memory_file = MEMORY_FILE
        self.memory_manager = MemoryManager(self.memory_file, self.llm)
        self.memory_manager.load_memory() # Load memory during initialization
        
        # Create memory objects needed for the chain
        # Use the ChatMessageHistory instance from MemoryManager for shared state
        self.buffer_memory = ConversationBufferMemory(
            chat_memory=self.memory_manager.message_history,
            memory_key="chat_history",
            return_messages=True
        )
        # Entity memory is already created inside MemoryManager
        
        # Track AI name for responses
        self.ai_name = AI_NAME
        
        # Create a chat chain with memory
        self.chat_chain = self._create_chat_chain()
        
        self.logger.info(f"Initialized {self.ai_name} with model {model_name}")
    
    def _create_chat_chain(self):
        """Create a chat chain with memory using LCEL."""
        self.logger.debug("Creating chat chain with memory")
        
        # System prompt updated to mention entities
        system_prompt = self._get_system_prompt()
        
        # Create chat prompt template with placeholders for entities and history
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Build the chat chain using RunnablePassthrough.assign to load memory
        chat_chain = (
            RunnablePassthrough.assign(
                # Load history from buffer memory
                chat_history=RunnableLambda(self.buffer_memory.load_memory_variables) | itemgetter("chat_history"),
            )
            # Add logging step to inspect chat_history
            | RunnableLambda(lambda x: self.logger.debug(f"Chat History being passed to prompt: {x['chat_history']}") or x)
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.logger.debug("Chat chain created successfully")
        return chat_chain
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines the assistant's capabilities and persona."""
        return (
            f"You are {self.ai_name}, a helpful and friendly AI assistant. "
            f"You MUST use the provided conversation history ('chat_history') to answer questions about past interactions. "
            f"Refer to the 'chat_history' when the user asks what you talked about previously. "
            f"Avoid commenting on repetitive user questions or conversational loops; simply answer the current question based on the available history, even if the question itself seems repetitive. "
            f"If you don't know something or can't remember based on the provided context (including the full 'chat_history'), just say so instead of making up information. "
            f"Keep your responses helpful, concise, and friendly."
        )
    
    def save_memory(self) -> None:
        """Save the raw memory state via the memory manager."""
        self.logger.debug("Triggering memory save via MemoryManager")
        try:
            self.memory_manager.save_memory()
            self.logger.debug("Memory save triggered successfully")
        except Exception as e:
            self.logger.error(f"Error triggering memory save: {e}")

    def generate_response(self, user_input: str) -> str:
        """Generate a response to user input using the memory-enhanced chat chain."""
        self.logger.debug(f"Generating response to: {user_input}")
        
        # Don't add human message to memory manager here, let memory objects handle it via save_context
        
        # Generate response using the chat chain
        try:
            # Invoke chain - memory variables are loaded automatically
            response = self.chat_chain.invoke({"question": user_input})
            
            # Save context to memory objects *after* getting the response
            self.buffer_memory.save_context({"input": user_input}, {"output": response})
            # Also save context for entity memory so it updates its store
            if self.memory_manager.entity_memory:
                self.memory_manager.entity_memory.save_context({"input": user_input}, {"output": response})
            
            # Add messages to MemoryManager for persistent saving
            # This ensures chat_memory.json is updated correctly for the next session
            human_message = HumanMessage(content=user_input)
            ai_message = AIMessage(content=response)
            self.memory_manager.add_message(human_message)
            self.memory_manager.add_message(ai_message)
            
            self.logger.debug(f"Generated response: {response}")
            # Print response to console immediately
            print(f"{self.ai_name}: {response}") # Added print statement here
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            error_message = "I'm sorry, I encountered an error generating a response. Please try again."
            
            # Add error message to persistent storage via MemoryManager
            error_ai_message = AIMessage(content=error_message)
            # We might not want to save the user input that caused the error,
            # but we should save the AI's error response.
            self.memory_manager.add_message(error_ai_message)
            
            print(f"{self.ai_name}: {error_message}") # Print error message
            return error_message

    def run(self) -> None:
        """Run the chatbot in an interactive loop"""
        # Memory is loaded during __init__
        
        print(f"\n{self.ai_name} is ready to chat! Type 'exit' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("You: ")
                user_input = sanitize_user_input(user_input)
                self.logger.debug(f"User input received: {user_input}")

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print(f"\n{self.ai_name}: Goodbye! Have a great day.")
                    self.logger.info("User has chosen to exit the chat.")
                    break

                # generate_response now also prints the response
                self.generate_response(user_input)

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received. Exiting chat.")
                print(f"\n{self.ai_name}: Exiting...") # Add print on interrupt
                break
            except Exception as e:
                self.logger.error(f"Error during chat: {e}")
                print(f"\n{self.ai_name}: Sorry, I encountered an error. Please try again.") # Add print on error

        # Saving happens incrementally in add_message, no explicit save needed here
        self.logger.info("Chat ended.")