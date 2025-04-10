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
from utils import process_memory_data, sanitize_user_input, setup_logging, configure_memory_retriever
from exceptions import APICallError, MemoryLoadError, MemorySaveError
import re
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Placeholder for sentiment analysis function (Task 4)
def analyze_sentiment(text: str) -> str:
    """Placeholder function to analyze sentiment. Returns 'positive', 'negative', or 'neutral'."""
    # TODO: Replace with actual sentiment analysis logic (e.g., NLTK VADER, HF model)
    text_lower = text.lower()
    if "sad" in text_lower or "upset" in text_lower or "bad" in text_lower or "terrible" in text_lower:
        return "negative"
    elif "happy" in text_lower or "great" in text_lower or "good" in text_lower or "awesome" in text_lower:
        return "positive"
    else:
        return "neutral"

class Chatbot:
    def __init__(self, user_name: str = "User", ai_name: str = "Ava", verbose: bool = False):
        self.logger = logging.getLogger(__name__)
        
        # Set up logging with appropriate verbosity
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            
        # Load API key from environment
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            self.logger.error("OpenAI API key not found in environment variables.")
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        # LLM Parameters (configurable)
        # Temperature: Controls randomness. Lower is more deterministic, higher is more creative.
        # Top_p: Nucleus sampling. Considers only tokens with cumulative probability mass up to top_p.
        # Top_k: Considers only the top_k most likely tokens.
        # Recommendation: Adjust temperature primarily, or use top_p, but generally not both temperature and top_p significantly > 0.
        self.llm_temperature = 0.7 # Default: Balance creativity/coherence
        self.llm_top_p = 1.0       # Default: No nucleus sampling unless adjusted
        self.llm_top_k = 50        # Default: Consider top 50 tokens
        self.llm_model_name = "gpt-4o-mini" # Or your preferred model

        self.logger.info(f"Initializing LLM: {self.llm_model_name} with temp={self.llm_temperature}, top_p={self.llm_top_p}, top_k={self.llm_top_k}")
        self.llm = ChatOpenAI(
            model_name=self.llm_model_name,
            openai_api_key=openai_api_key,
            temperature=self.llm_temperature,
            model_kwargs={
                "top_p": self.llm_top_p,
                # Note: top_k is often not a direct parameter for OpenAI models via Langchain, controlled via top_p/temp primarily.
                # We include self.llm_top_k for potential future use or other models, but it might not affect GPT-4/3.5 directly here.
            }
            # Add other parameters like max_tokens if needed
        )
        
        self.user_name = user_name
        self.ai_name = ai_name

        # --- Memory Manager --- 
        # Pass the LLM instance to the MemoryManager
        # Memory loading is now handled in start_chat()
        self.memory_manager = MemoryManager(file_path="chat_memory.json", llm=self.llm)

        # --- Chat Chain --- 
        # Create the chat chain. This now uses memory components 
        # initialized within MemoryManager (e.g., summary_buffer_memory).
        self.chat_chain = self._create_chat_chain()

        self.logger.info(f"Initialized {self.ai_name} with model {self.llm_model_name}")
    
    def _create_chat_chain(self):
        """Creates the Langchain Expression Language (LCEL) chain for the chatbot."""
        self.logger.debug("Creating chat chain with RAG and sentiment analysis...")

        # Get the base system prompt
        system_prompt = self._get_system_prompt()

        # Define the prompt template
        # Includes placeholders for history, retrieved context, and the user question
        prompt = ChatPromptTemplate.from_messages([
            # System prompt instructs how to use context from buffer and RAG
            ("system", system_prompt + "\n\n[Background Knowledge & Profile Notes]\n{retrieved_context}\n\n[User Sentiment: {user_sentiment}]\n\n[Current Conversation History]\n"),
            MessagesPlaceholder(variable_name="chat_history"), # From SummaryBufferMemory
            ("human", "{question}")
        ])

        # Helper function to format retrieved documents
        def format_docs(docs):
            # Consider adding metadata (e.g., timestamp) to the formatted string
            return "\n\n".join(doc.page_content for doc in docs)

        # Build the chat chain using RunnablePassthrough.assign to load memory components
        chat_chain = (
            RunnablePassthrough.assign(
                # Load history from the summary buffer memory
                chat_history_messages=RunnableLambda(self.memory_manager.summary_buffer_memory.load_memory_variables) | itemgetter(self.memory_manager.summary_buffer_memory.memory_key),
                question=itemgetter("question") # Pass the user question through
            )
            # Add retrieval step (RAG - Task 2B)
            .assign(
                retrieved_context=(
                    itemgetter("question") |
                    self.memory_manager.vector_store.as_retriever(search_kwargs=dict(k=3)) | # Retrieve top 3 relevant docs
                    RunnableLambda(format_docs)
                )
            )
            # Ensure chat_history (message list) is available for the prompt
            .assign(
                chat_history=itemgetter("chat_history_messages")
            )
            # Add Sentiment Analysis step (Task 4)
            .assign(
                user_sentiment=itemgetter("question") | RunnableLambda(analyze_sentiment)
            )
            # Log input before it hits the prompt template
            | RunnableLambda(lambda x: self.logger.debug(f"Input to prompt: { {k: v for k, v in x.items() if k not in ['chat_history', 'chat_history_messages']} } | History: {len(x.get('chat_history', []))} messages | Retrieved Context: {x.get('retrieved_context','')[:100]}...") or x)
            | prompt
            | self.llm
            | StrOutputParser()
        )

        self.logger.debug("Chat chain created successfully with RAG and sentiment analysis steps")
        return chat_chain
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines the assistant's capabilities and persona."""
        # Persona Definition: Aiming for a supportive, curious, engaging friend.
        # Traits: Empathetic, non-judgmental, good listener, remembers details, maybe slightly humorous/playful but grounded.
        # Communication Style: Natural, conversational, uses user's name, asks follow-up questions.
        # Goals: Build rapport, remember user details, provide companionship and thoughtful conversation.
        # Boundaries: Does not claim sentience, avoids harmful/unethical content, admits limitations.
        return (
            f"You are {self.ai_name}, an AI designed to be a supportive and engaging friend. Your goal is to build rapport, provide companionship, and have thoughtful conversations. "
            f"Act as a curious, empathetic, and non-judgmental listener. Remember details the user shares about themselves (like their name, preferences, experiences) and refer back to them when relevant to show you're paying attention. "
            f"Use the user's name ({self.user_name} if known, otherwise ask politely) to make the conversation feel personal. Ask follow-up questions to understand them better. "
            f"Maintain a natural, friendly, and conversational tone. You can be slightly playful or humorous when appropriate, but prioritize being supportive. "
            f"Pay attention to the user's likely sentiment ('user_sentiment'). If it's 'negative', respond with extra empathy and support. If it's 'positive', you can share their enthusiasm. Otherwise, maintain a balanced tone. "
            f"You MUST use the provided conversation history ('chat_history') and any relevant background knowledge ('retrieved_context', if provided) to inform your responses and maintain continuity. Refer to these explicitly when asked about past conversations. "
            f"Avoid commenting on repetitive user questions or conversational loops; simply answer the current question based on the available history. "
            f"If the user's message is ambiguous, unclear, or seems to lack sufficient context for you to provide a meaningful response, ask a polite clarifying question rather than making assumptions or giving a generic answer. "
            f"If you don't know something, can't remember based on the provided context, or if a request is outside your capabilities or ethical boundaries, say so clearly and politely. Do NOT claim to be sentient or have real feelings. "
            f"Keep your responses helpful, considerate, and focused on being a good conversational partner."
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
            self.memory_manager.save_context({"input": user_input}, {"output": response})
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

    def start_chat(self):
        """Starts the interactive chat loop."""
        self.logger.info(f"Starting chat session with {self.ai_name}. Type 'exit' to end.")
        
        # Load memory at the beginning of the session
        try:
            self.memory_manager.load_memory()
            self.logger.info("Memory loaded successfully.")
        except MemoryLoadError as e:
            self.logger.error(f"Failed to load memory: {e}")
            # Decide if we should proceed with empty memory or exit
            # For now, proceed with empty memory
        
        while True:
            user_input = input(f"{self.user_name}: ")
            if user_input.lower() == 'exit':
                self.logger.info("Exit command received. Processing end-of-session tasks...")
                # --- Task 3: Post-Session Summarization ---
                try:
                    # Get all messages from the current session history managed by the buffer
                    # Note: summary_buffer_memory holds state via the linked message_history
                    session_messages = self.memory_manager.message_history.messages
                    if session_messages:
                        self.logger.info(f"Creating and storing summary for {len(session_messages)} messages...")
                        self.memory_manager.create_and_store_session_summary(session_messages)
                        self.logger.info("Session summary processed and stored.")
                    else:
                        self.logger.info("No messages in session to summarize.")
                except Exception as e:
                    self.logger.error(f"Error during post-session summarization: {e}", exc_info=True)
                # --- End Task 3 ---
                
                # Optionally, perform final save if needed (though add_message might handle incremental saves)
                # self.memory_manager.save_memory()
                self.logger.info("Chat session ended.")
                break
            
            if not user_input:
                continue
            
            try:
                # Add user message to message_history (used by memories) and persistent store
                user_message = HumanMessage(content=user_input)
                self.memory_manager.message_history.add_message(user_message)
                self.memory_manager.add_message(user_message) # Saves to JSON + vector store (raw)
                
                # Invoke chain
                response = self.chat_chain.invoke({"question": user_input})
                
                # Add AI response to message_history and persistent store
                ai_response = AIMessage(content=response)
                self.memory_manager.message_history.add_message(ai_response)
                self.memory_manager.add_message(ai_response) # Saves to JSON + vector store (raw)
                
                print(f"{self.ai_name}: {response}")
                
            except APICallError as e:
                self.logger.error(f"API Call Error: {e}")
                print("Sorry, there was an error communicating with the AI service.")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                print("Sorry, an unexpected error occurred.")