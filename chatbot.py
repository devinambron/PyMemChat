# chatbot.py
import logging
import os
import uuid # For session IDs
from typing import List, Optional, Dict, Any
from operator import itemgetter
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from memory_manager import MemoryManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from utils import process_memory_data, sanitize_user_input, setup_logging
from exceptions import APICallError, MemoryLoadError, MemorySaveError
import re
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Chatbot:
    # Now takes a configured MemoryManager instance
    def __init__(self, memory_manager: MemoryManager, user_name: str = "User", ai_name: str = "Ava", verbose: bool = False, streaming: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Set chatbot's logger level ONLY if verbose is explicitly True
        # Otherwise, it will inherit the level set by setup_logging (e.g., WARNING)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
            
        # Load API key from environment
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            self.logger.error("OpenAI API key not found in environment variables.")
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        # LLM Parameters
        self.llm_temperature = 0.7
        self.llm_top_p = 1.0
        self.llm_top_k = 50 # Note: Top K might not be directly used by ChatOpenAI
        self.llm_model_name = "gpt-4o-mini" # Using a faster model for testing
        
        self.streaming = streaming
        # Use logger.debug for this potentially noisy log message
        self.logger.debug(f"Initializing LLM: {self.llm_model_name} with temp={self.llm_temperature}, top_p={self.llm_top_p}, streaming={streaming}")
        
        # LLM configuration with optional streaming
        llm_kwargs = {
            "model_name": self.llm_model_name,
            "openai_api_key": openai_api_key,
            "temperature": self.llm_temperature,
            "top_p": self.llm_top_p, # Pass top_p directly
            # model_kwargs can be used for less common parameters if needed
            # "model_kwargs": {}
        }
        
        # Add streaming if requested
        if streaming:
            llm_kwargs["streaming"] = True
            llm_kwargs["callbacks"] = [StreamingStdOutCallbackHandler()]
            
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # --- Create Sentiment Analysis Chain --- 
        # Use a separate, non-streaming LLM instance for sentiment to avoid callback interference
        sentiment_llm = ChatOpenAI(
            model_name=self.llm_model_name, # Can use the same model or a cheaper/faster one
            openai_api_key=openai_api_key,
            temperature=0.0, # Low temp for deterministic sentiment
            streaming=False # Ensure this is False
        )
        sentiment_prompt = ChatPromptTemplate.from_template(
            "Analyze the sentiment of the following text. Respond with only one word: \'positive\', \'negative\', or \'neutral\'.\n\nText: {user_input}"
        )
        # Define the chain: Input dict -> Prompt -> Sentiment LLM -> String Output
        self.sentiment_chain = (
            sentiment_prompt 
            | sentiment_llm 
            | StrOutputParser()
        )
        self.logger.debug("Sentiment analysis chain created with separate LLM instance.")
        # -------------------------------------
        
        self.user_name = user_name
        self.ai_name = ai_name
        self.memory_manager = memory_manager # Use the passed-in manager

        # --- Core Chat Chain (without history management) --- 
        # This defines the logic for a single turn, given context and question
        core_chat_chain = self._create_core_chat_chain()

        # --- Chain with History --- 
        # Wrap the core chain with history management
        self.chain_with_history = RunnableWithMessageHistory(
            core_chat_chain, 
            # Function to retrieve message history based on session_id
            # Uses the message_history object from the MemoryManager instance
            lambda session_id: self.memory_manager.message_history, 
            input_messages_key="question", 
            history_messages_key="chat_history", # This MUST match the placeholder name in the prompt
            # output_messages_key="answer" # Optional: If set, AIMessage(content=...) is stored automatically
        )

        # Use logger.debug here as well
        self.logger.debug(f"Initialized {self.ai_name} chatbot with history management.")
    
    def _create_core_chat_chain(self):
        """Creates the core LCEL chain (without message history wrapper)."""
        self.logger.debug("Creating core chat chain with RAG and LLM-based sentiment analysis...")

        system_prompt = self._get_system_prompt()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\n[Background Knowledge & Profile Notes]\n{retrieved_context}\n\n[User Sentiment: {user_sentiment}]\n\n[Current Conversation History]"),
            MessagesPlaceholder(variable_name="chat_history"), # Placeholder for history (managed by RunnableWithMessageHistory)
            ("human", "{question}") # Placeholder for the user's input
        ])

        def format_docs(docs):
            if not docs:
                return "No relevant background knowledge found."
            return "\n\n".join(doc.page_content for doc in docs)

        # Define the sequence of operations for a single turn
        # Note: chat_history is now implicitly managed by the wrapper
        core_chain = (
            RunnablePassthrough.assign(
                # Retrieve context using the vector retriever from memory_manager
                retrieved_context=(
                    itemgetter("question") |
                    (self.memory_manager.vector_retriever if self.memory_manager.vector_retriever else RunnableLambda(lambda x: "No vector retriever available.")) | # Handle case where retriever might be None
                    RunnableLambda(format_docs)
                ),
                # Analyze sentiment using the dedicated LLM chain
                user_sentiment=RunnableLambda(
                    lambda x: {"user_input": x["question"]}
                 ) | self.sentiment_chain,
                # question is passed through automatically
            )
            # Log input details before the prompt
            | RunnableLambda(lambda x: self.logger.debug(f"Input to prompt: { {k: v for k, v in x.items() if k != 'chat_history'} } | History: [Handled by Wrapper]") or x)
            | prompt
            | self.llm
            | StrOutputParser()
        )

        self.logger.debug("Core chat chain created.")
        return core_chain
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines the assistant's capabilities and persona."""
        # (System prompt remains largely the same, ensuring it mentions using 'chat_history')
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
    
    # Removed save_memory method - persistence is handled by MemoryManager instance via start_chat loop

    def generate_response(self, user_input: str, session_id: str) -> str:
        """Generate a response using the history-aware chain."""
        self.logger.debug(f"Generating response for session '{session_id}' to: {user_input}")
        
        # Prepare config for RunnableWithMessageHistory
        config = RunnableConfig(configurable={"session_id": session_id})
        
        try:
            # Invoke the chain with history management
            # The wrapper handles loading history, passing it to the core chain,
            # and saving the human input and AI output to the history object.
            response = self.chain_with_history.invoke(
                {"question": user_input}, 
                config=config
            )
            
            # Manually add raw messages to MemoryManager for JSON persistence
            # This is separate from the history object used by the chain itself
            human_message = HumanMessage(content=user_input)
            ai_message = AIMessage(content=response)
            self.memory_manager.add_message(human_message)
            self.memory_manager.add_message(ai_message)
            
            self.logger.debug(f"Generated response for session '{session_id}': {response[:100]}...")
            
            # Only print to console if not using streaming output
            if not self.streaming:
                print(f"{self.ai_name}: {response}") 
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response for session '{session_id}': {e}", exc_info=True)
            error_message = "I'm sorry, I encountered an error generating a response. Please try again."
            
            # Add error message to persistent storage via MemoryManager
            try:
                error_ai_message = AIMessage(content=error_message)
                self.memory_manager.add_message(error_ai_message)
            except Exception as mem_e:
                 self.logger.error(f"Failed to add error message to memory: {mem_e}")

            print(f"{self.ai_name}: {error_message}") # Print error message
            return error_message

# --- Interactive Chat Loop (Example Usage - consider moving to main.py or similar) ---
def start_chat(verbose: bool = False, model_name: str = "gpt-4o-mini", user_name: str = "User", streaming: bool = True):
    """Starts the interactive chat loop."""
    ai_name = "Ava"
    session_id = str(uuid.uuid4()) # Generate a unique session ID
    memory_file = "chat_memory.json"

    # Use logger configured in __init__ or root
    logger.debug(f"Starting new chat session: {session_id}") # Changed from INFO to DEBUG
    
    # 1. Initialize Memory Manager
    # LLM is needed for summarization, provide it if available
    # Load API key here if MemoryManager needs its own LLM instance
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm_for_memory = None
    if openai_api_key:
        # Use a cheaper/faster model for background tasks like summarization if possible
        llm_for_memory = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.0) 
    else:
        logger.warning("OpenAI API key not found, summarization features will be disabled in MemoryManager.")

    memory_manager = MemoryManager(file_path=memory_file, llm=llm_for_memory)
    
    # 2. Load existing memory (populates message_history and vector store)
    try:
        memory_manager.load_memory()
        logger.debug("Memory loaded successfully.") # Changed from INFO to DEBUG
    except MemoryLoadError as e:
        logger.error(f"Failed to load memory: {e}")
        # Proceeding with empty memory

    # 3. Initialize Chatbot with the memory manager
    # Model name passed here is primarily for display/logging now, LLM is initialized within Chatbot
    chatbot = Chatbot(
        memory_manager=memory_manager, 
        user_name=user_name, 
        ai_name=ai_name, 
        verbose=verbose, 
        streaming=streaming # This controls LLM init behavior
    )
    
    # Use the actual model name from the chatbot instance for the log
    logger.debug(f"Chatbot initialized. Using model: {chatbot.llm_model_name}") # Changed from INFO to DEBUG
    
    print(f"\n{ai_name}: Hello {user_name}! I'm ready to chat. (Session: {session_id}). Type 'exit' to end.")

    # --- Main Loop ---
    try:
        while True:
            user_input = input(f"{user_name}: ")
            if user_input.lower() == 'exit':
                logger.debug("Exit command received. Processing end-of-session tasks...") # Changed from INFO to DEBUG
                break # Exit the loop
            
            # Generate response using the history-aware chain
            # Pass the current session_id
            _ = chatbot.generate_response(user_input, session_id=session_id)
            
            # Ensure a newline after streaming output before the next input prompt
            if chatbot.streaming:
                print() # Add a newline for better formatting after streaming

    finally:
        # --- End of Session Tasks ---
        logger.debug("Performing end-of-session tasks...") # Changed from INFO to DEBUG
        # 1. Save the raw memory log
        try:
            memory_manager.save_memory()
            logger.debug("Raw memory log saved.") # Changed from INFO to DEBUG
        except MemorySaveError as e:
            logger.error(f"Failed to save memory log: {e}")
            
        # 2. Create and store session summary (uses the history from memory_manager.message_history)
        try:
            session_messages = memory_manager.get_chat_history() # Get messages from the ChatMessageHistory object
            if session_messages:
                logger.debug(f"Creating and storing summary for {len(session_messages)} messages from session {session_id}...") # Changed from INFO to DEBUG
                memory_manager.create_and_store_session_summary(session_messages)
                logger.debug("Session summary processed and potentially stored.") # Changed from INFO to DEBUG
            else:
                logger.debug("No messages in session history to summarize.") # Changed from INFO to DEBUG
        except Exception as e:
            logger.error(f"Error during session summary processing: {e}", exc_info=True)
            
        logger.debug(f"Chat session {session_id} ended.") # Changed from INFO to DEBUG

# Example of how to run this if executed directly
# if __name__ == '__main__':
#     # Setup logging here if running as main script
#     setup_logging(level=logging.INFO) 
#     start_chat(verbose=False) # Set verbose=True for more detailed logs