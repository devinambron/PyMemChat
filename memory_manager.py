# memory_manager.py
import json
import logging
import re
from typing import List, Dict, Optional, Any, Union, Callable
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from exceptions import MemoryLoadError, MemorySaveError
from vector_store import create_vector_store
from utils import process_memory_data
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Import memory components from their correct locations
# REMOVED deprecated memory class imports - these will be handled differently
# try:
#     # Langchain >= 0.3.0
#     from langchain_community.memory import (
#         ConversationBufferMemory, 
#         ConversationSummaryMemory,
#         ConversationSummaryBufferMemory,
#         ConversationEntityMemory,
#         CombinedMemory,
#         VectorStoreRetrieverMemory
#     )
# except ImportError:
#     # Fallback for older versions
#     from langchain.memory import (
#         ConversationBufferMemory, 
#         ConversationSummaryMemory,
#         ConversationSummaryBufferMemory,
#         ConversationEntityMemory,
#         CombinedMemory,
#         VectorStoreRetrieverMemory
#     )

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, file_path: str, llm: Optional[ChatOpenAI] = None):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize the language model (still needed for summarization)
        self.llm = llm
        
        # Create a vector store for semantic search
        self.vector_store = create_vector_store()
        if self.vector_store:
            self.vector_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3} # Reduced k for context relevance
            )
        else:
            self.vector_retriever = None
            self.logger.warning("Vector store could not be initialized. Retrieval features will be disabled.")
        
        # Chat histories - store as raw dicts for serialization
        self.chat_history = []  # For persistently stored messages
        self.current_session_history = []  # For raw messages added this session before saving
        
        # LCEL-compatible message history object. This will be managed by RunnableWithMessageHistory.
        self.message_history = ChatMessageHistory()
        
        # REMOVED Initialization of deprecated ConversationEntityMemory
        # self.entity_memory = None # Not needed here anymore
        
        # REMOVED Initialization of deprecated ConversationSummaryBufferMemory
        # self.summary_buffer_memory = None # Not needed here anymore
        
        # Simple summary storage (if still desired for basic context)
        self.summary = ""
        
        # Flag to control whether to process and print summaries (currently unused)
        self.silent_mode = True
        
        # Status flag
        self.session_started = False
    
    # REMOVED unused chain creation methods (_create_condense_question_chain, etc.)
    # These chains are now primarily defined within the Chatbot class using LCEL directly.

    def load_memory(self) -> None:
        """Load memory from file storage and populate the message_history object."""
        self.logger.debug(f"Loading memory from file: {self.file_path}")
        try:
            # Reset histories before loading
            self.chat_history = []
            self.current_session_history = []
            self.message_history.clear()
            memory_data = []

            try:
                with open(self.file_path, 'r') as f:
                    memory_data = json.load(f)
                self.chat_history = memory_data # Store raw loaded data
                self.logger.info(f"Loaded {len(self.chat_history)} raw messages from {self.file_path}")
            except FileNotFoundError:
                self.logger.warning(f"Memory file {self.file_path} not found, initializing empty memory.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding memory file {self.file_path}: {e}")
                # Optionally, create a backup or handle the corrupted file
                # For now, proceed with empty memory
            
            # Populate vector store with messages from loaded history
            if memory_data and self.vector_store:
                self.logger.debug("Populating vector store with existing messages")
                documents = []
                for msg in memory_data:
                    content = msg.get('content')
                    # Add only user/AI messages with decent length to vector store
                    if msg.get('role') in ['user', 'ai'] and content and isinstance(content, str) and len(content) > 10:
                        # TODO: Consider adding metadata (timestamp, role) to Document?
                        documents.append(Document(page_content=content))
                if documents:
                    self.logger.debug(f"Adding {len(documents)} documents to vector store")
                    try:
                        self.vector_store.add_documents(documents)
                    except Exception as e:
                        self.logger.error(f"Error adding documents to vector store: {e}")
            
            # Populate message_history object for LCEL from loaded raw data
            processed_messages = process_memory_data(memory_data)
            if processed_messages:
                self.logger.debug(f"Populating message_history with {len(processed_messages)} BaseMessage objects")
                self.message_history.add_messages(processed_messages)
                # Note: Entity extraction from history is removed here. 
                # If needed, it would require a separate process or integration elsewhere.
            
            self.logger.debug(f"Memory loaded: {len(self.chat_history)} raw messages, {len(self.message_history.messages)} BaseMessages in history object.")
            
            # Simple summary flag (no actual summary generation here)
            if memory_data:
                 self.summary = "Previous conversations loaded."
            else:
                self.summary = ""
            
        except Exception as e:
            self.logger.error(f"Unexpected error loading memory: {e}", exc_info=True)
            # Reset to safe state
            self.chat_history = []
            self.current_session_history = []
            self.message_history.clear()
            self.summary = ""
            # Re-raise or handle as appropriate? For now, log and continue.
            
        finally:
            self.silent_mode = False # Allow normal operation
            self.session_started = True
            
    def create_and_store_session_summary(self, messages: List[BaseMessage]) -> None:
        """Generates a summary of the provided messages using the LLM and stores it in the vector store."""
        if not self.llm:
            self.logger.warning("LLM is not available, cannot generate session summary.")
            return
        if not self.vector_store:
            self.logger.warning("Vector store is not available, cannot store session summary.")
            return
        if not messages:
            self.logger.info("No messages provided to summarize.")
            return
            
        self.logger.debug(f"Starting summary generation for {len(messages)} messages.")
        
        # Format messages for the prompt
        # Use type attribute for role
        formatted_messages = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages])
        
        # Define summarization prompt
        summarization_prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert in summarizing conversations. Analyze the following conversation transcript. "
             "Extract key facts learned about the user (e.g., name, specific preferences like favorite color, stated goals, significant life events mentioned), "
             "and identify the main topics discussed. **Synthesize related information; for example, if the user mentions details about their location multiple times, provide a single consolidated fact.** "
             "Generate a concise summary focusing *only* on information crucial for remembering the user "
             "and maintaining context in future interactions. Structure the output clearly, perhaps using bullet points for facts/preferences. "
             "Do not include conversational fluff. If no significant new information was revealed, state that clearly."
            ),
            ("human", "Conversation Transcript:\n---\n{conversation_text}\n---\n\nConcise Summary for Future Recall:")
        ])
        
        # Create summarization chain
        summarization_chain = summarization_prompt_template | self.llm | StrOutputParser()
        
        try:
            # Invoke the chain
            summary_text = summarization_chain.invoke({"conversation_text": formatted_messages})
            self.logger.info(f"Generated session summary: {summary_text[:200]}...")
            
            if summary_text and "no significant new information" not in summary_text.lower():
                # Create a Document for the vector store
                # Add metadata, e.g., timestamp (optional)
                from datetime import datetime
                summary_doc = Document(
                    page_content=f"Summary of conversation ending around {datetime.now().strftime('%Y-%m-%d %H:%M')}:\n{summary_text}",
                    metadata={"source": "session_summary", "timestamp": datetime.now().isoformat()}
                )
                
                # Add to vector store
                if self.vector_store:
                    self.vector_store.add_documents([summary_doc])
                    self.logger.info("Session summary added to vector store.")
                else:
                    self.logger.warning("Vector store not available, cannot add summary document.")
            else:
                self.logger.info("Summary deemed not significant enough or empty; not adding to vector store.")
                
        except Exception as e:
            self.logger.error(f"Error during summary generation or storage: {e}", exc_info=True)

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the raw chat history list (for persistent saving)."""
        # This method ONLY updates the list used for JSON saving.
        # It does NOT update self.message_history (LCEL object) - that's handled by RunnableWithMessageHistory.
        try:
            if not isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
                self.logger.warning(f"Attempted to add unsupported message type: {type(message)}")
                return

            message_dict = {
                "role": message.type, # Use .type for role consistently
                "content": message.content
            }
            # Add to the list that gets saved to JSON
            self.current_session_history.append(message_dict)
            self.logger.debug(f"Added message to current raw session history: Role={message.type}, Content='{message.content[:50]}...'")

            # Add raw message content to vector store (consider relevance/length filters)
            # Avoid adding short greetings or system messages unless desired.
            if self.vector_store and isinstance(message.content, str) and len(message.content) > 20 and not isinstance(message, SystemMessage):
                try:
                    # Consider adding metadata here too
                    doc = Document(page_content=message.content, metadata={"role": message.type})
                    self.vector_store.add_documents([doc])
                    self.logger.debug(f"Added raw message content to vector store: {message.content[:50]}...")
                except Exception as e:
                    self.logger.warning(f"Failed to add raw message content to vector store: {e}")

            # Save the raw history to JSON incrementally (optional, could be done at end of session)
            # self.save_memory() # Uncomment if incremental saving is desired
        except Exception as e:
            self.logger.error(f"Error in add_message: {e}", exc_info=True)

    def save_memory(self) -> None:
        """Save the raw chat history (past + current session) to a file."""
        # This saves the combined history for persistence between application runs.
        self.logger.debug(f"Saving raw memory to file: {self.file_path}")
        try:
            # Combine previously loaded history with the raw history from the current session
            combined_history = self.chat_history + self.current_session_history
            
            with open(self.file_path, 'w') as f:
                json.dump(combined_history, f, indent=2)
            self.logger.info(f"Raw memory saved successfully to {self.file_path}: {len(combined_history)} total messages.")
            # Optional: Clear current_session_history after saving if combining happens elsewhere
            # self.current_session_history = [] 
        except Exception as e:
            self.logger.error(f"Error saving raw memory to file {self.file_path}: {e}")
            # Decide whether to raise MemorySaveError or just log
            # raise MemorySaveError(f"Error saving raw memory to file: {e}")

    # REMOVED add_to_vector_store - consolidation into add_message/load_memory

    def get_chat_history(self) -> List[BaseMessage]:
        """Get the current session chat history from the message_history object."""
        # This now correctly returns the list of BaseMessage objects held by
        # the ChatMessageHistory instance managed by RunnableWithMessageHistory.
        return self.message_history.messages