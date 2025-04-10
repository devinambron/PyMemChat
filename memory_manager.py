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
try:
    # Langchain >= 0.3.0
    from langchain_community.memory import (
        ConversationBufferMemory, 
        ConversationSummaryMemory,
        ConversationSummaryBufferMemory,
        ConversationEntityMemory,
        CombinedMemory,
        VectorStoreRetrieverMemory
    )
except ImportError:
    # Fallback for older versions
    from langchain.memory import (
        ConversationBufferMemory, 
        ConversationSummaryMemory,
        ConversationSummaryBufferMemory,
        ConversationEntityMemory,
        CombinedMemory,
        VectorStoreRetrieverMemory
    )

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, file_path: str, llm: Optional[ChatOpenAI] = None):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize the language model
        self.llm = llm
        
        # Create a vector store for semantic search
        self.vector_store = create_vector_store()
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 3} # Reduced k for context relevance
        )
        
        # Chat histories - store as raw dicts for serialization
        self.chat_history = []  # For persistently stored messages
        self.current_session_history = []  # For this session
        
        # Current session memory - store as ChatMessageHistory for LCEL compatibility
        self.message_history = ChatMessageHistory()
        
        # Initialize ConversationEntityMemory
        # Use the same message_history for shared state
        if self.llm:
            self.entity_memory = ConversationEntityMemory(
                llm=self.llm,
                chat_history=self.message_history,
                memory_key="entities", # Standard key for entity memory
                return_messages=False # Return summary string, not messages
            )
        else:
            self.entity_memory = None # No LLM, no entity memory
        
        # Initialize ConversationSummaryBufferMemory (uses message_history)
        # Configured for large context window as per Task 2A
        if self.llm:
            self.summary_buffer_memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                chat_message_history=self.message_history, # Link to the shared history
                max_token_limit=80000, # Utilize large context window
                memory_key="chat_history", # Standard key for chat history
                return_messages=True # Return BaseMessage objects for LCEL
            )
        else:
            # Provide a fallback or raise an error if LLM is needed but not provided
            self.logger.warning("LLM not provided, ConversationSummaryBufferMemory requires an LLM. Falling back to basic history or potentially erroring.")
            # Decide on fallback behavior: maybe a simple buffer or raise error
            # For now, let's set it to None, but Chatbot logic must handle this.
            self.summary_buffer_memory = None
        
        # Conversation summary (still potentially useful for simple context)
        self.summary = ""
        
        # Flag to control whether to process and print summaries
        self.silent_mode = True
        
        # Process previous messages - delayed until needed
        self.session_started = False
    
    def _create_condense_question_chain(self) -> RunnableSequence:
        """Create a chain that reformulates questions based on chat history"""
        condense_q_system_prompt = """Given a chat history and the latest user question 
which might reference the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""
        
        condense_q_prompt = ChatPromptTemplate.from_messages([
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        return condense_q_prompt | self.llm | StrOutputParser()
    
    def _create_entity_extraction_chain(self) -> RunnableSequence:
        """Create a chain that extracts entities from conversations"""
        entity_system_prompt = """Extract and summarize information about entities (people, places, concepts) 
mentioned in the conversation. Return a JSON-formatted string with entity names as keys and their descriptions as values.
Focus only on the most important details for each entity. If no entities are present, return an empty JSON object.
Pay special attention to the user's name and personal details that should be remembered across conversations."""
        
        entity_prompt = ChatPromptTemplate.from_messages([
            ("system", entity_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
        ])
        
        return entity_prompt | self.llm | StrOutputParser()
    
    def _create_summary_chain(self) -> RunnableSequence:
        """Create a chain that summarizes the conversation"""
        summary_system_prompt = """Progressively summarize the conversation provided, 
adding onto the previous summary and adding new information from the new messages.
If there's no previous summary, create a new summary. Keep the summary concise."""
        
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", summary_system_prompt),
            ("human", "Previous summary: {prev_summary}\n\nNew messages:\n{new_messages}\n\nNew summary:")
        ])
        
        return summary_prompt | self.llm | StrOutputParser()

    def load_memory(self) -> None:
        """Load memory from file storage and populate memory objects."""
        self.logger.debug(f"Loading memory from file: {self.file_path}")
        try:
            # Set silent mode to prevent automatic processing
            self.silent_mode = True
            
            # Check if file exists first
            try:
                with open(self.file_path, 'r') as f:
                    memory_data = json.load(f)
                self.chat_history = memory_data
                self.current_session_history = []  # Start with empty current session
            except FileNotFoundError:
                self.logger.warning("Memory file not found, initializing empty memory.")
                self.chat_history = []
                self.current_session_history = []
                memory_data = [] # Ensure memory_data is empty list
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding memory file: {e}")
                raise MemoryLoadError(f"Error decoding memory file: {e}")
            
            # Populate vector store with messages
            if memory_data and self.vector_store:
                self.logger.debug("Populating vector store with existing messages")
                documents = []
                for msg in memory_data:
                    # Ensure content exists and is string
                    content = msg.get('content')
                    if msg.get('role') in ['user', 'ai'] and content and isinstance(content, str) and len(content) > 10:
                        documents.append(Document(page_content=content))
                if documents:
                    self.logger.debug(f"Adding {len(documents)} documents to vector store")
                    try:
                        self.vector_store.add_documents(documents)
                    except Exception as e:
                        self.logger.error(f"Error adding documents to vector store: {e}")
            
            # Populate message history and entity memory from loaded data
            processed_messages = process_memory_data(memory_data)
            if processed_messages:
                self.logger.debug(f"Populating message history and entity memory with {len(processed_messages)} messages")
                # Clear existing histories first
                self.message_history.clear()
                if self.entity_memory:
                    self.entity_memory.clear()
                
                # Add messages sequentially to build history and entities
                for i in range(0, len(processed_messages), 2):
                    human_msg = processed_messages[i]
                    ai_msg = processed_messages[i+1] if (i+1) < len(processed_messages) else None
                    
                    self.message_history.add_message(human_msg)
                    if ai_msg:
                        self.message_history.add_message(ai_msg)
                        # Use save_context to populate entity memory from historical data
                        if self.entity_memory:
                            try:
                                # Use save_context to allow entity extraction from past messages
                                self.entity_memory.save_context(
                                    {"input": human_msg.content},
                                    {"output": ai_msg.content}
                                )
                            except Exception as e:
                                self.logger.warning(f"Error processing historical context into entity memory: {e}")
                    else:
                        # Handle case with odd number of messages (last human message)
                        if self.entity_memory:
                            try:
                                # Use save_context even for single input to potentially extract entities
                                self.entity_memory.save_context({"input": human_msg.content}, {"output": ""})
                            except Exception as e:
                                self.logger.warning(f"Error processing final human message into entity memory: {e}")
            
            self.logger.debug(f"Memory loaded successfully: {len(memory_data)} messages processed")
            
            # Summary generation logic (keep as is or replace with ConversationSummaryMemory)
            if memory_data and len(memory_data) >= 5:
                # Look for existing summary patterns in the data
                for msg in memory_data:
                    if msg.get('role') == 'system' and 'conversation summary' in msg.get('content', '').lower():
                        summary_text = msg.get('content', '')
                        if ':' in summary_text:
                            self.summary = summary_text.split(':', 1)[1].strip()
                            break
                
                # If no summary found, create a basic one without LLM calls
                if not self.summary:
                    self.summary = "Previous conversations loaded."
            
        except Exception as e:
            self.logger.error(f"Unexpected error loading memory: {e}", exc_info=True)
            self.chat_history = []
            self.current_session_history = []
            self.message_history.clear()
            if self.entity_memory:
                self.entity_memory.clear()
            
        finally:
            # Ensure silent mode is off after loading
            self.silent_mode = False
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
        formatted_messages = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages])
        
        # Define summarization prompt
        summarization_prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert in summarizing conversations. Analyze the following conversation transcript. "
             "Extract key facts learned about the user (e.g., name, specific preferences like favorite color, stated goals, significant life events mentioned), "
             "and identify the main topics discussed. Generate a concise summary focusing *only* on information crucial for remembering the user "
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
                self.vector_store.add_documents([summary_doc])
                self.logger.info("Session summary added to vector store.")
            else:
                self.logger.info("Summary deemed not significant enough or empty; not adding to vector store.")
                
        except Exception as e:
            self.logger.error(f"Error during summary generation or storage: {e}", exc_info=True)

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the raw chat history for persistent saving and update vector store (for raw messages)."""
        try:
            message_dict = {
                "role": "user" if isinstance(message, HumanMessage) else "ai" if isinstance(message, AIMessage) else "system",
                "content": message.content
            }
            # Add to the list that gets saved to JSON
            self.current_session_history.append(message_dict)
            self.logger.debug(f"Added message to current raw session history: {message.content}")

            # Add to vector store if applicable
            if self.vector_store and isinstance(message.content, str) and len(message.content) > 10 and not isinstance(message, SystemMessage) and not message.content.lower().startswith(("hi", "hello")):
                try:
                    self.vector_store.add_documents([Document(page_content=message.content)])
                    self.logger.debug(f"Added message to vector store: {message.content[:50]}...")
                except Exception as e:
                    self.logger.warning(f"Failed to add message to vector store: {e}")

            # Save the raw history to JSON incrementally
            self.save_memory()
        except Exception as e:
            self.logger.error(f"Error in add_message: {e}", exc_info=True)

    def save_memory(self) -> None:
        """Save the current raw chat history to a file."""
        # This now only saves the raw message list for persistence between runs.
        # The state of memory objects (buffer, entity) is held in memory during a session.
        self.logger.debug(f"Saving raw memory to file: {self.file_path}")
        try:
            # Combine past chat history with current session raw history
            combined_history = self.chat_history + self.current_session_history
            
            with open(self.file_path, 'w') as f:
                json.dump(combined_history, f, indent=2)
            self.logger.debug(f"Raw memory saved successfully: {len(combined_history)} messages")
        except Exception as e:
            self.logger.error(f"Error saving raw memory to file: {e}")
            raise MemorySaveError(f"Error saving raw memory to file: {e}")

    def add_to_vector_store(self, messages: List[BaseMessage]) -> None:
        """Add messages to vector store for semantic retrieval"""
        # This method might become redundant if add_message handles vector store updates.
        # Keeping it for now in case of bulk adds.
        texts_to_add = []
        for msg in messages:
            if isinstance(msg.content, str) and len(msg.content) > 10:
                self.logger.debug(f"Queueing message for vector store add: {msg.content[:50]}...")
                texts_to_add.append(msg.content)
        if texts_to_add and self.vector_store:
            try:
                self.vector_store.add_texts(texts_to_add)
                self.logger.debug(f"Added {len(texts_to_add)} messages to vector store.")
            except Exception as e:
                 self.logger.warning(f"Failed to add bulk messages to vector store: {e}")

    def get_chat_history(self) -> List[BaseMessage]:
        """Get the current session chat history from the message_history object."""
        return self.message_history.messages