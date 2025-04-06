# memory_manager.py
import json
import logging
from typing import List, Dict, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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
        ConversationEntityMemory,
        CombinedMemory,
        VectorStoreRetrieverMemory
    )
except ImportError:
    # Fallback for older versions
    from langchain.memory import (
        ConversationBufferMemory, 
        ConversationSummaryMemory,
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
            search_kwargs={"k": 5}
        )
        
        # Chat histories
        self.chat_history = []
        self.current_session_history = []
        
        # Initialize memory components if we have an LLM
        if llm:
            # Initialize Langchain memory components
            self.memory_histories = ChatMessageHistory()
            
            # Define consistent input and output keys
            self.input_key = "input"
            self.output_key = "output"
            
            # Create buffer memory for raw conversation history
            self.buffer_memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key=self.input_key,
                output_key=self.output_key,
                chat_memory=self.memory_histories,
                return_messages=True
            )
            
            # Create summary memory for conversation summaries
            self.summary_memory = ConversationSummaryMemory(
                llm=self.llm,
                input_key=self.input_key,
                output_key=self.output_key,
                memory_key="summary",
                return_messages=True
            )
            
            # Create entity memory for tracking entities
            self.entity_memory = ConversationEntityMemory(
                llm=self.llm,
                input_key=self.input_key,
                output_key=self.output_key,
                memory_key="entities",
                return_messages=True,
                k=5  # Store details about the 5 most recent entities
            )
            
            # Create vector store memory for semantic search
            self.vector_memory = VectorStoreRetrieverMemory(
                retriever=self.vector_retriever,
                input_key=self.input_key,
                memory_key="relevant_documents",
                return_messages=True
            )
            
            # Combine memories
            self.combined_memory = CombinedMemory(
                memories=[
                    self.buffer_memory,
                    self.summary_memory,
                    self.entity_memory,
                    self.vector_memory
                ]
            )
            
            # Initialize specialized chains
            self.condense_question_chain = self._create_condense_question_chain(llm)
            self.entity_extraction_chain = self._create_entity_extraction_chain(llm)
            self.summary_chain = self._create_summary_chain(llm)
            
        self.summary = ""
        self.session_started = False
    
    def _create_condense_question_chain(self, llm: ChatOpenAI):
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
        
        return condense_q_prompt | llm | StrOutputParser()
    
    def _create_entity_extraction_chain(self, llm: ChatOpenAI):
        """Create a chain that extracts and tracks entities from conversations"""
        entity_system_prompt = """Extract and summarize information about entities (people, places, concepts) 
mentioned in the conversation. Return a JSON-formatted string with entity names as keys and their descriptions as values.
Focus only on the most important details for each entity. If no entities are present, return an empty JSON object."""
        
        entity_prompt = ChatPromptTemplate.from_messages([
            ("system", entity_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
        ])
        
        return entity_prompt | llm | StrOutputParser()
    
    def _create_summary_chain(self, llm: ChatOpenAI):
        """Create a chain that summarizes the conversation"""
        summary_system_prompt = """Progressively summarize the conversation provided, 
adding onto the previous summary and adding new information from the new messages.
If there's no previous summary, create a new summary. Keep the summary concise."""
        
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", summary_system_prompt),
            ("human", "Previous summary: {prev_summary}\n\nNew messages:\n{new_messages}\n\nNew summary:")
        ])
        
        return summary_prompt | llm | StrOutputParser()

    def load_memory(self) -> None:
        """Load memory from file storage"""
        self.logger.debug(f"Loading memory from file: {self.file_path}")
        try:
            # Check if file exists first
            try:
                with open(self.file_path, 'r') as f:
                    memory_data = json.load(f)
                self.chat_history = memory_data
                self.current_session_history = []  # Start with empty current session
            except FileNotFoundError:
                message = "Memory file not found, initializing empty memory."
                self.logger.warning(message)
                self.chat_history = []
                self.current_session_history = []
                return
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding memory file: {e}")
                raise MemoryLoadError(f"Error decoding memory file: {e}")
            
            # Populate memory components with existing messages
            if memory_data and len(memory_data) > 0:
                # Convert to BaseMessage format
                messages = process_memory_data(memory_data)
                
                # Populate vector store with messages
                if self.vector_store:
                    self.logger.debug("Populating vector store with existing messages")
                    documents = []
                    for msg in memory_data:
                        if msg.get('role') in ['user', 'ai'] and len(msg.get('content', '')) > 10:
                            documents.append(Document(page_content=msg.get('content', '')))
                    
                    if documents:
                        self.logger.debug(f"Adding {len(documents)} documents to vector store")
                        self.vector_store.add_documents(documents)
                
                # Populate memory components
                if self.llm:
                    # Reset memory components
                    self.memory_histories = ChatMessageHistory()
                    
                    # Process the messages in pairs to preserve conversation context
                    for i in range(0, len(messages)-1, 2):
                        if i+1 < len(messages):
                            if messages[i].type == "human" and messages[i+1].type == "ai":
                                # Add human message
                                self.memory_histories.add_user_message(messages[i].content)
                                # Add AI message
                                self.memory_histories.add_ai_message(messages[i+1].content)
                                
                                # Add to memory components as a conversation pair
                                input_message = messages[i].content
                                output_message = messages[i+1].content
                                
                                # Add to buffer memory
                                self.buffer_memory.save_context(
                                    {self.input_key: input_message}, 
                                    {self.output_key: output_message}
                                )
                                
                                # Add to summary memory
                                self.summary_memory.save_context(
                                    {self.input_key: input_message}, 
                                    {self.output_key: output_message}
                                )
                                
                                # Add to entity memory
                                self.entity_memory.save_context(
                                    {self.input_key: input_message}, 
                                    {self.output_key: output_message}
                                )
                
                # Initialize summary from loaded messages if we have enough history
                if memory_data and len(memory_data) >= 3 and self.llm:
                    self._initialize_summary()
                    
            # Log success
            self.logger.debug(f"Memory loaded successfully: {len(memory_data)} messages")
                
        except Exception as e:
            self.logger.error(f"Unexpected error loading memory: {e}")
            self.chat_history = []
            self.current_session_history = []
    
    def _initialize_summary(self) -> None:
        """Initialize conversation summary from existing chat history"""
        if not self.chat_history or not self.llm:
            return
            
        try:
            # Format the first few messages
            initial_messages = self.chat_history[:min(5, len(self.chat_history))]
            messages_formatted = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in initial_messages
            ])
            
            system_prompt = """Summarize the beginning of this conversation concisely.
Focus on facts and important details mentioned by the user or assistant.
Keep the summary concise and focused on what was actually discussed."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", f"Conversation beginning:\n{messages_formatted}\n\nSummary:")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            summary = chain.invoke({})
            
            # Update the stored summary
            self.summary = summary
            self.logger.debug(f"Initialized conversation summary: {summary}")
        except Exception as e:
            self.logger.warning(f"Error initializing summary: {e}")

    def save_memory(self) -> None:
        """Save the current chat history to a file"""
        self.logger.debug(f"Saving memory to file: {self.file_path}")
        try:
            # Combine past chat history with current session
            combined_history = self.chat_history + self.current_session_history
            
            with open(self.file_path, 'w') as f:
                json.dump(combined_history, f, indent=2)
            self.logger.debug(f"Memory saved successfully: {len(combined_history)} messages")
        except Exception as e:
            self.logger.error(f"Error saving memory to file: {e}")
            raise MemorySaveError(f"Error saving memory to file: {e}")

    def add_to_vector_store(self, messages: List[BaseMessage]) -> None:
        """Add messages to vector store for semantic retrieval"""
        for msg in messages:
            self.logger.debug(f"Adding message to vector store: {msg.content}")
            self.vector_store.add_texts([msg.content])

    def get_context_from_question(self, question: str) -> List[BaseMessage]:
        """
        Get context relevant to a specific question.
        
        Args:
            question: The user's question
            
        Returns:
            List of system messages containing relevant context
        """
        messages = []
        
        # Skip for simple greetings
        if question.lower() in ["hi", "hello", "hey", "what's up"]:
            return messages
        
        try:
            # 1. Get summary of conversation so far
            conversation_summary = self._summarize_conversation()
            if conversation_summary:
                messages.append(SystemMessage(content=f"Current conversation summary: {conversation_summary}"))
                
            # 2. Extract entity information
            entity_information = self._extract_entities()
                
            # 3. Retrieve relevant documents
            # First condense the question if it references previous context
            condensed_question = self._condense_question(question)
            self.logger.debug(f"Condensed question: {condensed_question}")
            
            relevant_docs = []
            
            try:
                # Retrieve documents from vector store
                if condensed_question and len(condensed_question) > 3:
                    retrieved_docs = self.vector_retriever.invoke(condensed_question)
                    relevant_docs = [doc.page_content for doc in retrieved_docs]
            except Exception as e:
                self.logger.warning(f"Error retrieving documents: {e}")
            
            # Add entity information if available and not empty
            if entity_information and not entity_information.startswith("{}"):
                try:
                    entities_dict = json.loads(entity_information)
                    if entities_dict:
                        # Format entities as background information
                        entities_msg = "The following entities have been mentioned:"
                        for entity, description in entities_dict.items():
                            entities_msg += f"\n- {entity}: {description}"
                        messages.append(SystemMessage(content=entities_msg))
                except Exception as e:
                    self.logger.warning(f"Error formatting entity information: {e}")
            
            # Add retrieved documents if available
            if relevant_docs:
                # Format as relevant information from past conversations
                docs_msg = "Previous conversations contained the following relevant information:\n" + "\n---\n".join(relevant_docs)
                messages.append(SystemMessage(content=docs_msg))
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error getting context: {e}")
            return []

    def _condense_question(self, question: str) -> str:
        """
        Condense a question that might reference chat history into a standalone question.
        
        Args:
            question: The user's original question
            
        Returns:
            A standalone version of the question
        """
        if not self.chat_history or len(self.chat_history) < 2:
            return question
            
        try:
            # Only use the last few messages from chat history to provide context
            recent_messages = self.current_session_history[-6:]
            chat_history_messages = process_memory_data(recent_messages)
            
            # Skip for simple or short questions
            if len(question) < 10 or question.lower() in [
                "hi", "hello", "hey", "what's up", "how are you",
                "what do you think?", "why?", "how?", "really?", "go on"
            ]:
                return question
                
            # Use the condense question chain
            condensed_question = self.condense_question_chain.invoke({
                "chat_history": chat_history_messages,
                "question": question
            })
                
            if ":" in condensed_question and condensed_question.startswith('"'):
                # Remove quotation marks and any explanatory text before the question
                condensed_question = condensed_question.split(":", 1)[-1].strip().strip('"')
                
            return condensed_question
        except Exception as e:
            self.logger.warning(f"Error condensing question: {e}")
            return question

    def _extract_entities(self) -> str:
        """
        Extract entity information from the chat history.
        
        Returns:
            A JSON string containing entity information
        """
        if not self.llm:
            return "{}"
            
        try:
            # First try to get entities from the entity memory
            if hasattr(self, 'entity_memory') and self.entity_memory:
                try:
                    # Get entities from the memory component
                    entity_store = getattr(self.entity_memory, 'entity_store', None)
                    if entity_store:
                        entities = entity_store.store
                        if entities:
                            return json.dumps(entities)
                except Exception as e:
                    self.logger.warning(f"Error accessing entity memory: {e}")
            
            # If that fails or returns empty, use our extraction chain
            # Only process the last few messages
            recent_messages = self.current_session_history[-6:] if self.current_session_history else self.chat_history[-6:]
            if not recent_messages:
                return "{}"
                
            chat_history_messages = process_memory_data(recent_messages)
            
            system_prompt = """Extract and summarize information about entities (people, places, concepts) 
mentioned in the conversation. Return a JSON-formatted string with entity names as keys and their descriptions as values.
Focus only on the most important details for each entity. If no entities are present, return an empty JSON object.
Only include actual entities that have been mentioned with certainty. Do not include potential or hypothetical entities."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                *[(msg.type, msg.content) for msg in chat_history_messages]
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            entities = chain.invoke({})
            
            # Log but don't include in response
            self.logger.debug(f"Extracted entity information: {entities}")
            return entities
        except Exception as e:
            self.logger.warning(f"Error extracting entities: {e}")
            return "{}"

    def _summarize_conversation(self) -> str:
        """
        Create or update a summary of the conversation.
        
        Returns:
            A string containing a summary of the conversation
        """
        if not self.llm:
            return ""
            
        try:
            # First try to get summary from the summary memory
            if hasattr(self, 'summary_memory') and self.summary_memory:
                try:
                    memory_variables = self.summary_memory.load_memory_variables({})
                    if memory_variables and "summary" in memory_variables:
                        summary_content = memory_variables["summary"]
                        if isinstance(summary_content, str) and summary_content:
                            return summary_content
                        elif isinstance(summary_content, list) and summary_content:
                            # If it's a list of messages, extract content
                            summary_texts = [msg.content for msg in summary_content if hasattr(msg, 'content')]
                            if summary_texts:
                                return "\n".join(summary_texts)
                except Exception as e:
                    self.logger.warning(f"Error accessing summary memory: {e}")
            
            # If we have a stored summary and recent messages, update it
            if self.summary:
                # Get most recent messages
                if not self.current_session_history:
                    return self.summary
                    
                # Format recent messages
                recent_msgs = self.current_session_history[-3:]
                msgs_formatted = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in recent_msgs
                ])
                
                # Use summarization chain to update
                new_summary = self.summary_chain.invoke({
                    "prev_summary": self.summary,
                    "new_messages": msgs_formatted
                })
                
                # Update stored summary
                self.summary = new_summary
                return new_summary
            
            # If no summary exists, create one from scratch
            if self.current_session_history:
                # Use summarization chain to create new summary
                messages = process_memory_data(self.current_session_history)
                if not messages:
                    return ""
                    
                system_prompt = """Provide a concise summary of this conversation.
Focus on key points, facts, and information shared."""
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    *[(msg.type, msg.content) for msg in messages]
                ])
                
                chain = prompt | self.llm | StrOutputParser()
                new_summary = chain.invoke({})
                
                # Update stored summary
                self.summary = new_summary
                return new_summary
                
            return ""
        except Exception as e:
            self.logger.warning(f"Error summarizing conversation: {e}")
            return ""

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the chat history.
        
        Args:
            message: The message to add
        """
        # Convert to dict representation for storage
        message_dict = {"role": "user" if isinstance(message, HumanMessage) else "ai" 
                      if isinstance(message, AIMessage) else "system", 
                      "content": message.content}
        
        # Add to current session history
        self.current_session_history.append(message_dict)
        self.logger.debug(f"Added message to current session: {message.content}")
        
        # Add to Langchain memory components
        if self.llm:
            # Add message to appropriate memory component
            if isinstance(message, HumanMessage):
                self.memory_histories.add_user_message(message.content)
                # Since this is a user message, we don't save it to memory components yet
                # We save it after the AI responds as a conversation pair
            elif isinstance(message, AIMessage):
                self.memory_histories.add_ai_message(message.content)
                # Find the most recent user message to create a conversation pair
                user_message = ""
                for i in range(len(self.current_session_history)-2, -1, -1):
                    if self.current_session_history[i]["role"] == "user":
                        user_message = self.current_session_history[i]["content"]
                        break
                
                if user_message:
                    # Add the conversation pair to all memory components
                    input_dict = {self.input_key: user_message}
                    output_dict = {self.output_key: message.content}
                    
                    # Add to buffer memory
                    self.buffer_memory.save_context(input_dict, output_dict)
                    
                    # Add to entity memory
                    self.entity_memory.save_context(input_dict, output_dict)
                    
                    # Add to summary memory
                    self.summary_memory.save_context(input_dict, output_dict)
            
        # Add to vector store if it's a substantive message
        if len(message.content) > 10 and not isinstance(message, SystemMessage) and not message.content.lower().startswith(("hi", "hello")):
            try:
                if self.vector_store:
                    self.vector_store.add_documents([
                        Document(page_content=message.content)
                    ])
                    
                    # Also update vector memory
                    if hasattr(self, 'vector_memory'):
                        self.vector_memory.retriever = self.vector_store.as_retriever(
                            search_kwargs={"k": 5}
                        )
            except Exception as e:
                self.logger.warning(f"Failed to add message to vector store: {e}")
            
        # Save memory after each message
        self.save_memory()

    def get_chat_history(self) -> List[BaseMessage]:
        """
        Get the current session chat history as a list of BaseMessage objects.
        
        Returns:
            A list of BaseMessage objects
        """
        # Only return the current session history to avoid repeating past conversations
        return process_memory_data(self.current_session_history)

    def add_message_to_history(self, message: BaseMessage) -> None:
        """Add a message to the chat history"""
        message_dict = {"role": "user" if isinstance(message, HumanMessage) else "ai" 
                      if isinstance(message, AIMessage) else "system", 
                      "content": message.content}
        
        self.chat_history.append(message_dict)
        self.logger.debug(f"Added message to chat history: {message.content}")
        
        # Also add to vector store for future retrieval
        if not isinstance(message, SystemMessage) and len(message.content) > 10:
            self.vector_store.add_texts([message.content])