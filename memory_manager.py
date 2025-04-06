# memory_manager.py
import json
import logging
from typing import List, Dict, Optional, Any
from langchain.schema import AIMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from exceptions import MemoryLoadError, MemorySaveError
from vector_store import create_vector_store

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, file_path: str, llm: Optional[ChatOpenAI] = None):
        self.file_path = file_path
        self.vector_store = create_vector_store()
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        self.llm = llm
        self.chat_history = []
        
        # Initialize the condense question chain for better contextual understanding
        if llm:
            self.condense_question_chain = self._create_condense_question_chain(llm)
            self.entity_extraction_chain = self._create_entity_extraction_chain(llm)
            self.summary_chain = self._create_summary_chain(llm)
    
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

    def load_memory(self) -> List[Dict[str, str]]:
        """Load memory from file storage"""
        logger.debug(f"Loading memory from file: {self.file_path}")
        try:
            with open(self.file_path, 'r') as f:
                memory = json.load(f)
            logger.debug(f"Memory loaded successfully: {memory}")
            return memory
        except FileNotFoundError:
            message = "Memory file not found, initializing empty memory."
            logger.warning(message)
            return []
        except json.JSONDecodeError as e:
            raise MemoryLoadError(f"Error decoding memory file: {e}")

    def save_memory(self, messages: List[BaseMessage]) -> None:
        """Save memory to file storage"""
        logger.debug(f"Saving memory to file: {self.file_path}")
        try:
            serializable_memory = [
                {"role": "user", "content": msg.content} if isinstance(msg, HumanMessage) else
                {"role": "ai", "content": msg.content}
                for msg in messages
            ]
            with open(self.file_path, 'w') as f:
                json.dump(serializable_memory, f)
            logger.debug(f"Memory saved successfully: {serializable_memory}")
        except IOError as e:
            logger.error(f"Error saving memory to file: {e}")
            raise MemorySaveError(f"Error saving memory to file: {e}")

    def add_to_vector_store(self, messages: List[BaseMessage]) -> None:
        """Add messages to vector store for semantic retrieval"""
        for msg in messages:
            logger.debug(f"Adding message to vector store: {msg.content}")
            self.vector_store.add_texts([msg.content])

    def get_context_from_question(self, question: str) -> Dict[str, Any]:
        """Get relevant context for a question from vector store"""
        try:
            # If we have chat history, use the condense question chain
            if self.chat_history and self.llm:
                standalone_question = self.condense_question_chain.invoke({
                    "chat_history": self.chat_history,
                    "question": question
                })
                logger.debug(f"Standalone question: {standalone_question}")
            else:
                standalone_question = question
            
            # Retrieve relevant documents
            docs = self.vector_retriever.get_relevant_documents(standalone_question)
            retrieved_content = [doc.page_content for doc in docs]
            logger.debug(f"Retrieved {len(retrieved_content)} relevant documents")
            
            # Get entity information if available
            entity_info = ""
            if self.chat_history and self.llm:
                try:
                    entity_info = self.entity_extraction_chain.invoke({
                        "chat_history": self.chat_history
                    })
                    logger.debug(f"Extracted entity information: {entity_info}")
                except Exception as e:
                    logger.warning(f"Error extracting entities: {e}")
            
            # Get conversation summary if available
            summary = ""
            if self.chat_history and self.llm:
                new_messages_text = "\n".join([f"{msg.type}: {msg.content}" 
                                             for msg in self.chat_history[-4:]])
                try:
                    # If we've already generated a summary before, use it as a base
                    prev_summary = getattr(self, "_prev_summary", "")
                    summary = self.summary_chain.invoke({
                        "prev_summary": prev_summary,
                        "new_messages": new_messages_text
                    })
                    self._prev_summary = summary
                    logger.debug(f"Generated conversation summary: {summary}")
                except Exception as e:
                    logger.warning(f"Error generating summary: {e}")
            
            return {
                "retrieved_documents": retrieved_content,
                "entity_information": entity_info,
                "conversation_summary": summary,
                "standalone_question": standalone_question
            }
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return {
                "retrieved_documents": [],
                "entity_information": "",
                "conversation_summary": "",
                "standalone_question": question
            }
    
    def add_message_to_history(self, message: BaseMessage) -> None:
        """Add a message to the chat history"""
        self.chat_history.append(message)
        logger.debug(f"Added message to chat history: {message.content}")
        
        # Also add to vector store for future retrieval
        self.vector_store.add_texts([message.content])