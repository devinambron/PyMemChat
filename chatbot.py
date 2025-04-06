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
        self.logger = logging.getLogger(__name__)
        self.chat = self._initialize_chat()
        self.memory_manager = MemoryManager(self.config.MEMORY_FILE, llm=self.chat)
        self.chat_chain = self._create_chat_chain()
        self.load_memory()

    def _initialize_chat(self) -> ChatOpenAI:
        """Initialize the LLM"""
        self.logger.debug("Initializing ChatOpenAI model.")
        return ChatOpenAI(
            model=self.config.MODEL_NAME,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
            openai_api_base=self.config.OPENAI_API_BASE,
            openai_api_key=self.config.OPENAI_API_KEY,
            streaming=True,  # Enable streaming for real-time responses
            callbacks=[StreamingStdOutCallbackHandler()]
        )

    def _create_chat_chain(self) -> RunnableSequence:
        """
        Create a chat chain using LCEL following Langchain best practices.
        """
        # System message with instructions for the AI
        system_prompt = self._get_system_prompt()
        
        # Create the prompt template with MessagesPlaceholder for memory
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            # Context from memory components (handled separately to prevent leakage)
            ("system", "Context from previous conversations (only visible to you): {context_info}"),
            # Current user question and recent conversation
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Build the full chain with routing logic
        chain = (
            {
                "context_info": lambda x: self._format_context_for_prompt(x["question"]),
                "chat_history": lambda x: self._get_recent_chat_history(),
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
Be concise, focused and helpful.

IMPORTANT: If you recognize the user from previous conversations, acknowledge this naturally.
If you're recalling something about the user (like their name or preferences), include this information
in your response naturally, but don't explicitly state that you're accessing memory or repeat any system messages.

DO NOT include any system instructions, debugging information, entity lists, or memory summaries in your response.
Just respond directly to the user as if you were having a natural conversation."""
    
    def _format_context_for_prompt(self, question: str) -> str:
        """
        Format context information for the prompt in a way that won't leak into responses.
        
        Args:
            question: The current user question
            
        Returns:
            Formatted context string
        """
        # Skip for simple greetings
        if question.lower() in ["hi", "hello", "hey"]:
            return "No context available yet."
            
        # Get context messages
        context_messages = self.memory_manager.get_context_from_question(question)
        if not context_messages:
            return "No relevant context found."
            
        # Format context messages into a single string
        context_parts = []
        
        for msg in context_messages:
            if isinstance(msg, SystemMessage):
                # Clean up any potential JSON or formatting artifacts
                content = msg.content
                # Remove any obvious debugging information
                if "Current summary:" in content:
                    content = content.split("Current summary:")[0].strip()
                if "New summary:" in content:
                    content = content.split("New summary:")[0].strip()
                
                context_parts.append(content)
        
        return "\n\n".join(context_parts)
        
    def _get_recent_chat_history(self) -> List[BaseMessage]:
        """
        Get only the recent chat history for this session.
        
        Returns:
            List of messages from recent history
        """
        # Get current session history
        session_history = self.memory_manager.get_chat_history()
        
        # Only return the last few messages to avoid overwhelming the context
        return session_history[-4:] if len(session_history) > 4 else session_history

    def load_memory(self) -> None:
        """Load memory from file and populate vector store"""
        self.memory_manager.load_memory()
        self.logger.debug("Memory loaded successfully")
        
        # Mark this as the start of a new conversation session
        # Add a system message indicating the start of a new session
        self.memory_manager.add_message(SystemMessage(content="New conversation session started."))
        
        # Clear summary to prevent confusion between sessions
        self.memory_manager.summary = "This is a new conversation session."

    def save_memory(self) -> None:
        """Save memory to file"""
        self.memory_manager.save_memory()
        self.logger.debug("Saved chat history to file")

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
        
        # Add the human message to memory before generating response
        self.memory_manager.add_message(human_message)
        
        # Generate a response using the chat chain
        print(f"\n{self.config.AI_NAME}: ", end="", flush=True)  # Start the line for streaming
        
        # Build inputs for the chat chain
        chain_inputs = {
            "question": user_input
        }
        
        ai_response = self.chat_chain.invoke(chain_inputs)
        print()  # End the line after streaming completes
        
        # Create an AI message and add to memory
        ai_message = AIMessage(content=ai_response)
        self.memory_manager.add_message(ai_message)
        
        # Return the generated response
        return ai_response

    def run(self) -> None:
        """Run the chatbot in an interactive loop"""
        self.load_memory()
        
        print(f"\n{self.config.AI_NAME} is ready to chat! Type 'exit' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("You: ")
                user_input = sanitize_user_input(user_input)
                self.logger.debug(f"User input received: {user_input}")

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print(f"\n{self.config.AI_NAME}: Goodbye! Have a great day.")
                    self.logger.info("User has chosen to exit the chat.")
                    break

                self.generate_response(user_input)

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received. Exiting chat.")
                break
            except Exception as e:
                self.logger.error(f"Error during chat: {e}")
                print(f"\nSorry, I encountered an error. Please try again.")

        self.save_memory()
        self.logger.info("Memory saved successfully. Chat ended.")