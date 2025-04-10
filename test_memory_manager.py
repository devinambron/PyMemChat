import unittest
import os
import tempfile
import json
import logging
from unittest.mock import patch, MagicMock
from memory_manager import MemoryManager
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory

# Disable logging for tests
logging.disable(logging.CRITICAL)

class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        """Set up a test environment with a temporary memory file"""
        # Create a temporary file for memory storage
        self.temp_memory_fd, self.temp_memory_path = tempfile.mkstemp(suffix='.json')
        
        # Initialize empty memory file
        with open(self.temp_memory_path, 'w') as f:
            json.dump([], f)
            
        # Create a proper mock LLM that can be recognized as a BaseLanguageModel
        self.mock_llm = MagicMock(spec=BaseLanguageModel)
        self.mock_llm.invoke = MagicMock(return_value="This is a test summary")
        
        # Create a patched vector_store and vector_retriever
        self.mock_vector_store = MagicMock()
        # Add necessary methods expected by the code (e.g., add_documents)
        self.mock_vector_store.add_documents = MagicMock() 
        self.mock_vector_retriever = MagicMock()
        self.mock_vector_store.as_retriever.return_value = self.mock_vector_retriever
        
        # Patch the create_vector_store function
        self.vector_store_patcher = patch('memory_manager.create_vector_store', 
                                         return_value=self.mock_vector_store)
        self.mock_create_vector_store = self.vector_store_patcher.start()
        
        # Initialize memory manager with our mock LLM and patched vector store
        # No longer patching __init__
        self.memory_manager = MemoryManager(self.temp_memory_path, llm=self.mock_llm)
        # Ensure message_history is a real object for testing get_chat_history
        self.memory_manager.message_history = ChatMessageHistory() 
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove the temporary file
        os.close(self.temp_memory_fd)
        os.unlink(self.temp_memory_path)
        
        # Restore original __init__ method - No longer needed
        # MemoryManager.__init__ = self.original_init 
        
        # Stop the vector store patcher
        self.vector_store_patcher.stop()
    
    def test_empty_memory_initialization(self):
        """Test initializing with an empty memory file"""
        # Memory should be empty after initialization with empty file
        self.assertEqual(len(self.memory_manager.chat_history), 0)
        self.assertEqual(len(self.memory_manager.current_session_history), 0)
        # Check that message_history (used by LCEL) is also empty initially
        self.assertEqual(len(self.memory_manager.message_history.messages), 0) 
    
    def test_add_message(self):
        """Test adding messages to memory (current_session_history for saving)"""
        # Add a human message
        human_msg = HumanMessage(content="Hello, I'm a test user")
        self.memory_manager.add_message(human_msg)
        
        # Verify it was added to current session history (for saving)
        self.assertEqual(len(self.memory_manager.current_session_history), 1)
        self.assertEqual(self.memory_manager.current_session_history[0]["role"], "user")
        self.assertEqual(self.memory_manager.current_session_history[0]["content"], "Hello, I'm a test user")
        
        # Add an AI message
        ai_msg = AIMessage(content="Hello test user, I'm an AI")
        self.memory_manager.add_message(ai_msg)
        
        # Verify it was added
        self.assertEqual(len(self.memory_manager.current_session_history), 2)
        self.assertEqual(self.memory_manager.current_session_history[1]["role"], "ai")
        self.assertEqual(self.memory_manager.current_session_history[1]["content"], "Hello test user, I'm an AI")

        # Note: This test doesn't check self.message_history because add_message 
        # doesn't directly update it. message_history is updated by memory components
        # like ConversationBufferMemory using save_context in the chatbot logic.
    
    def test_get_chat_history(self):
        """Test retrieving chat history from message_history object."""
        # Directly add messages to the message_history object for this test
        msg1 = HumanMessage(content="Test message 1")
        msg2 = AIMessage(content="Test response 1")
        msg3 = HumanMessage(content="Test message 2")
        msg4 = AIMessage(content="Test response 2")
        
        self.memory_manager.message_history.add_message(msg1)
        self.memory_manager.message_history.add_message(msg2)
        self.memory_manager.message_history.add_message(msg3)
        self.memory_manager.message_history.add_message(msg4)
        
        # Get chat history using the method under test
        history = self.memory_manager.get_chat_history()
        
        # Verify correct format and content from message_history
        self.assertEqual(len(history), 4)
        self.assertIsInstance(history[0], BaseMessage)
        self.assertEqual(history[0].content, "Test message 1")
        self.assertIsInstance(history[1], AIMessage)
        self.assertEqual(history[1].content, "Test response 1")
        self.assertEqual(history[2].content, "Test message 2")
        self.assertEqual(history[3].content, "Test response 2")
        
        # Ensure it's retrieving from the correct object
        self.assertIs(history, self.memory_manager.message_history.messages)
    
    def test_save_and_load_memory(self):
        """Test saving and loading memory from file (chat_history)"""
        # Add some messages using add_message (populates current_session_history)
        self.memory_manager.add_message(HumanMessage(content="Save test message"))
        self.memory_manager.add_message(AIMessage(content="Save test response"))
        
        # Save the memory (combines chat_history + current_session_history)
        self.memory_manager.save_memory()
        
        # Create a new memory manager instance to test loading
        new_memory_manager = MemoryManager(self.temp_memory_path, llm=self.mock_llm)
        
        # Load memory using the actual load_memory method
        new_memory_manager.load_memory()
        
        # Verify the messages were loaded into the persistent chat_history list
        self.assertEqual(len(new_memory_manager.chat_history), 2)
        self.assertEqual(new_memory_manager.chat_history[0]["content"], "Save test message")
        self.assertEqual(new_memory_manager.chat_history[1]["content"], "Save test response")
        
        # Optionally: Verify message_history was populated during load
        # This depends on the process_memory_data helper function
        self.assertEqual(len(new_memory_manager.message_history.messages), 2)
        self.assertEqual(new_memory_manager.message_history.messages[0].content, "Save test message")
        self.assertEqual(new_memory_manager.message_history.messages[1].content, "Save test response")

    # Removed test_context_from_question as the method doesn't exist

    # Removed test_scan_for_entities as the method doesn't exist 
    # and entity memory population happens via save_context elsewhere.
    
    def test_add_system_message(self):
        """Test adding a system message to current_session_history"""
        # Add a system message
        sys_msg = SystemMessage(content="New conversation session started.")
        self.memory_manager.add_message(sys_msg)
        
        # Verify it was added to current session history
        self.assertEqual(len(self.memory_manager.current_session_history), 1)
        self.assertEqual(self.memory_manager.current_session_history[0]["role"], "system")
        self.assertEqual(self.memory_manager.current_session_history[0]["content"], "New conversation session started.")
    
    def test_session_separation(self):
        """Test that current session messages don't affect previous history upon save/load"""
        # --- First Session ---
        # Add messages to the first session
        self.memory_manager.add_message(HumanMessage(content="First session message"))
        self.memory_manager.add_message(AIMessage(content="First session response"))
        
        # Verify current history before save
        self.assertEqual(len(self.memory_manager.current_session_history), 2)
        
        # Save the memory (writes chat_history + current_session_history)
        self.memory_manager.save_memory() 
        
        # --- Second Session ---
        # Create a new session instance (clears internal state like current_session_history)
        new_session = MemoryManager(self.temp_memory_path, llm=self.mock_llm)
        
        # Load memory in the new session (populates chat_history and message_history)
        new_session.load_memory()
        
        # Verify loaded history
        self.assertEqual(len(new_session.chat_history), 2, "Should load 2 messages into chat_history")
        self.assertEqual(new_session.chat_history[0]["content"], "First session message")
        # Verify current session history is reset
        self.assertEqual(len(new_session.current_session_history), 0, "current_session_history should be empty after load")
        
        # Add messages to the new session
        new_session.add_message(HumanMessage(content="Second session message"))
        
        # Verify that the current_session_history only contains the new session message
        self.assertEqual(len(new_session.current_session_history), 1, "Should have 1 message in current history")
        self.assertEqual(new_session.current_session_history[0]["content"], "Second session message")
        
        # Verify that chat_history still contains only the first session messages
        self.assertEqual(len(new_session.chat_history), 2, "chat_history should remain unchanged in the new session")
        self.assertEqual(new_session.chat_history[0]["content"], "First session message")

if __name__ == '__main__':
    unittest.main() 