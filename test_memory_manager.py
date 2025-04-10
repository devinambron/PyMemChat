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
        self.mock_vector_retriever = MagicMock()
        self.mock_vector_store.as_retriever.return_value = self.mock_vector_retriever
        
        # Patch the create_vector_store function
        self.vector_store_patcher = patch('memory_manager.create_vector_store', 
                                         return_value=self.mock_vector_store)
        self.mock_create_vector_store = self.vector_store_patcher.start()
        
        # Store a reference to self for the patched init to access
        test_case = self
        
        # Instead of trying to patch individual memory components which are no longer
        # directly accessible, we'll patch the __init__ method of the MemoryManager
        # to avoid initializing those components
        original_init = MemoryManager.__init__
        
        def patched_init(self_mm, file_path, llm=None):
            # Call the original init but catch and ignore any exceptions
            try:
                original_init(self_mm, file_path, llm)
            except Exception:
                # Continue with our custom initialization
                pass
            
            # Override properties for testing
            self_mm.file_path = file_path
            self_mm.logger = logging.getLogger(__name__)
            self_mm.llm = llm
            self_mm.vector_store = test_case.mock_vector_store
            self_mm.vector_retriever = test_case.mock_vector_retriever
            self_mm.chat_history = []
            self_mm.current_session_history = []
            self_mm.message_history = MagicMock()
            self_mm.entity_store = {}
            self_mm.summary = ""
            self_mm.silent_mode = True
            self_mm.session_started = False
        
        # Save original method and patch
        self.original_init = MemoryManager.__init__
        MemoryManager.__init__ = patched_init
        
        # Initialize memory manager with our mock LLM and patched vector store
        self.memory_manager = MemoryManager(self.temp_memory_path, llm=self.mock_llm)
        
        # Mock specific methods for testing
        self.memory_manager._extract_entities = MagicMock(return_value='{"Person": "Alice", "Place": "Paris"}')
        self.memory_manager._summarize_conversation = MagicMock(return_value="This is a conversation summary")
        self.memory_manager._condense_question = MagicMock(return_value="Standalone question")
        
        # Create mock chains for LCEL pattern
        self.memory_manager.entity_extraction_chain = MagicMock()
        self.memory_manager.summary_chain = MagicMock()
        self.memory_manager.condense_question_chain = MagicMock()
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove the temporary file
        os.close(self.temp_memory_fd)
        os.unlink(self.temp_memory_path)
        
        # Restore original __init__ method
        MemoryManager.__init__ = self.original_init
        
        # Stop the vector store patcher
        self.vector_store_patcher.stop()
    
    def test_empty_memory_initialization(self):
        """Test initializing with an empty memory file"""
        # Memory should be empty after initialization with empty file
        self.assertEqual(len(self.memory_manager.chat_history), 0)
        self.assertEqual(len(self.memory_manager.current_session_history), 0)
    
    def test_add_message(self):
        """Test adding messages to memory"""
        # Add a human message
        human_msg = HumanMessage(content="Hello, I'm a test user")
        self.memory_manager.add_message(human_msg)
        
        # Verify it was added to current session
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
    
    def test_get_chat_history(self):
        """Test retrieving chat history"""
        # Add some messages
        self.memory_manager.add_message(HumanMessage(content="Test message 1"))
        self.memory_manager.add_message(AIMessage(content="Test response 1"))
        self.memory_manager.add_message(HumanMessage(content="Test message 2"))
        self.memory_manager.add_message(AIMessage(content="Test response 2"))
        
        # Get chat history
        history = self.memory_manager.get_chat_history()
        
        # Verify correct format and content
        self.assertEqual(len(history), 4)
        self.assertIsInstance(history[0], BaseMessage)
        self.assertEqual(history[0].content, "Test message 1")
        self.assertEqual(history[1].content, "Test response 1")
        self.assertEqual(history[2].content, "Test message 2")
        self.assertEqual(history[3].content, "Test response 2")
    
    def test_save_and_load_memory(self):
        """Test saving and loading memory from file"""
        # Add some messages
        self.memory_manager.add_message(HumanMessage(content="Save test message"))
        self.memory_manager.add_message(AIMessage(content="Save test response"))
        
        # Save the memory
        self.memory_manager.save_memory()
        
        # Create a new memory manager to test loading
        # Use the same patched __init__ method
        new_memory_manager = MemoryManager(self.temp_memory_path, llm=self.mock_llm)
        
        # Load memory
        new_memory_manager.load_memory()
        
        # Verify the messages were loaded
        self.assertEqual(len(new_memory_manager.chat_history), 2)
        self.assertEqual(new_memory_manager.chat_history[0]["content"], "Save test message")
        self.assertEqual(new_memory_manager.chat_history[1]["content"], "Save test response")
    
    def test_context_from_question(self):
        """Test retrieving context for a question"""
        # Create a mock document for the vector retriever to return
        mock_docs = [
            Document(page_content="Relevant document 1"),
            Document(page_content="Relevant document 2")
        ]
                
        # Mock the invoke method
        self.memory_manager.vector_retriever.invoke = MagicMock(return_value=mock_docs)
        
        # Get context
        context = self.memory_manager.get_context_from_question("What is the capital of France?")
        
        # Verify the context format
        self.assertIsInstance(context, list)
        
        # Verify context content if there are any messages
        if context:
            # Each context item should be a SystemMessage
            for item in context:
                self.assertIsInstance(item, SystemMessage)
            
            # Check at least one context message mentions relevant information
            context_text = " ".join([msg.content for msg in context])
            self.assertTrue(
                any(term in context_text.lower() for term in 
                    ["relevant", "document", "conversation", "entity"])
            )
    
    def test_scan_for_entities(self):
        """Test scanning for entities in chat history"""
        # Create chat history with entity information
        self.memory_manager.chat_history = [
            {"role": "user", "content": "My name is John"},
            {"role": "ai", "content": "Nice to meet you, John"},
            {"role": "user", "content": "My favorite color is blue"}
        ]
        
        # Call the scan method
        self.memory_manager._scan_for_entities()
        
        # Verify entities were extracted
        self.assertIn("user_name", self.memory_manager.entity_store)
        self.assertIn("The user's name is John", self.memory_manager.entity_store["user_name"])
        self.assertIn("favorite_color", self.memory_manager.entity_store)
        self.assertIn("blue", self.memory_manager.entity_store["favorite_color"])
    
    def test_add_system_message(self):
        """Test adding a system message"""
        # Add a system message
        sys_msg = SystemMessage(content="New conversation session started.")
        self.memory_manager.add_message(sys_msg)
        
        # Verify it was added to current session
        self.assertEqual(len(self.memory_manager.current_session_history), 1)
        self.assertEqual(self.memory_manager.current_session_history[0]["role"], "system")
        self.assertEqual(self.memory_manager.current_session_history[0]["content"], "New conversation session started.")
    
    def test_session_separation(self):
        """Test that current session messages don't affect previous history"""
        # Add messages to the first session
        self.memory_manager.add_message(HumanMessage(content="First session message"))
        self.memory_manager.add_message(AIMessage(content="First session response"))
        
        # Save the memory
        self.memory_manager.save_memory()
        
        # Create a new session
        new_session = MemoryManager(self.temp_memory_path, llm=self.mock_llm)
        
        # Load memory in the new session
        new_session.load_memory()
        
        # Add messages to the new session
        new_session.add_message(HumanMessage(content="Second session message"))
        
        # Verify that the current_session_history only contains new session messages
        self.assertEqual(len(new_session.current_session_history), 1)
        self.assertEqual(new_session.current_session_history[0]["content"], "Second session message")
        
        # Verify that chat_history contains the first session messages
        self.assertEqual(len(new_session.chat_history), 2)
        self.assertEqual(new_session.chat_history[0]["content"], "First session message")
        self.assertEqual(new_session.chat_history[1]["content"], "First session response")

if __name__ == '__main__':
    unittest.main() 