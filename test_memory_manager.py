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
            self_mm.file_path = file_path
            self_mm.logger = logging.getLogger(__name__)
            self_mm.llm = llm
            self_mm.vector_store = test_case.mock_vector_store
            self_mm.vector_retriever = test_case.mock_vector_retriever
            self_mm.chat_history = []
            self_mm.current_session_history = []
            self_mm.summary = ""
            self_mm.session_started = False
            
            # Mock the memory components but don't try to initialize them
            self_mm.buffer_memory = MagicMock()
            self_mm.summary_memory = MagicMock()
            self_mm.entity_memory = MagicMock()
            self_mm.vector_memory = MagicMock()
            self_mm.combined_memory = MagicMock()
            
            # Mock the specialized chains
            self_mm.condense_question_chain = MagicMock()
            self_mm.entity_extraction_chain = MagicMock()
            self_mm.summary_chain = MagicMock()
            
            # Mock memory histories
            self_mm.memory_histories = MagicMock()
        
        # Save original method and patch
        self.original_init = MemoryManager.__init__
        MemoryManager.__init__ = patched_init
        
        # Initialize memory manager with our mock LLM and patched vector store
        self.memory_manager = MemoryManager(self.temp_memory_path, llm=self.mock_llm)
        
        # Mock specific methods for testing
        self.memory_manager._extract_entities = MagicMock(return_value='{"Person": "Alice", "Place": "Paris"}')
        self.memory_manager._summarize_conversation = MagicMock(return_value="This is a conversation summary")
        self.memory_manager._condense_question = MagicMock(return_value="Standalone question")
    
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
        class MockDocument:
            def __init__(self, content):
                self.page_content = content
                
        # Mock the retrieve method
        self.memory_manager.vector_retriever.invoke = MagicMock(return_value=[
            MockDocument("Relevant document 1"),
            MockDocument("Relevant document 2")
        ])
        
        # Get context
        context = self.memory_manager.get_context_from_question("What is the capital of France?")
        
        # Verify the methods were called
        self.memory_manager._extract_entities.assert_called_once()
        self.memory_manager._summarize_conversation.assert_called_once()
        
        # Verify the context format
        self.assertIsInstance(context, list)
        
        # Each context item should be a SystemMessage
        for item in context:
            self.assertIsInstance(item, SystemMessage)
        
        # Verify context content
        context_text = "".join([msg.content for msg in context])
        self.assertIn("conversation", context_text)
        self.assertIn("entities", context_text)
        self.assertIn("Relevant document", context_text)
    
    def test_initialize_summary(self):
        """Test initializing summary from existing messages"""
        # Create mock chat history with enough messages
        self.memory_manager.chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "ai", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "ai", "content": "I'm doing well"},
            {"role": "user", "content": "What's the weather like?"}
        ]
        
        # Set up the mock LLM to return a summary
        self.mock_llm.invoke = MagicMock(return_value="This conversation is about greetings and weather")
        
        # Initialize summary
        self.memory_manager._initialize_summary()
        
        # Verify the summary was set
        self.assertEqual(self.memory_manager.summary, "This conversation is about greetings and weather")
    
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