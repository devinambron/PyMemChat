# Chatbot Memory Enhancements TODO

This document outlines potential improvements for the chatbot's memory system.

## 1. Implement History Management (Trimming/Summarization)

**Issue:** The current `ConversationBufferMemory` loads the entire chat history from `chat_memory.json`. As conversations grow, this can exceed the LLM's context window limit, causing errors or inefficient processing.

**Solution Ideas:**

*   **Token-Based Trimming (`trim_messages`):**
    *   **Concept:** As shown in the [Chatbot Tutorial](https://python.langchain.com/docs/tutorials/chatbot/#managing-conversation-history), Langchain provides a `trim_messages` helper. This function can be inserted into the Langchain Expression Language (LCEL) chain *after* loading the messages from memory but *before* they are passed to the prompt template.
    *   **Implementation:**
        1.  Import `trim_messages` from `langchain_core.messages`.
        2.  Instantiate the trimmer, configuring it with `max_tokens`, the `token_counter` (likely `self.llm`), and a `strategy` (e.g., `"last"` to keep recent messages). You can also choose to always `include_system` messages.
        ```python
        from langchain_core.messages import trim_messages

        trimmer = trim_messages(
            max_tokens=2000, # Adjust token limit as needed
            strategy="last",
            token_counter=self.llm,
            include_system=True,
            allow_partial=False,
        )
        ```
        3.  Modify the `_create_chat_chain` method in `chatbot.py`. Add a `RunnableLambda` step after loading `chat_history` to apply the trimmer to the loaded messages before they reach the `prompt`.
        ```python
        from operator import itemgetter
        from langchain.memory import ConversationBufferMemory
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import SystemMessage # Assuming BaseMessage types used in history

        # Inside _create_chat_chain method...
        chat_chain = (
            RunnablePassthrough.assign(
                # Load history (might return a list of BaseMessage objects or a string depending on memory type)
                chat_history_messages=RunnableLambda(self.buffer_memory.load_memory_variables) | itemgetter(self.buffer_memory.memory_key) # Assuming memory_key='chat_history' and returns list of messages
            )
            # Apply the trimmer to the list of messages
            | RunnablePassthrough.assign(
                chat_history=RunnableLambda(lambda x: trimmer.invoke(x["chat_history_messages"]))
            )
            | RunnableLambda(lambda x: self.logger.debug(f"Trimmed Chat History: {x['chat_history']}")) # Optional logging
            | prompt # The prompt expects 'chat_history' variable
            | self.llm
            | StrOutputParser()
        )
        ```
        *Note:* This assumes `buffer_memory.load_memory_variables` returns a list of `BaseMessage` objects suitable for `trim_messages`. If it returns a formatted string, a different approach or memory type might be needed. `ConversationBufferWindowMemory` might be simpler if just keeping the last K turns is sufficient.

*   **Summarization (`ConversationSummaryBufferMemory`):**
    *   **Concept:** Instead of just trimming, this memory type keeps recent messages in a buffer and progressively summarizes older messages into a condensed summary, passing both to the LLM. This preserves context from older parts of the conversation more effectively than simple trimming. See [Summary Memory Docs](https://python.langchain.com/docs/how_to/chatbots_memory/#summary-memory).
    *   **Implementation:**
        1.  In `memory_manager.py`, change `ConversationBufferMemory` to `ConversationSummaryBufferMemory`.
        2.  Pass the `llm` instance to `ConversationSummaryBufferMemory` during initialization, as it needs an LLM to create the summaries. Also set `max_token_limit`.
        ```python
        # In MemoryManager.__init__
        from langchain.memory import ConversationSummaryBufferMemory

        self.buffer_memory = ConversationSummaryBufferMemory(
            llm=llm, # Pass the LLM instance
            max_token_limit=1500, # Set desired token limit
            memory_key="chat_history",
            return_messages=True # Important for prompt template
        )
        ```
        3.  Ensure the `load_memory` and `save_memory` methods in `MemoryManager` correctly interact with this memory type (the interface should be similar).
        4.  The LCEL chain in `chatbot.py` might not need significant changes if `ConversationSummaryBufferMemory` is configured with `return_messages=True` and the prompt uses `MessagesPlaceholder(variable_name="chat_history")`.

## 2. Re-integrate Vector Store Retrieval

**Issue:** The vector store (`self.vector_store` in `MemoryManager`) is being populated with message embeddings but is not currently used for context retrieval in the chat chain. This limits recall to only what's in the buffer/summary memory.

**Solution Ideas:**

*   **Custom Retrieval Step in LCEL:**
    *   **Concept:** Add a step in the LCEL chain to query the vector store for relevant historical messages based on the current user input, and inject this retrieved context into the prompt.
    *   **Implementation:**
        1.  In `MemoryManager`, ensure the `_find_relevant_context` method works correctly (it should take user input, query the vector store, and return formatted context).
        2.  In `chatbot.py`, modify `_create_chat_chain`:
            *   Add a placeholder in the `ChatPromptTemplate` for the retrieved context (e.g., `{retrieved_context}`).
            *   Use `RunnablePassthrough.assign` to call `memory_manager._find_relevant_context` using the user's input (`itemgetter("question")`) and assign the result to `retrieved_context`.
        ```python
        # Inside _create_chat_chain method...
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\nRelevant historical context:\n{retrieved_context}"), # Add placeholder
            MessagesPlaceholder(variable_name="chat_history"), # Existing buffer/summary history
            ("human", "{question}")
        ])

        chat_chain = (
            RunnablePassthrough.assign(
                # Existing history loading (trimmed/summarized)
                chat_history_messages=RunnableLambda(self.buffer_memory.load_memory_variables) | itemgetter(self.buffer_memory.memory_key),
                question=itemgetter("question") # Ensure question is available
            )
            | RunnablePassthrough.assign(
                # Apply trimmer if using that approach
                chat_history=RunnableLambda(lambda x: trimmer.invoke(x["chat_history_messages"])) # Example if using trimmer
            )
             # Add retrieval step
            | RunnablePassthrough.assign(
                retrieved_context=itemgetter("question") | RunnableLambda(self.memory_manager._find_relevant_context)
            )
            | RunnableLambda(lambda x: self.logger.debug(f"Retrieved Context: {x['retrieved_context']}")) # Optional logging
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # In invoke_chain in Chatbot class, ensure input dict has 'question' key
        # Example:
        # response = self.chat_chain.invoke({"question": user_input})
        ```
        3.  Update the `system_prompt` to instruct the LLM on how to use both the `chat_history` and the `retrieved_context`.

*   **Using `VectorStoreRetrieverMemory`:**
    *   **Concept:** Langchain offers `VectorStoreRetrieverMemory` which combines a retriever (from a vector store) with memory management. It automatically retrieves relevant documents and formats them into a memory variable.
    *   **Implementation:**
        1.  Instead of (or in addition to) `ConversationBufferMemory`, instantiate `VectorStoreRetrieverMemory` in `MemoryManager`, passing it the vector store's retriever (`self.vector_store.as_retriever()`).
        ```python
        # In MemoryManager.__init__
        from langchain.memory import VectorStoreRetrieverMemory

        retriever = self.vector_store.as_retriever(search_kwargs=dict(k=3)) # Get top 3 results
        self.vector_memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="retrieved_context" # Use a distinct key
        )
        ```
        2.  Modify the LCEL chain in `chatbot.py` to load variables from this memory (`self.vector_memory.load_memory_variables`) and assign them to the appropriate prompt variable (`retrieved_context`).
        3.  Ensure the `save_context` calls in `Chatbot` also update this memory type if necessary (though often vector store memory primarily reads, and saving happens directly to the vector store elsewhere).

## 3. Re-integrate Entity Memory

**Issue:** `ConversationEntityMemory` was removed for debugging. If dedicated entity tracking and summarization are desired, it needs to be added back.

**Solution Ideas:**

*   **Add Back to LCEL Chain:**
    *   **Concept:** Similar to vector store retrieval, load the entity summary from `ConversationEntityMemory` and inject it into the prompt.
    *   **Implementation:**
        1.  Ensure `ConversationEntityMemory` is initialized correctly in `MemoryManager` (likely needs the `llm` instance).
        2.  In `chatbot.py`, modify `_create_chat_chain`:
            *   Add a placeholder (e.g., `{entities}`) to the `ChatPromptTemplate`. The system prompt should explain what this is.
            *   Use `RunnablePassthrough.assign` to load the entity memory using `RunnableLambda(self.memory_manager.entity_memory.load_memory_variables) | itemgetter("entities")` (adjust `itemgetter` key based on the memory object's output, often it's the `memory_key`).
        ```python
        # Inside _create_chat_chain method...

        # Ensure entity_memory is initialized in MemoryManager
        # self.entity_memory = ConversationEntityMemory(llm=self.llm, memory_key="entities")

        system_prompt = self._get_system_prompt() # Make sure prompt mentions entities

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\nRelevant entities mentioned so far:\n{entities}\n\nRelevant historical context:\n{retrieved_context}\n\nCurrent conversation:"), # Added entities placeholder
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        chat_chain = (
            RunnablePassthrough.assign(
                # Load buffer/summary history (potentially trimmed)
                chat_history_messages=RunnableLambda(self.buffer_memory.load_memory_variables) | itemgetter(self.buffer_memory.memory_key),
                question=itemgetter("question")
            )
            | RunnablePassthrough.assign(
                # Apply trimmer if used
                chat_history=RunnableLambda(lambda x: trimmer.invoke(x["chat_history_messages"])) # Example
            )
            | RunnablePassthrough.assign(
                # Load vector context if used
                retrieved_context=itemgetter("question") | RunnableLambda(self.memory_manager._find_relevant_context) # Example
            )
             # Load entity memory
            | RunnablePassthrough.assign(
                entities=RunnableLambda(self.memory_manager.entity_memory.load_memory_variables) | itemgetter(self.memory_manager.entity_memory.memory_key) # Use memory_key
            )
            | RunnableLambda(lambda x: self.logger.debug(f"Entities: {x['entities']}")) # Optional logging
            | prompt
            | self.llm
            | StrOutputParser()
        )
        ```
        3.  Make sure `save_context` in `Chatbot` calls `self.memory_manager.entity_memory.save_context` after each turn.
        4.  Update `_get_system_prompt` to explain the `entities` context variable to the LLM.

**Priority:** Addressing history management (Item 1) is likely the most crucial for preventing errors in longer conversations. Integrating vector store retrieval (Item 2) offers better scalability and recall than simple buffering. Entity memory (Item 3) is useful if specific entity tracking is a core requirement. 