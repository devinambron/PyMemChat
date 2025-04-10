# Chatbot Memory & Persona Enhancements TODO

This document outlines potential improvements for evolving the chatbot Ava towards a more persistent, context-aware, and personable AI, inspired by concepts like the AI in "Her".

## Foundational Goal: Create a "Friend" Persona

**Objective:** Move beyond a simple assistant to an AI that acts as a friend, remembers personal details, and adapts to the user's preferences. This involves deep memory and nuanced interaction.

---

## 1. Define and Refine Ava's Personality & Role

**Issue:** Ava currently has a basic "helpful assistant" prompt. To act as a friend, her core personality, motivations, and communication style need explicit definition.

**Solution Ideas:**

*   **Advanced System Prompt Engineering:** - **DONE**
    *   **Concept:** Craft a detailed system prompt that defines Ava's persona. Inspired by AI character templates (like those suggested in the `gpt-j-chatbot` repo [[GitHub](https://github.com/machaao/gpt-j-chatbot)]), this prompt should specify:
        *   **Role:** A supportive, curious, empathetic, and engaging friend.
        *   **Traits:** Perhaps slightly quirky, non-judgmental, humorous, good listener.
        *   **Communication Style:** Natural, conversational, uses user's name, asks follow-up questions, expresses opinions appropriately.
        *   **Goals:** To build rapport, remember user details, provide companionship.
        *   **Boundaries:** Define what Ava *won't* do (e.g., claim sentience, harmful actions), learning from examples like Microsoft Tay [[Chatbot.com Blog](https://www.chatbot.com/blog/chatbot-guide/)].
    *   **Implementation:** Iteratively refine the `_get_system_prompt` method in `chatbot.py`. Test different phrasings and levels of detail.

*   **Response Generation Tuning:** - **DONE**
    *   **Concept:** Adjust LLM parameters (`temperature`, `top_p`, `top_k`) mentioned in resources like the `gpt-j-chatbot` `.env` setup [[GitHub](https://github.com/machaao/gpt-j-chatbot)] to encourage more creative, less deterministic responses suitable for a friend persona.
    *   **Implementation:** Expose these parameters in the `Chatbot` class or configuration and experiment with values.

*   **Dynamic Prompting/Tone Adaptation:** - **DEFERRED**
    *   **Concept:** Go beyond a static system prompt. Adapt Ava's tone, style, or even parts of the prompt dynamically based on the user's detected emotional state (from Task 4) or explicitly stated preferences (from Task 2B - User Profile).
    *   **Implementation:** This might involve more complex logic, potentially using LangGraph (Task 6) to modify the prompt/context before calling the LLM based on sentiment or profile data.

---

## 2. Implement Robust Multi-Layered Memory Architecture

**Objective:** Combine short-term session memory with persistent long-term memory, utilizing the 128K token context window effectively and ensuring key details/preferences are retained across sessions.

**A. Short-Term Memory (Active Session Context):** - **DONE**

*   **Tool:** `ConversationSummaryBufferMemory`
    *   **Concept:** Use this memory type to manage the immediate conversation history. It keeps recent messages directly and summarizes older ones, balancing context richness with token limits. This acts like a sophisticated **sliding window** approach. See [Summary Memory Docs](https://python.langchain.com/docs/how_to/chatbots_memory/#summary-memory).
    *   **Implementation:**
        1.  In `memory_manager.py`, replace `ConversationBufferMemory` with `ConversationSummaryBufferMemory`.
        2.  **Configure for Large Context:** Initialize it with a large `max_token_limit` (e.g., 50,000-100,000 tokens) to leverage the 128K window, allowing substantial direct history before summarization kicks in. Pass the `llm` instance for summarization.
            ```python
            # In MemoryManager.__init__
            from langchain.memory import ConversationSummaryBufferMemory
            self.buffer_memory = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=80000, # Utilize large context window
                memory_key="chat_history",
                return_messages=True # Crucial for prompt templates
            )
            ```
        3.  Ensure `load_memory` and `save_memory` work correctly. The LCEL chain in `chatbot.py` should use `MessagesPlaceholder(variable_name="chat_history")` [[Langchain Docs](https://python.langchain.com/docs/modules/agents/#adding-memory)].

**B. Long-Term Memory (Cross-Session Knowledge & Profile):**

*   **Core Idea:** Store distilled knowledge (key facts, summaries, user preferences) persistently, likely in a vector store for efficient retrieval (**Retrieval-Augmented Generation - RAG**), potentially augmented by a structured user profile.

*   **Tool 1: Vector Store for Semantic Recall (RAG)** - **Retrieval Implemented**
    *   **Concept:** Store summaries of past conversations, key facts learned (about the user or the world), and perhaps user preferences as distinct documents/chunks in the vector store (`self.vector_store`). Retrieve relevant snippets based on the current conversation turn using similarity search.
    *   **Implementation Strategy:**
        1.  **Saving:** Implement the \"Post-Session Summarization\" task (see Task 3 below) to generate summaries/facts and embed/add them to `self.vector_store` in `MemoryManager`. - **PENDING**
        2.  **Retrieval (LCEL):** In `_create_chat_chain` (`chatbot.py`), add a step to retrieve relevant documents from the vector store based on the current `question`. - **DONE**
            *   Inject a `retrieved_context` variable into the prompt. - **DONE**
            *   Use `RunnablePassthrough.assign` with `itemgetter(\"question\") | self.vector_store.as_retriever() | format_docs` (where `format_docs` is a function to format retrieved documents nicely). - **DONE**
            *   Update the system prompt to instruct the LLM to synthesize information from `chat_history` (short-term) and `retrieved_context` (long-term). - **DONE**
            ```python
            # Inside _create_chat_chain...
            from langchain_core.runnables import RunnableLambda, RunnablePassthrough
            from operator import itemgetter

            def format_docs(docs):
                # Consider adding metadata (e.g., timestamp) to the formatted string
                return "\n\n".join(doc.page_content for doc in docs)

            prompt = ChatPromptTemplate.from_messages([
                # System prompt instructs how to use both history types
                ("system", system_prompt + "\n\n[Background Knowledge & Profile Notes]\n{retrieved_context}\n\n[Current Conversation History]\n"),
                MessagesPlaceholder(variable_name="chat_history"), # From SummaryBufferMemory
                ("human", "{question}")
            ])

            chat_chain = (
                RunnablePassthrough.assign(
                    # Load buffer/summary history
                    chat_history_messages=RunnableLambda(self.buffer_memory.load_memory_variables) | itemgetter(self.buffer_memory.memory_key),
                    question=itemgetter("question")
                )
                # Add retrieval step (RAG)
                | RunnablePassthrough.assign(
                   retrieved_context=(
                       itemgetter("question") |
                       self.memory_manager.vector_store.as_retriever(search_kwargs=dict(k=5)) | # Retrieve top 5 relevant docs
                       RunnableLambda(format_docs)
                   )
                )
                # Ensure chat_history is passed correctly (as message list)
                | RunnablePassthrough.assign(
                     chat_history=itemgetter("chat_history_messages")
                 )
                | prompt
                | self.llm
                | StrOutputParser()
            )
            ```

*   **Tool 2: Structured User Profile (Optional/Complementary)**
    *   **Concept:** Maintain a separate, structured store (e.g., `user_profile.json` or a dedicated DB table) for readily accessible, non-semantic data like name, stated preferences (e.g., communication tone, topics to avoid), or explicitly confirmed key facts.
    *   **Implementation:**
        1.  Create methods in `MemoryManager` to load/save/update this profile.
        2.  Inject key profile data into the prompt (e.g., `user_profile_summary` or specific fields) or use it to configure other steps (like dynamic prompting).
        3.  Update the profile based on session summaries or explicit user statements identified via LLM or rules.

**C. Entity Memory (Optional Enhancement):**

*   **Tool:** `ConversationEntityMemory`
    *   **Concept:** If explicitly tracking and summarizing specific entities (people, places mentioned frequently) is valuable beyond general summarization/retrieval, re-integrate this.
    *   **Implementation:** As detailed previously, add another `RunnablePassthrough.assign` step to load entity memory into the prompt context and ensure `save_context` updates it. Update the system prompt accordingly.

---

## 3. Implement Post-Session Summarization for Long-Term Memory - **DONE**

**Issue:** Raw conversation history grows large. Distilling key information after each session is crucial for efficient long-term memory.

**Solution Idea:**

*   **LLM-Powered Summarization/Fact Extraction:** - **DONE**
    *   **Concept:** When a session ends (e.g., user types `exit` or after a period of inactivity), trigger an LLM call to process the session's history.
    *   **Implementation:** - **DONE**
        1.  In `chatbot.py`, modify the main loop's exit/idle condition. Before exiting/saving, capture the final state of `self.buffer_memory`. - **DONE**
        2.  Create a new method `memory_manager.create_and_store_session_summary(messages)`. - **DONE**
        3.  Inside this method, use `self.llm` with a dedicated prompt (e.g., \"Analyze this conversation: {conversation_text}. Extract key facts learned about the user (name, preferences like favorite color: blue, stated goals: finding a new job), significant events discussed, and generate a concise summary focusing on information crucial for remembering the user and maintaining context in future interactions.\") - **DONE**
        4.  Take the generated summary and/or extracted facts. - **DONE**
        5.  **Store:** Embed the summary/facts and add them as documents to `self.vector_store` (for Task 2B). Potentially also use extracted facts to update the structured user profile (Task 2B, Tool 2). - **DONE (Vector Store part)**

---

## 4. Enhance Emotional Intelligence and Natural Interaction

**Issue:** To feel like a friend, Ava needs to understand and respond to nuances beyond literal meaning.

**Solution Ideas:**

*   **Sentiment Analysis:** - **Structure Implemented (Placeholder)**
    *   **Concept:** Analyze user input for sentiment (positive, negative, neutral) to tailor responses. E.g., respond more empathetically to negative sentiment. Discussed in [[Chatbot.com Blog](https://www.chatbot.com/blog/chatbot-guide/)].
    *   **Implementation:**
        1.  Find/implement a sentiment analysis method (library like NLTK/VADER, Hugging Face model, or dedicated LLM call). - **Placeholder Added**
        2.  In `_create_chat_chain`, add a step to perform analysis on the user `question`. - **DONE**
        3.  Assign the sentiment (e.g., `user_sentiment`) to the prompt context or use it in downstream logic (like Dynamic Prompting in Task 1). - **DONE (Context)**
        4.  Update the system prompt or response generation logic to explicitly consider `user_sentiment` for **Emotionally Adaptive Responses** (e.g., \"If user_sentiment is negative, respond with extra empathy and support.\"). - **DONE (Prompt)**

*   **Ask Clarifying Questions:** - **DONE**
    *   **Concept:** Train or prompt Ava to ask for clarification when user input is ambiguous or lacks context, rather than guessing or giving a generic response.
    *   **Implementation:** Include instructions in the system prompt (e.g., \"If you are unsure about the user's meaning or need more information to provide a helpful response, ask a clarifying question.\"). Monitor responses and refine the prompt. - **DONE (Prompt)**

*   **Memory-Driven Storytelling & Continuity:** - **DONE (Covered by existing prompt)**
    *   **Concept:** Instruct Ava (via system prompt) to leverage retrieved context (`retrieved_context`) and recent history (`chat_history`) to make callbacks to previous topics, shared experiences, or facts learned, creating conversational continuity.
    *   **Implementation:** Prompt engineering: \"Refer back to things you've learned about the user or topics you've discussed previously when relevant to build rapport and show you remember.\" - **DONE (Prompt)**

*   **Interactive Learning (Advanced):** - **DEFERRED**
    *   **Concept:** Explore techniques where Ava might occasionally use recall (e.g., \"Didn't we talk about your interest in X last week?\") to reinforce shared memory, similar to human interaction.
    *   **Implementation:** Requires careful prompting and potentially tracking interaction patterns. Lower priority.

---

## 5. Leverage Langchain Architecture (LCEL/LangGraph)

**Overarching Principle:** Use Langchain's tools effectively.

*   **LCEL:** Continue using the Langchain Expression Language for chaining components as primary approach.
*   **LangGraph:** For more complex flows (e.g., multi-step reasoning, dynamic agent selection, complex conditional logic based on sentiment/state), consider migrating parts of the logic to a LangGraph state machine [[Langchain Docs](https://python.langchain.com/docs/langgraph)]. Needed if simple LCEL becomes unwieldy. [[Langchain Docs](https://python.langchain.com/docs/versions/migrating_memory/conversation_buffer_memory/)].

---

## 6. Personalization and Adaptability (Future Enhancement)

*   **Customizable Personas:**
    *   **Concept:** Allow users to potentially select or influence aspects of Ava's personality or communication style.
    *   **Implementation:** Store user preferences (in User Profile) and use them to dynamically adjust the system prompt or LLM parameters.

---

## 7. Continuous Testing and Refinement

**Issue:** Building a nuanced AI requires iteration and feedback.

*   **A/B Testing:**
    *   **Concept:** Systematically test different prompts, memory configurations, or retrieval strategies with subsets of interactions to identify what works best.
    *   **Implementation:** Requires infrastructure for deploying variations and tracking performance metrics (e.g., user satisfaction, task success, recall accuracy).

*   **User Feedback Loop:**
    *   **Concept:** Implement mechanisms for users to provide explicit feedback (e.g., thumbs up/down on responses, short comments) or implicitly analyze interaction patterns (e.g., conversation length, sentiment progression).
    *   **Implementation:** Add feedback endpoints/UI elements. Create analysis pipeline for feedback to inform prompt/logic adjustments.

---

**Priority:**
1.  **(High)** Task 1 (Personality Basics) & Task 2A (Short-Term Memory Config).
2.  **(High)** Task 2B (Vector Store RAG Basics) & Task 3 (Post-Session Summarization) - Core long-term memory loop.
3.  **(Medium)** Task 4 (Sentiment Analysis & Clarifying Questions).
4.  **(Medium)** Task 1 (Dynamic Prompting), Task 4 (Memory-Driven Storytelling).
5.  **(Medium/Low)** Task 2C (Entity Memory), Task 2B (Structured Profile), Task 6 (Customization).
6.  **(Ongoing/Infrastructure)** Task 7 (Testing & Feedback).
7.  **(As needed)** Task 5 (LangGraph). 