import logging
from collections.abc import Callable

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from app.config import Config
from app.exceptions import APICallError
from app.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class Chatbot:
    def __init__(self, *, config: Config | None = None, memory_manager: MemoryManager | None = None) -> None:
        self.config = config or Config()
        self.memory_manager = memory_manager or MemoryManager(
            self.config.MEMORY_FILE,
            api_key=self.config.OPENAI_API_KEY,
        )

        self._histories: dict[str, ChatMessageHistory] = {}
        self.chat = self._initialize_chat()
        self.prompt = self._create_prompt()
        self.chain = self._create_chain()

    def _initialize_chat(self) -> ChatOpenAI:
        logger.debug("Initializing ChatOpenAI model.")
        return ChatOpenAI(
            model=self.config.MODEL_NAME,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
            base_url=self.config.BASE_URL,
            api_key=self.config.OPENAI_API_KEY,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

    def _create_prompt(self) -> ChatPromptTemplate:
        logger.debug("Defining prompt template.")
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant named {ai_name}."),
                ("system", "Relevant memory context:\n{memory_context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

    def _create_chain(self) -> RunnableWithMessageHistory:
        def get_session_history(session_id: str) -> ChatMessageHistory:
            if session_id not in self._histories:
                self._histories[session_id] = ChatMessageHistory()
            return self._histories[session_id]

        runnable = self.prompt | self.chat
        return RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    @staticmethod
    def _format_memory_context(memories: list[str], *, limit: int = 8) -> str:
        if not memories:
            return "(none)"
        lines = [f"- {m}" for m in memories[:limit]]
        return "\n".join(lines)

    def generate_response(
        self,
        user_input: str,
        *,
        user_id: str,
        session_id: str,
        on_memory_context: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        try:
            relevant = self.memory_manager.get_relevant_memory(user_id=user_id, query=user_input)
            memory_context = self._format_memory_context(relevant)

            if on_memory_context:
                on_memory_context(memory_context)

            response = self.chain.invoke(
                {
                    "ai_name": self.config.AI_NAME,
                    "memory_context": memory_context,
                    "input": user_input,
                },
                config={"configurable": {"session_id": session_id}},
            )

            ai_text = getattr(response, "content", str(response))

            self.memory_manager.add_memory(user_id=user_id, message=user_input, role="user")
            self.memory_manager.add_memory(user_id=user_id, message=ai_text, role="assistant")

            return ai_text, memory_context
        except Exception as e:
            raise APICallError(f"Error generating response: {e}") from e

