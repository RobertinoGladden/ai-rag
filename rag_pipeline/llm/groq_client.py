from typing import List, Optional, Iterator
from loguru import logger
from langchain_groq import ChatGroq
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ..config import get_settings


class GroqClient:
    """
    Wrapper di atas ChatGroq dari LangChain.
    Mendukung regular call dan streaming.
    """

    def __init__(self, streaming: bool = False):
        settings = get_settings()
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else []

        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            streaming=streaming,
            callbacks=callbacks,
        )
        self.model_name = settings.groq_model
        logger.info(f"Groq client initialized. Model: {settings.groq_model}")

    def invoke(self, messages: List[BaseMessage]) -> str:
        """Kirim messages dan return response string."""
        response = self.llm.invoke(messages)
        return response.content

    def stream(self, messages: List[BaseMessage]) -> Iterator[str]:
        """Streaming response — yield token per token."""
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def get_langchain_llm(self):
        """Return raw LangChain LLM object untuk dipakai di chain."""
        return self.llm

    @staticmethod
    def build_messages(
        question: str,
        chat_history: Optional[List[dict]] = None,
    ) -> List[BaseMessage]:
        """
        Convert chat history format ke LangChain messages.
        chat_history format: [{"role": "user"/"assistant", "content": "..."}]
        """
        messages = []
        for msg in (chat_history or []):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=question))
        return messages
