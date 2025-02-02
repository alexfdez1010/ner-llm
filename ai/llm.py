"""
Implementions of LLM models using Ollama.
"""

import re
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_together import ChatTogether
import ollama
from ollama._client import Message

TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
OLLAMA_TIMEOUT = 600  # Timeout in seconds for Ollama models
MAX_TOKENS = 16384
CONTEXT_SIZE = 4096


class LLM:
    """
    Class for interacting with Ollama.
    """

    def __init__(self, model: str = "llama3.2-vision"):
        """Initialize the LLM class with Ollama client."""
        self.model = model
        self.client = ollama.Client(
            host="http://localhost:11434", timeout=OLLAMA_TIMEOUT
        )

    def create_messages(
        self, system_prompt: str, user_prompt: str
    ) -> list[Message] | list[BaseMessage]:
        """Create message list for the chat model."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_completion(
        self, system_prompt: str, user_prompt: str, stream_output: bool = False
    ) -> str:
        """
        Generate a completion using Ollama, optionally streaming the output.

        Args:
            system_prompt (str): The system prompt to guide the model's behavior
            user_prompt (str): The user's input prompt
            stream_output (bool): Whether to print the response as it arrives (default: False)

        Returns:
            str: The complete generated response
        """
        messages = self.create_messages(system_prompt, user_prompt)
        try:
            if not stream_output:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    options={"num_predict": MAX_TOKENS, "num_ctx": CONTEXT_SIZE},
                )
                return response["message"]["content"]
            response = ""
            for chunk in self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"num_predict": MAX_TOKENS, "num_ctx": CONTEXT_SIZE},
            ):
                content = chunk["message"]["content"]
                print(content, end="", flush=True)
                response += content
            if response:
                print()
            return response
        except Exception as _e:
            return ""


class LRM(LLM):
    """
    Extension of LLM class to be able to work with reasoning models.
    """

    def __init__(self, model: str = "deepseek-r1:14b"):
        """Initialize the LRM class with Ollama client."""
        super().__init__(model=model)

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        stream_output: bool = False,
    ) -> str:
        response = super().generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stream_output=stream_output,
        )

        # Remove the reasoning part, it is enclosed by <think> and </think>
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


class LLMTogether:
    """
    Class for interacting with Together AI models.
    """

    def __init__(self, api_key: str):
        """
        Initialize the LLMTogether class with ChatTogether client.

        Args:
            api_key (str): The API key for Together AI
        """
        self.client = ChatTogether(model=TOGETHER_MODEL, temperature=0, api_key=api_key)

    def create_messages(
        self, system_prompt: str, user_prompt: str
    ) -> list[Message] | list[BaseMessage]:
        """Create message list for the Together chat model."""
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

    def generate_completion(
        self, system_prompt: str, user_prompt: str, stream_output: bool = False
    ) -> str:
        """
        Generate a completion using the Together model, optionally streaming the output.
        """
        messages = self.create_messages(system_prompt, user_prompt)
        if not stream_output:
            response = self.client.invoke(input=messages)
            return response.content
        response = ""
        for chunk in self.client.stream(input=messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            print(content, end="", flush=True)
            response += content
        if response:
            print()
        return response
