"""
Implementions of LLM models using Ollama.
"""
import re
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether

TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


class LLM:
    """
    Class for interacting with Ollama.
    """

    def __init__(self, model: str = "llama3.2-vision"):
        """Initialize the LLM class with Ollama client."""
        self.model = model

        if model == TOGETHER_MODEL:
            self.client = ChatTogether(
                model=TOGETHER_MODEL,
                temperature=0,
            )
        else:
            self.client = ChatOllama(
                model=self.model, num_predict=-2, num_ctx=128000, temperature=0
            )

    def _create_messages(self, system_prompt: str, user_prompt: str) -> list[BaseMessage]:
        """Create message list for the chat model."""
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        stream_output: bool = False
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
        messages = self._create_messages(system_prompt, user_prompt)

        if not stream_output:
            # For non-streaming, just return the complete response
            response = self.client.invoke(input=messages)
            return response.content

        # For streaming, collect chunks
        chunks = self.client.stream(input=messages)
        response = ""
        for chunk in chunks:
            if hasattr(chunk, "content"):
                content = chunk.content
            else:
                content = str(chunk)
            print(content, end="", flush=True)
            response += content
        if response:
            print()
        return response


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
            stream_output=stream_output
        )

        # Remove the reasoning part, it is enclosed by <think> and </think>
        return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
