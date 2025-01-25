from typing import Optional, Literal
from langchain_ollama import ChatOllama
import os


class LLM:
    """
    Class for interacting with Ollama.
    """
    def __init__(self, model: str = "llama3.2-vision"):
        """Initialize the LLM class with Ollama client."""

        self.model = model

        # Initialize Ollama client
        self.client = ChatOllama(model=self.model)

    def generate_completion(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a completion using Ollama.

        Args:
            system_prompt (str): The system prompt to guide the model's behavior
            user_prompt (str): The user's input prompt

        Returns:
            str: The generated completion
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.invoke(messages)

        return response.content
