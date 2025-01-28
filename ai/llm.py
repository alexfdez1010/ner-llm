"""
Implementions of LLM models using Ollama.
"""

from langchain_ollama import ChatOllama


class LLM:
    """
    Class for interacting with Ollama.
    """
    def __init__(self, model: str = "llama3.2-vision"):
        """Initialize the LLM class with Ollama client."""

        self.model = model

        # Initialize Ollama client
        self.client = ChatOllama(model=self.model, num_predict=-1, num_ctx=128000)

    def generate_completion(self, system_prompt: str, user_prompt: str, stream_output: bool = False) -> str:
        """
        Generate a completion using Ollama, optionally streaming the output.

        Args:
            system_prompt (str): The system prompt to guide the model's behavior
            user_prompt (str): The user's input prompt
            stream_output (bool): Whether to print the response as it arrives (default: False)

        Returns:
            str: The complete generated response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        full_response = ""
        for chunk in self.client.stream(messages):
            if stream_output:
                print(chunk.content, end="", flush=True)
            full_response += chunk.content

        if stream_output:
            print()

        return full_response

class LRM(LLM):
    """
    Extension of LLM class to be able to work with reasoning models.
    """
    def __init__(self, model: str = "deepseek-r1:1.5b"):
        """Initialize the LRM class with Ollama model."""
        super().__init__(model=model)

    def generate_completion(self, system_prompt: str, user_prompt: str, stream_output: bool = False) -> str:
        response = super().generate_completion(system_prompt, user_prompt, stream_output)

        # Remove the reasoning part, it is enclosed by <think> and </think>
        start_idx = response.find("<think>")
        end_idx = response.find("</think>")

        if start_idx != -1 and end_idx != -1:
            return response[end_idx + 8:].strip()

        return response.strip()
