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
            print()  # Add a newline after streaming

        return full_response
