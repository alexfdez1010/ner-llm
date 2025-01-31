"""
Implementions of LLM models using Ollama.
"""
import httpx
import signal
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether

DEFAULT_TIMEOUT = 600
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


def timeout_handler(_signum, _frame):
    """
    Signal handler for timeout.

    Args:
        signum (int): Signal number
        frame (FrameType): Frame object
    
    Raises:
        TimeoutError: If the LLM call times out
    
    """
    raise TimeoutError("LLM call timed out")


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
                model=self.model, num_predict=-1, num_ctx=128000, temperature=0
            )

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        stream_output: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        """
        Generate a completion using Ollama, optionally streaming the output.

        Args:
            system_prompt (str): The system prompt to guide the model's behavior
            user_prompt (str): The user's input prompt
            stream_output (bool): Whether to print the response as it arrives (default: False)
            timeout (int): Timeout in seconds, overrides the default timeout

        Returns:
            str: The complete generated response

        Raises:
            TimeoutError: If the LLM call times out
            Exception: For other errors during LLM call
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Set timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            full_response = ""
            for chunk in self.client.stream(messages):
                if stream_output:
                    print(chunk.content, end="", flush=True)
                full_response += chunk.content

            if stream_output:
                print()

            return full_response

        except (httpx.ReadTimeout, TimeoutError):
            print(full_response)
            return ""
        finally:
            if self.model != TOGETHER_MODEL:
                signal.alarm(0)


class LRM(LLM):
    """
    Extension of LLM class to be able to work with reasoning models.
    """

    def __init__(self, model: str = "deepseek-r1:14b"):
        """Initialize the LRM class with Ollama model."""
        super().__init__(model=model)

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        stream_output: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        response = super().generate_completion(
            system_prompt, user_prompt, stream_output, timeout
        )

        # Remove the reasoning part, it is enclosed by <think> and </think>
        start_idx = response.find("<think>")
        end_idx = response.find("</think>")

        if start_idx != -1 and end_idx != -1:
            return response[end_idx + 8 :].strip()

        return response.strip()
