"""
Implementions of LLM models using Ollama.
"""
import concurrent.futures
from typing import List
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether

DEFAULT_TIMEOUT = 600
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


class TimeoutException(Exception):
    """Exception raised when LLM call times out."""
    pass


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

    def _create_messages(self, system_prompt: str, user_prompt: str) -> List[BaseMessage]:
        """Create message list for the chat model."""
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

    def _execute_with_timeout(self, messages: List[BaseMessage], timeout: int) -> str:
        """Execute LLM call with timeout using concurrent.futures."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.client.invoke, messages)
            try:
                result = future.result(timeout=timeout)
                return result.content if isinstance(result, AIMessage) else str(result)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutException("LLM call timed out")
            except Exception as e:
                future.cancel()
                raise e

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
            TimeoutException: If the LLM call times out
            Exception: For other errors during LLM call
        """
        messages = self._create_messages(system_prompt, user_prompt)
        
        try:
            if stream_output:
                # For streaming, collect chunks with timeout
                response = ""
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(lambda: list(self.client.stream(messages)))
                    try:
                        chunks = future.result(timeout=timeout)
                        for chunk in chunks:
                            if hasattr(chunk, 'content'):
                                content = chunk.content
                            else:
                                content = str(chunk)
                            print(content, end="", flush=True)
                            response += content
                        if response:
                            print()
                        return response
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        print("\nLLM streaming timed out")
                        return response if response else ""
                    except Exception as e:
                        future.cancel()
                        print(f"\nError during streaming: {e}")
                        return response if response else ""
            else:
                # For non-streaming, use simple timeout
                return self._execute_with_timeout(messages, timeout)

        except TimeoutException:
            print("\nLLM call timed out")
            return ""
        except Exception as e:
            print(f"\nError during LLM call: {e}")
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
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        response = super().generate_completion(
            system_prompt, user_prompt, stream_output, timeout
        )

        # Remove the reasoning part, it is enclosed by <think> and </think>
        start_idx = response.find("<think>")
        end_idx = response.find("</think>")

        if start_idx != -1 and end_idx != -1:
            return response[end_idx + 8:].strip()

        return response.strip()
