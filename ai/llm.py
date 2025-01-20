from typing import Optional
from langchain_together import ChatTogether
from dotenv import load_dotenv
import os

class LLM:
    def __init__(self):
        """Initialize the LLM class with Together AI client."""
        load_dotenv()
        
        self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        
        # Initialize Together AI client
        self.client = ChatTogether(
            model=self.model,
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = 1024
    ) -> str:
        """
        Generate a completion using Together AI.
        
        Args:
            system_prompt (str): The system prompt to guide the model's behavior
            user_prompt (str): The user's input prompt
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
            
        Returns:
            str: The generated completion
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.invoke(
            messages,
            max_tokens=max_tokens
        )
        
        return response.content
