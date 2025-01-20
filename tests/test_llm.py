import pytest
from ai.llm import LLM
import os
from dotenv import load_dotenv

@pytest.fixture
def llm():
    """Fixture to create an LLM instance."""
    return LLM()

def test_generate_completion(llm):
    """Test the generate_completion method."""
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is 2+2?"
    
    response = llm.generate_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=50
    )
    
    assert isinstance(response, str)
    assert len(response) > 0
