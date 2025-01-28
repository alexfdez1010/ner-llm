import pytest
from ai.llm import LLM, LRM


@pytest.fixture
def llm():
    """Fixture to create an LLM instance."""
    return LLM()


@pytest.fixture
def lrm():
    """Fixture to create an LRM instance."""
    return LRM()


def test_generate_completion(llm):
    """Test the generate_completion method."""
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is 2+2?"

    response = llm.generate_completion(
        system_prompt=system_prompt, user_prompt=user_prompt
    )

    assert isinstance(response, str)
    assert len(response) > 0


def test_lrm_generate_completion(lrm):
    """Test the LRM generate_completion method with reasoning."""
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is 2+2?"

    response = lrm.generate_completion(
        system_prompt=system_prompt, user_prompt=user_prompt
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert "<think>" not in response
    assert "</think>" not in response
    assert lrm.model == "deepseek-r1:1.5b"
