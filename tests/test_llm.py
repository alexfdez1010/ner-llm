import pytest
from ai.llm import LLM, LRM, TimeoutException


@pytest.fixture
def llm() -> LLM:
    """Fixture to create an LLM instance."""
    return LLM()


@pytest.fixture
def lrm() -> LRM:
    """Fixture to create an LRM instance."""
    return LRM()


def test_generate_completion(llm: LLM) -> None:
    """Test the generate_completion method."""
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is 2+2?"

    try:
        response = llm.generate_completion(
            system_prompt=system_prompt, user_prompt=user_prompt, timeout=30
        )

        assert isinstance(response, str)
        assert len(response) > 0
    except TimeoutException:
        pytest.skip("LLM call timed out")
    except Exception as e:
        pytest.fail(f"LLM call failed: {str(e)}")


def test_generate_completion_with_timeout(llm: LLM) -> None:
    """Test that the generate_completion method properly handles timeouts."""
    system_prompt = "You are a helpful assistant."
    user_prompt = "Write a very long essay about the history of the world."

    try:
        response = llm.generate_completion(
            system_prompt=system_prompt, user_prompt=user_prompt, timeout=1
        )

        assert isinstance(response, str)
    except TimeoutException:
        pass
    except Exception as e:
        pytest.fail(f"Unexpected error during timeout test: {str(e)}")


def test_lrm_generate_completion(lrm: LRM) -> None:
    """Test the LRM generate_completion method with reasoning."""
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is 2+2?"

    try:
        response = lrm.generate_completion(
            system_prompt=system_prompt, user_prompt=user_prompt, timeout=30
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert "<think>" not in response
        assert "</think>" not in response
        assert any(name in lrm.model.lower() for name in ["deepseek", "mixtral", "llama"])
    except TimeoutException:
        pytest.skip("LLM call timed out")
    except Exception as e:
        pytest.fail(f"LLM call failed: {str(e)}")


def test_stream_output(llm: LLM) -> None:
    """Test streaming output functionality."""
    system_prompt = "You are a helpful assistant."
    user_prompt = "Count from 1 to 5."

    try:
        response = llm.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stream_output=True,
            timeout=30
        )

        assert isinstance(response, str)
        assert len(response) > 0
    except TimeoutException:
        pytest.skip("LLM call timed out")
    except Exception as e:
        pytest.fail(f"LLM call failed during streaming: {str(e)}")
