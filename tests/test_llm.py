import pytest
from ai.llm import LLM, LRM


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
    except TimeoutError:
        pytest.skip("LLM call timed out")
    except Exception as e:
        pytest.fail(f"LLM call failed: {str(e)}")


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
        # The response should not contain any reasoning tags since they should be removed
        assert "<think>" not in response
        assert "</think>" not in response
        # Check that we're using a reasoning model but don't assert specific version
        assert any(name in lrm.model.lower() for name in ["deepseek", "mixtral", "llama"])
    except TimeoutError:
        pytest.skip("LLM call timed out")
    except Exception as e:
        pytest.fail(f"LLM call failed: {str(e)}")
