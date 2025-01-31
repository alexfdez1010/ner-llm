"""Test the Streamlit app functionality."""

import sys
from typing import Generator, List, Optional
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from ai.llm import LLM, TimeoutException
from ai.extractor_ner import ExtractorNER
from model.category import Category

# Mock streamlit_tags module
mock_st_tags = MagicMock()
mock_st_tags.st_tags.return_value = ["Person", "Organization"]
sys.modules["streamlit_tags"] = mock_st_tags


@pytest.fixture
def app() -> Generator[AppTest, None, None]:
    """Create a test client for the Streamlit app."""
    try:
        at = AppTest.from_file(
            "app.py", default_timeout=30
        )  # Increase timeout to 30 seconds
        at.run()
        yield at
    except Exception as e:
        pytest.fail(f"Failed to create AppTest: {str(e)}")


def find_selectbox_by_label(app: AppTest, label: str) -> Optional[MagicMock]:
    """Helper function to find a selectbox by its label."""
    for selectbox in app.get("selectbox"):
        if selectbox.label == label:
            return selectbox
    return None


def find_text_area_by_label_content(app: AppTest, content: str) -> Optional[MagicMock]:
    """Helper function to find a text area by content in its label."""
    for area in app.get("text_area"):
        if content in str(area.label).lower():
            return area
    return None


def find_button_by_label_content(app: AppTest, content: str) -> Optional[MagicMock]:
    """Helper function to find a button by content in its label."""
    for button in app.get("button"):
        if content in str(button.label):
            return button
    return None


def test_language_options(app: AppTest) -> None:
    """Test that all language options are available."""
    expected_languages: List[str] = ["English", "Spanish", "Italian"]

    language_selector = find_selectbox_by_label(app, "Select language:")
    assert language_selector is not None, "Language selector not found"
    assert all(
        lang in language_selector.options for lang in expected_languages
    ), "Not all expected languages are available"


def test_text_input(app: AppTest) -> None:
    """Test text input area."""
    text_input = find_text_area_by_label_content(app, "analyze")
    assert text_input is not None, "Text input area not found"
    assert (
        "analyze" in str(text_input.label).lower()
    ), "Text input area has incorrect label"


def test_extract_button(app: AppTest) -> None:
    """Test extract button presence."""
    extract_button = find_button_by_label_content(app, "Extract")
    assert extract_button is not None, "Extract button not found"
    assert "Extract Entities" in str(
        extract_button.label
    ), "Extract button has incorrect label"


def test_complete_workflow(app: AppTest) -> None:
    """Test the complete workflow of the NER app."""
    # Initialize app
    app.run()

    # 1. Test language selection
    language_selectbox = find_selectbox_by_label(app, "Select language:")
    assert language_selectbox is not None, "Language selectbox not found"
    language_selectbox.set_value("English")
    app.run()

    # 2. Test text input
    sample_text = """John works at Microsoft as a software engineer. 
    Sarah from Google contacted him about a new project."""
    text_area = find_text_area_by_label_content(app, "enter text to analyze")
    assert text_area is not None, "Text area not found"
    text_area.set_value(sample_text)
    app.run()

    # 3. Test category input using st_tags
    app.session_state["category_names"] = ["Person", "Organization"]
    app.session_state["desc_Person"] = "Names of people"
    app.session_state["desc_Organization"] = "Names of organizations"
    app.run()  # Update state after setting session values

    # 4. Test extract button
    extract_button = find_button_by_label_content(app, "Extract Entities")
    assert extract_button is not None, "Extract button not found"
    assert not extract_button.disabled, "Extract button should be enabled"
    extract_button.click()
    app.run()  # Run after clicking the button

    # 5. Test results section
    # First, verify that results section exists
    results_found = False
    for markdown in app.get("markdown"):
        if "Results" in str(markdown.value):
            results_found = True
            break
    assert results_found, "Results section not found after extraction"

    # Then, look for entities in all markdown elements
    all_markdown_text = " ".join(str(markdown.value) for markdown in app.get("markdown"))
    assert "John" in all_markdown_text, "Expected entity 'John' not found in results"
    assert "Microsoft" in all_markdown_text, "Expected entity 'Microsoft' not found in results"
    assert "Sarah" in all_markdown_text, "Expected entity 'Sarah' not found in results"
    assert "Google" in all_markdown_text, "Expected entity 'Google' not found in results"


def test_error_handling(app: AppTest) -> None:
    """Test error handling in the app."""
    try:
        # 1. Test with missing category descriptions
        app.session_state["desc_Person"] = ""  # Empty description

        text_input = find_text_area_by_label_content(app, "analyze")
        assert text_input is not None, "Text input area not found"
        text_input.set_value("Some test text")

        extract_button = find_button_by_label_content(app, "Extract")
        assert extract_button is not None, "Extract button not found"

        # Trigger the button click
        extract_button.click().run()

        # Check for error message
        error_found = False
        for element in app.get("error"):
            if "Please provide descriptions for all categories" in str(element.value):
                error_found = True
                break
        assert (
            error_found
        ), "Error message not displayed for missing category description"

    except Exception as e:
        pytest.fail(f"Error handling test failed: {str(e)}")
