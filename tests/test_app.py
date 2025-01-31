"""Test the Streamlit app functionality."""

import sys
from typing import Generator, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from model.entity import Entity

# Mock streamlit_tags module
mock_st_tags = MagicMock()
mock_st_tags.st_tags.return_value = ["Person", "Organization"]
sys.modules["streamlit_tags"] = mock_st_tags

# Mock LLM and ExtractorNER
mock_llm = MagicMock()
mock_extractor = MagicMock()
mock_extractor.extract_entities.return_value = [
    Entity(entity="John", category="Person", span=(0, 4)),
    Entity(entity="Microsoft", category="Organization", span=(14, 23)),
    Entity(entity="Sarah", category="Person", span=(44, 49)),
    Entity(entity="Google", category="Organization", span=(55, 61)),
]

@pytest.fixture
def app() -> Generator[AppTest, None, None]:
    """Create a test client for the Streamlit app."""
    try:
        at = AppTest.from_file("app.py")
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
    assert all(lang in language_selector.options for lang in expected_languages), \
        "Not all expected languages are available"

def test_text_input(app: AppTest) -> None:
    """Test text input area."""
    text_input = find_text_area_by_label_content(app, "analyze")
    assert text_input is not None, "Text input area not found"
    assert "analyze" in str(text_input.label).lower(), "Text input area has incorrect label"

def test_extract_button(app: AppTest) -> None:
    """Test extract button presence."""
    extract_button = find_button_by_label_content(app, "Extract")
    assert extract_button is not None, "Extract button not found"
    assert "Extract Entities" in str(extract_button.label), "Extract button has incorrect label"

@patch("app.get_llm")
@patch("ai.extractor_ner.ExtractorNER")
def test_complete_workflow(mock_extractor_class: MagicMock, mock_get_llm: MagicMock, app: AppTest) -> None:
    """Test the complete workflow of the NER app."""
    try:
        # Set up mocks
        mock_get_llm.return_value = mock_llm
        mock_extractor_class.return_value = mock_extractor
        
        # 1. Set up session state for categories and descriptions
        app.session_state["desc_Person"] = "Names of people, including first names, last names, or full names"
        app.session_state["desc_Organization"] = "Names of companies, institutions, or other organizations"
        
        # 2. Enter text for analysis
        text_input = find_text_area_by_label_content(app, "analyze")
        assert text_input is not None, "Text input area not found"
        
        sample_text = "John works at Microsoft and collaborates with Sarah from Google."
        text_input.set_value(sample_text)
        
        # Rerun to update state
        app.run()
        
        # 3. Select language (English is default)
        language_selector = find_selectbox_by_label(app, "Select language:")
        assert language_selector is not None, "Language selector not found"
        language_selector.set_value("English")
        
        # 4. Click extract button
        extract_button = find_button_by_label_content(app, "Extract")
        assert extract_button is not None, "Extract button not found"
        
        # The button should be enabled since we have categories and text
        assert not extract_button.disabled, "Extract button should be enabled when all required fields are filled"
        
        # Trigger the button click
        extract_button.click().run()
        
        # 5. Verify results are displayed
        # Check for results section
        markdown_elements = app.get("markdown")
        results_found = False
        for element in markdown_elements:
            if "Results" in str(element.value):
                results_found = True
                break
        assert results_found, "Results section not found after extraction"
        
        # Verify that the extractor was called with correct parameters
        mock_extractor_class.assert_called_once()
        mock_extractor.extract_entities.assert_called_once()
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")
