"""Test the Streamlit app functionality."""

import sys
from typing import Generator, List, Optional
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from ai.llm import LLM, TimeoutException
from ai.extractor_ner import ExtractorNER
from model.category import Category
from model.entity import Entity

# Mock streamlit_tags module
mock_st_tags = MagicMock()
mock_st_tags.st_tags.return_value = ["Person", "Organization"]
sys.modules["streamlit_tags"] = mock_st_tags


@pytest.fixture
def app() -> Generator[AppTest, None, None]:
    """Create a test client for the Streamlit app."""
    try:
        at = AppTest.from_file("app.py", default_timeout=30)  # Increase timeout to 30 seconds
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


def test_complete_workflow(app: AppTest) -> None:
    """Test the complete workflow of the NER app."""
    try:
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
        
        # 6. Test actual LLM and ExtractorNER functionality
        try:
            # Create real instances
            llm = LLM()
            extractor = ExtractorNER(llm=llm, language="en")
            
            # Create test categories
            categories = [
                Category(name="Person", description="Names of people"),
                Category(name="Organization", description="Names of organizations")
            ]
            
            # Extract entities
            entities = extractor.extract_entities(categories=categories, text=sample_text)
            
            # Verify entities
            assert len(entities) > 0, "No entities were extracted"
            
            # Verify we found at least one person and one organization
            found_person = False
            found_org = False
            for entity in entities:
                if entity.category == "Person" and entity.entity in ["John", "Sarah"]:
                    found_person = True
                elif entity.category == "Organization" and entity.entity in ["Microsoft", "Google"]:
                    found_org = True
                
                # Verify spans are within text bounds
                assert 0 <= entity.span[0] < len(sample_text), f"Invalid start span for entity {entity.entity}"
                assert 0 < entity.span[1] <= len(sample_text), f"Invalid end span for entity {entity.entity}"
                assert entity.span[0] < entity.span[1], f"Invalid span range for entity {entity.entity}"
                
            assert found_person, "No person entities were found"
            assert found_org, "No organization entities were found"
            
        except TimeoutException:
            pytest.skip("LLM call timed out")
        except Exception as e:
            pytest.fail(f"Error during LLM testing: {str(e)}")
            
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")


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
        assert error_found, "Error message not displayed for missing category description"
        
    except Exception as e:
        pytest.fail(f"Error handling test failed: {str(e)}")
