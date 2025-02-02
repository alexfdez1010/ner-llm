"""Test the Streamlit app functionality."""

import sys
from typing import Generator, List, Optional
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

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


def find_text_area_by_key(app: AppTest, key: str) -> Optional[MagicMock]:
    """Helper function to find a text area by its key."""
    for area in app.get("text_area"):
        if area.key == key:
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
    """Test the complete workflow of the app."""
    # Initialize app
    app.run()

    # 1. Set up categories with descriptions
    app.session_state["category_names"] = ["Person", "Organization"]
    app.run()

    # 2. Enter text with known entities
    text_area = find_text_area_by_label_content(app, "enter text to analyze")
    assert text_area is not None, "Text area not found"
    text_area.set_value("John works at Microsoft as a software engineer.")
    app.run()

    # 3. Click extract button
    extract_button = find_button_by_label_content(app, "Extract")
    assert extract_button is not None, "Extract button not found"
    assert not extract_button.disabled, "Extract button should be enabled"
    extract_button.click()
    app.run()

    # 4. Verify results
    # Check that results section appears
    results_found = False
    for markdown in app.markdown:
        if markdown.value and "Results" in str(markdown.value):
            results_found = True
            break
    assert results_found, "Results section not found"

    # Check that entities table appears
    entities_found = False
    for markdown in app.markdown:
        if markdown.value and "Extracted Entities" in str(markdown.value):
            entities_found = True
            break
    assert entities_found, "Entities table not found"

    # Check that entities are in the table
    table_found = False
    person_found = False
    org_found = False
    for table in app.table:
        if not table.value.empty:
            table_found = True
            for _, row in table.value.iterrows():
                if row.get("Category") == "Person" and row.get("Entity") == "John":
                    person_found = True
                if (
                    row.get("Category") == "Organization"
                    and row.get("Entity") == "Microsoft"
                ):
                    org_found = True
    assert table_found, "Table with entities not found"
    assert person_found, "Person entity not found"
    assert org_found, "Organization entity not found"
