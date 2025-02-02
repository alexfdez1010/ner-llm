from typing import List
from unittest.mock import Mock

import pytest

from ai.extractor_ner import Category, Entity, ExtractorNER
from ai.llm import LLM


class TestExtractorNER:
    @pytest.fixture
    def mock_llm(self) -> Mock:
        """Create a mock LLM instance."""
        return Mock()

    @pytest.fixture
    def extractor(self, mock_llm: Mock) -> ExtractorNER:
        """Create an ExtractorNER instance with mock LLM."""
        return ExtractorNER(mock_llm, language="en", example_prompt="")

    def test_extract_entities_basic(
        self, mock_llm: Mock, extractor: ExtractorNER
    ) -> None:
        """Test basic entity extraction functionality."""
        try:
            # Setup
            categories: List[Category] = [
                Category("PERSON", "Names of people"),
                Category("ORG", "Names of organizations"),
            ]
            text = "John works at Google and Mary works at Apple."

            # Mock LLM response
            mock_llm.generate_completion.return_value = """<PERSON>:John
<PERSON>:Mary
<ORG>:Google
<ORG>:Apple"""

            # Execute
            entities = extractor.extract_entities(categories, text)

            # Assert
            expected_entities = [
                Entity("PERSON", "John", (0, 4)),
                Entity("ORG", "Google", (14, 20)),
                Entity("PERSON", "Mary", (25, 29)),
                Entity("ORG", "Apple", (39, 44)),
            ]

            assert len(entities) == len(
                expected_entities
            ), "Number of extracted entities does not match expected"
            for actual, expected in zip(entities, expected_entities):
                assert (
                    actual.category == expected.category
                ), f"Category mismatch for entity {actual.entity}"
                assert (
                    actual.entity == expected.entity
                ), f"Entity text mismatch for {actual.entity}"
                assert (
                    actual.span == expected.span
                ), f"Span mismatch for entity {actual.entity}"
        except Exception as e:
            pytest.fail(f"Basic entity extraction test failed: {str(e)}")

    def test_extract_entities_with_examples(
        self, mock_llm: Mock, extractor: ExtractorNER
    ) -> None:
        """Test entity extraction with example prompts."""
        try:
            # Setup
            categories: List[Category] = [Category("PRODUCT", "Names of products")]
            text = "The new iPhone 15 Pro is amazing"

            # Mock LLM response
            mock_llm.generate_completion.return_value = "<PRODUCT>:iPhone 15 Pro"

            # Execute
            entities = extractor.extract_entities(categories, text)

            # Assert
            expected_entities = [Entity("PRODUCT", "iPhone 15 Pro", (8, 21))]
            assert len(entities) == len(
                expected_entities
            ), "Number of extracted entities does not match expected"
            assert (
                entities[0].category == expected_entities[0].category
            ), "Category mismatch"
            assert (
                entities[0].entity == expected_entities[0].entity
            ), "Entity text mismatch"
            assert entities[0].span == expected_entities[0].span, "Span mismatch"
        except Exception as e:
            pytest.fail(f"Entity extraction with examples test failed: {str(e)}")

    def test_multiple_occurrences(
        self, mock_llm: Mock, extractor: ExtractorNER
    ) -> None:
        """Test handling of multiple occurrences of the same entity."""
        try:
            # Setup
            categories: List[Category] = [Category("COMPANY", "Names of companies")]
            text = "Apple makes great products. I love Apple products."

            # Mock LLM response
            mock_llm.generate_completion.return_value = (
                "<COMPANY>:Apple\n<COMPANY>:Apple"
            )

            # Execute
            entities = extractor.extract_entities(categories, text)

            # Assert
            expected_entities = [
                Entity("COMPANY", "Apple", (0, 5)),
                Entity("COMPANY", "Apple", (35, 40)),
            ]
            assert len(entities) == len(
                expected_entities
            ), "Number of extracted entities does not match expected"
            for actual, expected in zip(entities, expected_entities):
                assert (
                    actual.category == expected.category
                ), f"Category mismatch for entity {actual.entity}"
                assert (
                    actual.entity == expected.entity
                ), f"Entity text mismatch for {actual.entity}"
                assert (
                    actual.span == expected.span
                ), f"Span mismatch for entity {actual.entity}"
        except Exception as e:
            pytest.fail(f"Multiple occurrences test failed: {str(e)}")


@pytest.mark.flaky(retries=3, delay=1)
def test_integration_with_llm() -> None:
    """Test integration with real LLM."""
    try:
        # Setup
        llm = LLM()
        extractor = ExtractorNER(llm, language="en", example_prompt=None)
        categories = get_categories()
        text = get_test_text()

        # Execute
        entities = extractor.extract_entities(categories, text)

        # Verify
        verify_entities(entities, text)
    except TimeoutError:
        pytest.skip("LLM call timed out")
    except Exception as e:
        pytest.fail(f"Integration test failed: {str(e)}")


def get_categories() -> List[Category]:
    """Get test categories."""
    return [
        Category("PERSON", "Names of people"),
        Category("ORG", "Names of organizations or companies"),
        Category("LOCATION", "Names of places, cities, or countries"),
    ]


def get_test_text() -> str:
    """Get test text."""
    return """
    Tim Cook, the CEO of Apple, announced new products at their headquarters in Cupertino.
    Meanwhile, Sundar Pichai from Google was presenting in Mountain View, California.
    """


def verify_entities(entities: List[Entity], text: str) -> None:
    """Verify extracted entities."""
    # Check that we got some entities
    assert len(entities) > 0, "No entities were extracted"

    # Verify each entity's span matches its text
    for entity in entities:
        assert (
            text[entity.span[0] : entity.span[1]] == entity.entity
        ), f"Entity span does not match text for {entity.entity}"

        # Verify entity categories
        assert entity.category in [
            "PERSON",
            "ORG",
            "LOCATION",
        ], f"Invalid category {entity.category} for entity {entity.entity}"

    # Check for specific entities
    expected_entities = [
        Entity("PERSON", "Tim Cook", (5, 13)),
        Entity("ORG", "Apple", (26, 31)),
        Entity("LOCATION", "Cupertino", (76, 85)),
        Entity("PERSON", "Sundar Pichai", (101, 114)),
        Entity("ORG", "Google", (120, 126)),
        Entity("LOCATION", "Mountain View", (144, 157)),
        Entity("LOCATION", "California", (159, 169)),
    ]

    for expected in expected_entities:
        assert any(
            e.category == expected.category and e.entity == expected.entity
            for e in entities
        ), f"Expected entity not found: {expected.category}:{expected.entity}"
