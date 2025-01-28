from unittest.mock import Mock

import pytest

from ai.extractor_ner import Category, Entity, ExtractorNER
from ai.llm import LLM


class TestExtractorNER:
    @pytest.fixture
    def mock_llm(self):
        return Mock()

    @pytest.fixture
    def extractor(self, mock_llm):
        return ExtractorNER(mock_llm, language="en", example_prompt="")

    def test_extract_entities_basic(self, mock_llm, extractor):
        # Setup
        categories = [
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

        assert len(entities) == len(expected_entities)
        for actual, expected in zip(entities, expected_entities):
            assert actual.category == expected.category
            assert actual.entity == expected.entity
            assert actual.span == expected.span

    def test_extract_entities_with_examples(self, mock_llm, extractor):
        # Setup
        categories = [Category("PRODUCT", "Names of products")]
        text = "The new iPhone 15 Pro is amazing"
        examples = "Text: The MacBook Pro is great\nOutput: <PRODUCT>:MacBook Pro"

        # Mock LLM response
        mock_llm.generate_completion.return_value = "<PRODUCT>:iPhone 15 Pro"

        # Execute
        entities = extractor.extract_entities(categories, text, examples=examples)

        # Assert
        expected_entities = [Entity("PRODUCT", "iPhone 15 Pro", (8, 21))]
        assert len(entities) == len(expected_entities)
        assert entities[0].category == expected_entities[0].category
        assert entities[0].entity == expected_entities[0].entity
        assert entities[0].span == expected_entities[0].span

    def test_multiple_occurrences(self, mock_llm, extractor):
        # Setup
        categories = [Category("COMPANY", "Names of companies")]
        text = "Apple makes great products. I love Apple products."

        # Mock LLM response
        mock_llm.generate_completion.return_value = "<COMPANY>:Apple\n<COMPANY>:Apple"

        # Execute
        entities = extractor.extract_entities(categories, text)

        # Assert
        expected_entities = [
            Entity("COMPANY", "Apple", (0, 5)),
            Entity("COMPANY", "Apple", (35, 40)),
        ]
        assert len(entities) == len(expected_entities)
        for actual, expected in zip(entities, expected_entities):
            assert actual.category == expected.category
            assert actual.entity == expected.entity
            assert actual.span == expected.span

def test_integration_with_llm():
    # Setup
    llm = LLM()
    extractor = ExtractorNER(llm, language="en", example_prompt=None)
    categories = get_categories()
    text = get_test_text()

    entities = extractor.extract_entities(categories, text)
    verify_entities(entities, text)

def get_categories():
    return [
        Category("PERSON", "Names of people"),
        Category("ORG", "Names of organizations or companies"),
        Category("LOCATION", "Names of places, cities, or countries"),
    ]

def get_test_text():
    return """
    Tim Cook, the CEO of Apple, announced new products at their headquarters in Cupertino.
    Meanwhile, Sundar Pichai from Google was presenting in Mountain View, California.
    """

def verify_entities(entities, text):
    assert len(entities) > 0
    for entity in entities:
        assert hasattr(entity, "category")
        assert hasattr(entity, "entity")
        assert hasattr(entity, "span")
        assert isinstance(entity.span, tuple)
        assert len(entity.span) == 2
        # Verify that the span actually matches the entity in the text
        start, end = entity.span
        assert text[start:end] == entity.entity

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
