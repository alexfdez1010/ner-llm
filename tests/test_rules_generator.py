import pytest
from unittest.mock import Mock
from ai.llm import LLM
from ai.rules_generator import RulesGenerator
from model.entity import Entity
from model.category import Category


class TestRulesGenerator:
    @pytest.fixture
    def mock_llm(self) -> Mock:
        mock = Mock(spec=LLM)
        mock.generate_completion.return_value = """[
            {
                "label": "PERSON",
                "pattern": [
                    {"LOWER": "john"},
                    {"POS": "PROPN", "OP": "?"}
                ],
                "id": "john_smith"
            }
        ]"""
        return mock

    @pytest.fixture
    def rules_generator(self, mock_llm: Mock) -> RulesGenerator:
        return RulesGenerator(mock_llm)

    def test_generate_rules_with_mock(self, rules_generator: RulesGenerator):
        # Test data
        categories = [Category("PERSON", "Names of people")]
        texts = ["John Smith is a developer"]
        entities = [[Entity("PERSON", "John Smith", (0, 10))]]

        # Generate rules
        rules = rules_generator.generate_rules(
            categories=categories, texts=texts, entities=entities
        )

        # Assertions
        assert isinstance(rules, list)
        assert len(rules) > 0
        assert rules[0]["label"] == "PERSON"
        assert isinstance(rules[0]["pattern"], list)

    def test_generate_rules_with_old_rules(self, rules_generator: RulesGenerator):
        # Test data
        categories = [Category("PERSON", "Names of people")]
        texts = ["John Smith is a developer"]
        entities = [[Entity("PERSON", "John Smith", (0, 10))]]
        old_rules = [
            {"label": "PERSON", "pattern": [{"LOWER": "jane"}], "id": "jane_smith"}
        ]

        # Generate rules
        rules = rules_generator.generate_rules(
            categories=categories, texts=texts, entities=entities, old_rules=old_rules
        )

        # Assertions
        assert isinstance(rules, list)
        assert len(rules) > 1  # Should include both old and new rules
        assert any(r["pattern"][0].get("LOWER") == "jane" for r in rules)

    def test_invalid_llm_response(
        self, rules_generator: RulesGenerator, mock_llm: Mock
    ):
        # Make LLM return invalid JSON
        mock_llm.generate_completion.return_value = "invalid json"

        # Test data
        categories = [Category("PERSON", "Names of people")]
        texts = ["John Smith is a developer"]
        entities = [[Entity("PERSON", "John Smith", (0, 10))]]

        # Assert it raises ValueError with the new error message format
        with pytest.raises(
            ValueError, match="Failed to generate valid rules after 3 attempts"
        ):
            rules_generator.generate_rules(
                categories=categories, texts=texts, entities=entities
            )

    def test_retry_behavior(self, rules_generator: RulesGenerator, mock_llm: Mock):
        """Test that rules generation retries on failure and raises error after max attempts."""
        # Configure mock to return invalid JSON responses
        mock_llm.generate_completion.side_effect = [
            "Invalid JSON 1",  # First attempt fails
            "Also invalid",  # Second attempt fails
            "Still invalid",  # Third attempt fails
        ]

        # Test with default max_attempts (3)
        with pytest.raises(ValueError) as exc_info:
            rules_generator.generate_rules(
                categories=[Category(name="TEST", description="Test category")],
                texts=["Test text"],
                entities=[[]],
            )

        assert "Failed to generate valid rules after 3 attempts" in str(exc_info.value)
        assert mock_llm.generate_completion.call_count == 3

        # Reset mock
        mock_llm.generate_completion.reset_mock()

        # Test with custom max_attempts
        mock_llm.generate_completion.side_effect = [
            "Invalid JSON 1",  # First attempt fails
            '[{"pattern": [{"LOWER": "test"}], "label": "TEST"}]',  # Second attempt succeeds
        ]

        rules = rules_generator.generate_rules(
            categories=[Category(name="TEST", description="Test category")],
            texts=["Test text"],
            entities=[[]],
            max_attempts=2,
        )

        assert len(rules) == 1
        assert mock_llm.generate_completion.call_count == 2

    def test_validate_pattern_token(self, rules_generator: RulesGenerator):
        """Test validation of individual token patterns."""
        # Valid patterns
        assert rules_generator.is_valid_pattern_token({"LOWER": "test"})
        assert rules_generator.is_valid_pattern_token({"POS": "NOUN", "OP": "+"})
        assert rules_generator.is_valid_pattern_token({"IS_DIGIT": True, "OP": "?"})
        assert rules_generator.is_valid_pattern_token({})
        assert rules_generator.is_valid_pattern_token({"OP": "{2,5}"})

        # Invalid patterns
        assert not rules_generator.is_valid_pattern_token({"INVALID": "test"})
        assert not rules_generator.is_valid_pattern_token({"OP": "invalid"})
        assert not rules_generator.is_valid_pattern_token({"OP": "{invalid}"})
        assert not rules_generator.is_valid_pattern_token("not a dict")

    def test_validate_pattern(self, rules_generator: RulesGenerator):
        """Test validation of complete patterns."""
        # Valid pattern
        valid_pattern = {
            "label": "TEST",
            "pattern": [
                {"LOWER": "test"},
                {"POS": "NOUN", "OP": "+"},
                {}  # Wildcard
            ],
            "id": "test"
        }
        assert rules_generator.validate_pattern(valid_pattern)
        
        # Invalid patterns
        invalid_patterns = [
            # Missing label
            {"pattern": [{"LOWER": "test"}]},
            # Missing pattern
            {"label": "TEST"},
            # Invalid token pattern
            {"label": "TEST", "pattern": [{"INVALID": "test"}]},
            # Pattern not a list
            {"label": "TEST", "pattern": "not a list"},
            # Not a dict
            "not a dict"
        ]
        for pattern in invalid_patterns:
            assert not rules_generator.validate_pattern(pattern)

    def test_filter_valid_rules(self, rules_generator: RulesGenerator):
        """Test filtering of valid rules."""
        rules = [
            # Valid rule
            {
                "label": "PERSON",
                "pattern": [{"LOWER": "dr"}, {"TEXT": "."}, {"POS": "PROPN", "OP": "+"}],
                "id": "doctor"
            },
            # Invalid rule (invalid attribute)
            {
                "label": "ORG",
                "pattern": [{"INVALID": "test"}],
                "id": "org"
            },
            # Valid rule
            {
                "label": "NUMBER",
                "pattern": [{"IS_DIGIT": True, "LENGTH": {">=": 2}}],
                "id": "number"
            }
        ]
        
        valid_rules = rules_generator.filter_valid_rules(rules)
        assert len(valid_rules) == 2
        assert valid_rules[0]["label"] == "PERSON"
        assert valid_rules[1]["label"] == "NUMBER"

def test_integration_with_real_llm():
    """Integration test using real LLM with a simple example."""
    # Initialize real components
    llm = LLM()
    generator = RulesGenerator(llm)

    # Simple test case
    categories = [Category("COMPANY", "Names of companies and organizations")]
    texts = ["Apple Inc. announced new products today"]
    entities = [[Entity("COMPANY", "Apple Inc.", (0, 9))]]

    # Generate rules
    rules = generator.generate_rules(
        categories=categories, texts=texts, entities=entities
    )

    # Basic validation
    assert isinstance(rules, list)
    assert len(rules) > 0
    assert all(isinstance(r, dict) for r in rules)
    assert all("label" in r and "pattern" in r for r in rules)
    assert any(r["label"] == "COMPANY" for r in rules)
