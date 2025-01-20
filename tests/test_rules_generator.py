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
        mock.generate_completion.return_value = '''[
            {
                "label": "PERSON",
                "pattern": [
                    {"LOWER": "john"},
                    {"POS": "PROPN", "OP": "?"}
                ]
            }
        ]'''
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
            categories=categories,
            texts=texts,
            entities=entities
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
        old_rules = [{
            "label": "PERSON",
            "pattern": [{"LOWER": "jane"}]
        }]
        
        # Generate rules
        rules = rules_generator.generate_rules(
            categories=categories,
            texts=texts,
            entities=entities,
            old_rules=old_rules
        )
        
        # Assertions
        assert isinstance(rules, list)
        assert len(rules) > 1  # Should include both old and new rules
        assert any(r["pattern"][0].get("LOWER") == "jane" for r in rules)

    def test_invalid_llm_response(self, rules_generator: RulesGenerator, mock_llm: Mock):
        # Make LLM return invalid JSON
        mock_llm.generate_completion.return_value = "invalid json"
        
        # Test data
        categories = [Category("PERSON", "Names of people")]
        texts = ["John Smith is a developer"]
        entities = [[Entity("PERSON", "John Smith", (0, 10))]]
        
        # Assert it raises ValueError
        with pytest.raises(ValueError, match="Generated rules are not in valid JSON format"):
            rules_generator.generate_rules(
                categories=categories,
                texts=texts,
                entities=entities
            )

def test_integration_with_real_llm():
    """Integration test using real LLM with a simple example."""
    # Initialize real components
    llm = LLM()
    generator = RulesGenerator(llm)
    
    # Simple test case
    categories = [
        Category("COMPANY", "Names of companies and organizations")
    ]
    texts = ["Apple Inc. announced new products today"]
    entities = [[Entity("COMPANY", "Apple Inc.", (0, 9))]]
    
    # Generate rules
    rules = generator.generate_rules(
        categories=categories,
        texts=texts,
        entities=entities
    )
    
    # Basic validation
    assert isinstance(rules, list)
    assert len(rules) > 0
    assert all(isinstance(r, dict) for r in rules)
    assert all("label" in r and "pattern" in r for r in rules)
    assert any(r["label"] == "COMPANY" for r in rules)