from unittest.mock import Mock, patch
import pytest
import tempfile
import os
from typing import Any

from pipeline import Pipeline
from model.entity import Entity
from model.category import Category
from dataset import Dataset, Instance


@pytest.fixture
def mock_components():
    """Fixture to create mock components for testing."""
    # Create mock instances
    mock_extractor = Mock()
    mock_rules_generator = Mock()

    # Create a simple dataset with 5 training instances
    training_instances = [
        Instance(tokens=["John", "lives", "in", "London"], labels=[1, 0, 0, 2]),
        Instance(tokens=["Mary", "visited", "Paris"], labels=[1, 0, 2]),
    ]

    validation_instances = [
        Instance(tokens=["Peter", "works", "in", "Berlin"], labels=[1, 0, 0, 2]),
    ]

    dataset = Dataset(
        training=training_instances,
        validation=validation_instances,
        test=[],
        index_to_category={0: None, 1: "PERSON", 2: "LOCATION"},
    )

    categories = [
        Category(name="PERSON", description="Person"),
        Category(name="LOCATION", description="Location"),
    ]

    # Configure mock rules generator
    def generate_rules(categories, texts, entities, old_rules=None):
        # Simple mock that just adds one rule per iteration
        current_rules = old_rules or []
        new_rule = {"pattern": [{"LOWER": "test"}], "label": "TEST"}
        return current_rules + [new_rule]

    mock_rules_generator.generate_rules.side_effect = generate_rules

    # Configure mock extractor
    def extract_entities(text: str, rules: list[dict[str, Any]]) -> list[Entity]:
        # Simple mock that extracts entities based on token position
        tokens = text.split()
        entities = []
        
        # Mock entity extraction for test data
        if "John" in tokens:
            entities.append(Entity(category="PERSON", entity="John", span=(text.index("John"), text.index("John") + len("John"))))
        if "London" in tokens:
            entities.append(Entity(category="LOCATION", entity="London", span=(text.index("London"), text.index("London") + len("London"))))
        if "Peter" in tokens:
            entities.append(Entity(category="PERSON", entity="Peter", span=(text.index("Peter"), text.index("Peter") + len("Peter"))))
        if "Berlin" in tokens:
            entities.append(Entity(category="LOCATION", entity="Berlin", span=(text.index("Berlin"), text.index("Berlin") + len("Berlin"))))
        
        return entities

    mock_extractor.extract_entities.side_effect = extract_entities

    return mock_extractor, mock_rules_generator, dataset, categories


def test_pipeline_execution(mock_components):
    """Test that pipeline executes correctly for a small number of iterations."""
    mock_extractor, mock_rules_generator, dataset, categories = mock_components

    # Create pipeline
    pipeline = Pipeline(
        extractor=mock_extractor, rules_generator=mock_rules_generator, dataset=dataset
    )

    # Create temporary file for output
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        output_file = tmp.name

        try:
            # Execute pipeline for 2 iterations with 50% of data
            pipeline.execute(
                output_file=output_file,
                num_iterations=2,
                categories=categories,
                sample_percentage=0.5,
            )

            # Verify rules generator was called twice
            assert mock_rules_generator.generate_rules.call_count == 2

            # Verify each call used the correct number of instances
            for call_args in mock_rules_generator.generate_rules.call_args_list:
                _, kwargs = call_args
                texts = kwargs["texts"]  # texts is passed as a keyword argument
                # With 2 total instances and 50% sampling, should use 1 instance
                assert len(texts) == 1

            # Verify final rules were saved
            with open(output_file, "r") as f:
                import json

                rules = json.load(f)
                # Should have 2 rules (one added per iteration)
                assert len(rules) == 2
                # Verify rule structure
                assert all("pattern" in rule and "label" in rule for rule in rules)

        finally:
            # Clean up temporary file
            os.unlink(output_file)


def test_pipeline_empty_dataset(mock_components):
    """Test pipeline behavior with an empty dataset."""
    mock_extractor, mock_rules_generator, _, categories = mock_components

    # Create empty dataset
    empty_dataset = Dataset(
        training=[],
        validation=[],
        test=[],
        index_to_category={0: None, 1: "PERSON", 2: "LOCATION"},
    )

    pipeline = Pipeline(
        extractor=mock_extractor,
        rules_generator=mock_rules_generator,
        dataset=empty_dataset,
    )

    # Create temporary file for output
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        output_file = tmp.name

        try:
            # Execute pipeline
            pipeline.execute(
                output_file=output_file,
                num_iterations=2,
                categories=categories,
                sample_percentage=0.5,
            )

            # Verify rules generator was still called
            assert mock_rules_generator.generate_rules.call_count == 2

            # Verify each call used empty list of texts (since dataset is empty)
            for call_args in mock_rules_generator.generate_rules.call_args_list:
                _, kwargs = call_args
                texts = kwargs["texts"]  # texts is passed as a keyword argument
                assert len(texts) == 0  # Should be empty since dataset is empty

            # Verify final rules were saved
            with open(output_file, "r") as f:
                import json

                rules = json.load(f)
                # Should have 2 rules (one added per iteration)
                assert len(rules) == 2

        finally:
            # Clean up temporary file
            os.unlink(output_file)


def test_pipeline_evaluate(mock_components):
    """Test the evaluation method of the pipeline."""
    mock_extractor, mock_rules_generator, dataset, categories = mock_components
    
    # Create pipeline
    pipeline = Pipeline(mock_extractor, mock_rules_generator, dataset)
    
    rules = [
        {"pattern": [{"LOWER": "peter"}], "category": "PERSON"},
        {"pattern": [{"LOWER": "berlin"}], "category": "LOCATION"}
    ]
    
    # Test evaluation on validation set
    metrics = pipeline.evaluate(rules, subset="validation")
    
    # Since our mock extractor perfectly matches the validation data
    # we expect perfect scores
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    
    # Test evaluation on training set
    metrics = pipeline.evaluate(rules, subset="training")
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.6666666666666666
