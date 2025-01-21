from unittest.mock import Mock, patch
import pytest
import tempfile
import os

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
        Instance(tokens=["This", "is", "text", str(i)], labels=[1, 0, 0, 2])
        for i in range(5)
    ]
    
    dataset = Dataset(
        training=training_instances,
        validation=[],
        test=[],
        index_to_category={0: None, 1: "PERSON", 2: "LOCATION"}
    )
    
    # Configure mock rules generator
    def generate_rules(categories, texts, entities, old_rules=None):
        # Simple mock that just adds one rule per iteration
        current_rules = old_rules or []
        new_rule = {"pattern": [{"LOWER": "test"}], "label": "TEST"}
        return current_rules + [new_rule]
    
    mock_rules_generator.generate_rules.side_effect = generate_rules
    
    return mock_extractor, mock_rules_generator, dataset


def test_pipeline_execution(mock_components):
    """Test that pipeline executes correctly for a small number of iterations."""
    mock_extractor, mock_rules_generator, dataset = mock_components
    
    # Create pipeline
    pipeline = Pipeline(
        extractor=mock_extractor,
        rules_generator=mock_rules_generator,
        dataset=dataset
    )
    
    # Create temporary file for output
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        output_file = tmp.name
        
        try:
            # Execute pipeline for 2 iterations with 50% of data
            pipeline.execute(
                output_file=output_file,
                num_iterations=2,
                sample_percentage=0.5
            )
            
            # Verify rules generator was called twice
            assert mock_rules_generator.generate_rules.call_count == 2
            
            # Verify each call used the correct number of instances
            for call_args in mock_rules_generator.generate_rules.call_args_list:
                _, kwargs = call_args
                texts = kwargs['texts']  # texts is passed as a keyword argument
                # With 5 total instances and 50% sampling, should use 2-3 instances
                assert 2 <= len(texts) <= 3
            
            # Verify final rules were saved
            with open(output_file, 'r') as f:
                import json
                rules = json.load(f)
                # Should have 2 rules (one added per iteration)
                assert len(rules) == 2
                # Verify rule structure
                assert all('pattern' in rule and 'label' in rule for rule in rules)
        
        finally:
            # Clean up temporary file
            os.unlink(output_file)


def test_pipeline_empty_dataset(mock_components):
    """Test pipeline behavior with an empty dataset."""
    mock_extractor, mock_rules_generator, _ = mock_components
    
    # Create empty dataset
    empty_dataset = Dataset(
        training=[],
        validation=[],
        test=[],
        index_to_category={0: None, 1: "PERSON", 2: "LOCATION"}
    )
    
    pipeline = Pipeline(
        extractor=mock_extractor,
        rules_generator=mock_rules_generator,
        dataset=empty_dataset
    )
    
    # Create temporary file for output
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        output_file = tmp.name
        
        try:
            # Execute pipeline
            pipeline.execute(
                output_file=output_file,
                num_iterations=2,
                sample_percentage=0.5
            )
            
            # Verify rules generator was still called
            assert mock_rules_generator.generate_rules.call_count == 2
            
            # Verify each call used empty list of texts (since dataset is empty)
            for call_args in mock_rules_generator.generate_rules.call_args_list:
                _, kwargs = call_args
                texts = kwargs['texts']  # texts is passed as a keyword argument
                assert len(texts) == 0  # Should be empty since dataset is empty
            
            # Verify final rules were saved
            with open(output_file, 'r') as f:
                import json
                rules = json.load(f)
                # Should have 2 rules (one added per iteration)
                assert len(rules) == 2
        
        finally:
            # Clean up temporary file
            os.unlink(output_file)
