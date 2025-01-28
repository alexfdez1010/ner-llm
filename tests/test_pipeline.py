"""
Tests for the Pipeline class.
"""

from unittest.mock import Mock, patch
import pytest

from model.category import Category
from model.entity import Entity
from pipeline import Pipeline
from dataset import Dataset, Instance


@pytest.fixture
def mock_extractor():
    """Create a mock extractor that returns entities based on input text."""
    extractor = Mock()
    
    def extract_entities_side_effect(categories, text, sentences_per_call=0):
        if "first" in text:
            return [
                Entity("DISEASE", "diabetes", (15, 23)),  # Matches token boundaries
                Entity("DRUG", "metformin", (29, 38)),    # Matches token boundaries
            ]
        else:
            return [
                Entity("DISEASE", "diabetes", (16, 24)),  # Matches token boundaries
                Entity("DRUG", "metformin", (30, 39)),    # Matches token boundaries
            ]
    
    extractor.extract_entities.side_effect = extract_entities_side_effect
    return extractor


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with test instances."""
    dataset = Mock(spec=Dataset)
    
    # Create test instances with different texts to trigger different mock responses
    # Each entity span matches exactly with token boundaries
    instances = [
        Instance(
            text="first has test diabetes takes metformin now",  # Simple space-separated tokens
            entities=[
                Entity("DISEASE", "diabetes", (15, 23)),
                Entity("DRUG", "metformin", (29, 38)),
            ],
        ),
        Instance(
            text="second has test diabetes takes metformin now",  # Simple space-separated tokens
            entities=[
                Entity("DISEASE", "diabetes", (16, 24)),
                Entity("DRUG", "metformin", (30, 39)),
            ],
        ),
    ]
    
    dataset.get_instances.return_value = instances
    return dataset


@pytest.fixture
def categories():
    """Create test categories."""
    return [
        Category("DISEASE", "Names of diseases"),
        Category("DRUG", "Names of medications"),
    ]


def test_pipeline_evaluate(mock_extractor, mock_dataset, categories):
    """Test pipeline evaluation with mock components."""
    # Create pipeline
    pipeline = Pipeline(mock_extractor, mock_dataset, categories)
    
    # Get an instance to check BIO annotations
    instance = mock_dataset.get_instances()[0]
    gold_bio = instance.get_bio_annotations()
    
    # Create predicted instance with same entities
    predicted_instance = Instance(
        text=instance.text,
        entities=[
            Entity("DISEASE", "diabetes", (15, 23)),
            Entity("DRUG", "metformin", (29, 38)),
        ],
    )
    pred_bio = predicted_instance.get_bio_annotations()
    
    print("\nText tokens:", instance.text.split())
    print("Gold BIO annotations:", gold_bio)
    print("Pred BIO annotations:", pred_bio)
    
    # Run evaluation
    micro_metrics, macro_metrics = pipeline.evaluate()
    
    # Verify extractor was called correctly
    assert mock_extractor.extract_entities.call_count == 2
    
    # Verify metrics structure
    for metrics in [micro_metrics, macro_metrics]:
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        # Check metric values are valid
        for value in metrics.values():
            assert isinstance(value, float)
            assert 0 <= value <= 1
    
    # Since our mock returns perfect predictions, metrics should be 1.0
    assert all(v == 1.0 for v in micro_metrics.values())
    assert all(v == 1.0 for v in macro_metrics.values())
