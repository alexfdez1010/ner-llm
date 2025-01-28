"""
Tests for the Pipeline class.
"""

from unittest.mock import Mock
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
                Entity("DRUG", "metformin", (29, 38)),  # Matches token boundaries
            ]
        else:
            return [
                Entity("DISEASE", "diabetes", (16, 24)),  # Matches token boundaries
                Entity("DRUG", "metformin", (30, 39)),  # Matches token boundaries
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


@pytest.fixture
def pipeline(mock_extractor, mock_dataset, categories):
    """Create a pipeline instance for testing."""
    return Pipeline(mock_extractor, mock_dataset, categories)


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


class TestComputeInstanceMetrics:
    """Tests for compute_instance_metrics method."""

    def test_exact_match(self, pipeline):
        """Test when predictions exactly match gold standard."""
        gold_bio = ["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]
        pred_bio = ["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]

        metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
        assert metrics.tp == 3  # B-DISEASE, I-DISEASE, B-DRUG
        assert metrics.fp == 0
        assert metrics.fn == 0

    def test_no_match(self, pipeline):
        """Test when predictions completely miss gold standard."""
        gold_bio = ["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]
        pred_bio = ["O", "O", "O", "O", "O", "O"]

        metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
        assert metrics.tp == 0
        assert metrics.fp == 0
        assert metrics.fn == 3  # Missed B-DISEASE, I-DISEASE, B-DRUG

    def test_partial_match_same_entity(self, pipeline):
        """Test when predictions partially match gold standard for same entity type."""
        gold_bio = ["O", "B-DISEASE", "I-DISEASE", "O"]
        pred_bio = ["O", "I-DISEASE", "I-DISEASE", "O"]

        metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
        assert metrics.tp == 1.5  # Full match for I-DISEASE, partial for B/I mismatch
        assert metrics.fp == 0.5  # Penalty for B/I mismatch
        assert metrics.fn == 0.5  # Penalty for B/I mismatch

    def test_wrong_entity_type(self, pipeline):
        """Test when predictions get the wrong entity type."""
        gold_bio = ["O", "B-DISEASE", "I-DISEASE", "O"]
        pred_bio = ["O", "B-DRUG", "I-DRUG", "O"]

        metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
        assert metrics.tp == 0
        assert metrics.fp == 2  # Wrong B-DRUG, I-DRUG
        assert metrics.fn == 2  # Missed B-DISEASE, I-DISEASE

    def test_mixed_scenarios(self, pipeline):
        """Test with a mix of correct, incorrect, and partial matches."""
        gold_bio = ["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]
        pred_bio = ["O", "I-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]

        metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
        assert (
            metrics.tp == 2.5
        )  # Full match for I-DISEASE and B-DRUG, partial for B/I mismatch
        assert metrics.fp == 0.5  # Penalty for B/I mismatch
        assert metrics.fn == 0.5  # Penalty for B/I mismatch

    def test_empty_sequences(self, pipeline):
        """Test with empty sequences."""
        metrics = pipeline.compute_instance_metrics([], [])
        assert metrics.tp == 0
        assert metrics.fp == 0
        assert metrics.fn == 0

    def test_different_lengths(self, pipeline):
        """Test with sequences of different lengths."""
        gold_bio = ["O", "B-DISEASE", "I-DISEASE"]
        pred_bio = ["O", "B-DISEASE"]

        with pytest.raises(AssertionError):
            pipeline.compute_instance_metrics(gold_bio, pred_bio)

    def test_single_token_exact_match(self, pipeline):
        """Test when a single token entity matches exactly."""
        gold_bio = ["O", "O", "B-FARMACO", "O", "O"]
        pred_bio = ["O", "O", "B-FARMACO", "O", "O"]

        metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
        assert metrics.tp == 1  # Exact match for B-FARMACO
        assert metrics.fp == 0
        assert metrics.fn == 0

    def test_real_text_exact_match(self, pipeline):
        """Test with real text where entity spans match exactly."""
        text = "The patient was prescribed amiodarone for arrhythmia."
        gold_instance = Instance(
            text=text,
            entities=[Entity("FARMACO", "amiodarone", (24, 34))]
        )
        pred_instance = Instance(
            text=text,
            entities=[Entity("FARMACO", "amiodarone", (24, 34))]
        )
        
        gold_bio = gold_instance.get_bio_annotations()
        pred_bio = pred_instance.get_bio_annotations()
        
        metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
        assert metrics.tp == 2  # B-FARMACO and I-FARMACO match
        assert metrics.fp == 0
        assert metrics.fn == 0
