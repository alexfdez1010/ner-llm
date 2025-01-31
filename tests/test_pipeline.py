"""
Tests for the Pipeline class.
"""

from typing import Dict, List, Tuple
from unittest.mock import Mock
import pytest

from model.category import Category
from model.entity import Entity
from pipeline import Pipeline, TokenMetrics
from dataset import Dataset, Instance


@pytest.fixture
def mock_extractor() -> Mock:
    """Create a mock extractor that returns entities based on input text."""
    extractor = Mock()

    def extract_entities_side_effect(categories: List[Category], text: str, sentences_per_call: int = 0) -> List[Entity]:
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
def mock_dataset() -> Mock:
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
def categories() -> List[Category]:
    """Create test categories."""
    return [
        Category("DISEASE", "Names of diseases"),
        Category("DRUG", "Names of medications"),
    ]


@pytest.fixture
def pipeline(mock_extractor: Mock, mock_dataset: Mock, categories: List[Category]) -> Pipeline:
    """Create a pipeline instance for testing."""
    return Pipeline(mock_extractor, mock_dataset, categories)


def test_pipeline_evaluate(mock_extractor: Mock, mock_dataset: Mock, categories: List[Category]) -> None:
    """Test pipeline evaluation with mock components."""
    try:
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
        assert mock_extractor.extract_entities.call_count == 2, "Extractor should be called twice"

        # Verify metrics structure
        for metrics in [micro_metrics, macro_metrics]:
            assert isinstance(metrics, dict), "Metrics should be a dictionary"
            assert all(k in metrics for k in ["precision", "recall", "f1"]), "Missing required metrics"
            assert all(isinstance(v, float) for v in metrics.values()), "Metric values should be floats"
            assert all(0 <= v <= 1 for v in metrics.values()), "Metric values should be between 0 and 1"

    except Exception as e:
        pytest.fail(f"Pipeline evaluation test failed: {str(e)}")


class TestComputeInstanceMetrics:
    """Tests for compute_instance_metrics method."""

    def test_exact_match(self, pipeline: Pipeline) -> None:
        """Test when predictions exactly match gold standard."""
        try:
            gold_bio = ["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]
            pred_bio = ["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]
            metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
            assert metrics == TokenMetrics(tp=3.0, fp=0.0, fn=0.0), "Metrics incorrect for exact match"
        except Exception as e:
            pytest.fail(f"Exact match test failed: {str(e)}")

    def test_no_match(self, pipeline: Pipeline) -> None:
        """Test when predictions completely miss gold standard."""
        try:
            gold_bio = ["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]
            pred_bio = ["O", "O", "O", "O", "O", "O"]
            metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
            assert metrics == TokenMetrics(tp=0.0, fp=0.0, fn=3.0), "Metrics incorrect for no match"
        except Exception as e:
            pytest.fail(f"No match test failed: {str(e)}")

    def test_partial_match_same_entity(self, pipeline: Pipeline) -> None:
        """Test when predictions partially match gold standard for same entity type."""
        try:
            gold_bio = ["O", "B-DISEASE", "I-DISEASE", "O"]
            pred_bio = ["O", "I-DISEASE", "I-DISEASE", "O"]
            metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
            assert metrics == TokenMetrics(tp=1.5, fp=0.5, fn=0.5), "Metrics incorrect for partial match"
        except Exception as e:
            pytest.fail(f"Partial match test failed: {str(e)}")

    def test_wrong_entity_type(self, pipeline: Pipeline) -> None:
        """Test when predictions have wrong entity type."""
        try:
            gold_bio = ["O", "B-DISEASE", "I-DISEASE", "O"]
            pred_bio = ["O", "B-DRUG", "I-DRUG", "O"]
            metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
            assert metrics == TokenMetrics(tp=0.0, fp=2.0, fn=2.0), "Metrics incorrect for wrong entity type"
        except Exception as e:
            pytest.fail(f"Wrong entity type test failed: {str(e)}")

    def test_mixed_scenarios(self, pipeline: Pipeline) -> None:
        """Test with a mix of correct, wrong, and missing predictions."""
        try:
            gold_bio = ["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]
            pred_bio = ["O", "B-DISEASE", "O", "O", "B-SYMPTOM", "O"]
            metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
            assert metrics == TokenMetrics(tp=1.0, fp=1.0, fn=2.0), "Metrics incorrect for mixed scenarios"
        except Exception as e:
            pytest.fail(f"Mixed scenarios test failed: {str(e)}")

    def test_empty_sequences(self, pipeline: Pipeline) -> None:
        """Test with empty sequences."""
        try:
            gold_bio = ["O", "O", "O"]
            pred_bio = ["O", "O", "O"]
            metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
            assert metrics == TokenMetrics(tp=0.0, fp=0.0, fn=0.0), "Metrics incorrect for empty sequences"
        except Exception as e:
            pytest.fail(f"Empty sequences test failed: {str(e)}")

    def test_different_lengths(self, pipeline: Pipeline) -> None:
        """Test with sequences of different lengths."""
        try:
            gold_bio = ["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]
            pred_bio = ["O", "B-DISEASE", "O"]
            with pytest.raises(AssertionError):
                pipeline.compute_instance_metrics(gold_bio, pred_bio)
        except Exception as e:
            pytest.fail(f"Different lengths test failed: {str(e)}")

    def test_single_token_exact_match(self, pipeline: Pipeline) -> None:
        """Test when a single token entity matches exactly."""
        try:
            gold_bio = ["O", "O", "B-FARMACO", "O", "O"]
            pred_bio = ["O", "O", "B-FARMACO", "O", "O"]
            metrics = pipeline.compute_instance_metrics(gold_bio, pred_bio)
            assert metrics == TokenMetrics(tp=1.0, fp=0.0, fn=0.0), "Metrics incorrect for single token exact match"
        except Exception as e:
            pytest.fail(f"Single token exact match test failed: {str(e)}")

    def test_real_text_exact_match(self, pipeline: Pipeline) -> None:
        """Test with real text where entity spans match exactly."""
        try:
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
            assert metrics == TokenMetrics(tp=2.0, fp=0.0, fn=0.0), "Metrics incorrect for real text exact match"
        except Exception as e:
            pytest.fail(f"Real text exact match test failed: {str(e)}")
