"""
Tests for utils module.
"""

import os
from pathlib import Path
from typing import Dict

import pytest
from utils import save_experiment_results


def test_save_experiment_results(tmp_path: Path) -> None:
    """Test saving experiment results to CSV."""
    try:
        # Test file path
        test_file = tmp_path / "test_results.csv"

        # Test data
        model_name = "test-model"
        dataset_name = "test-dataset"
        paragraphs_by_call = 5
        micro_metrics: Dict[str, float] = {
            "precision": 0.85,
            "recall": 0.82,
            "f1": 0.83,
        }
        macro_metrics: Dict[str, float] = {
            "precision": 0.80,
            "recall": 0.78,
            "f1": 0.79,
        }

        # Save first row
        save_experiment_results(
            test_file,
            model_name,
            dataset_name,
            paragraphs_by_call,
            micro_metrics,
            macro_metrics,
        )

        # Save second row with different values
        save_experiment_results(
            test_file,
            "another-model",
            dataset_name,
            10,
            {"precision": 0.90, "recall": 0.88, "f1": 0.89},
            {"precision": 0.87, "recall": 0.86, "f1": 0.86},
        )

        # Check file exists
        assert test_file.exists(), "Results file was not created"

        # Read contents and verify
        with open(test_file, "r", encoding="utf-8") as f:
            contents = f.read().strip().split("\n")

        # Check header and number of rows
        assert len(contents) == 3, "Expected header + 2 rows"

        # Check header format
        header = contents[0].split(",")
        expected_columns = [
            "model_name",
            "dataset",
            "paragraphs_per_call",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ]
        assert header == expected_columns, "Incorrect header columns"

        # Check first row values
        first_row = contents[1].split(",")
        assert first_row[0] == "test-model", "Model name not found in first row"
        assert first_row[1] == "test-dataset", "Dataset name not found in first row"
        assert first_row[2] == "5", "Paragraphs by call not found in first row"
        assert float(first_row[3]) == 0.85, "Micro precision not found in first row"
        assert float(first_row[4]) == 0.82, "Micro recall not found in first row"
        assert float(first_row[5]) == 0.83, "Micro F1 not found in first row"
        assert float(first_row[6]) == 0.80, "Macro precision not found in first row"
        assert float(first_row[7]) == 0.78, "Macro recall not found in first row"
        assert float(first_row[8]) == 0.79, "Macro F1 not found in first row"

        # Check second row values
        second_row = contents[2].split(",")
        assert second_row[0] == "another-model", "Model name not found in second row"
        assert second_row[1] == "test-dataset", "Dataset name not found in second row"
        assert second_row[2] == "10", "Paragraphs by call not found in second row"
        assert float(second_row[3]) == 0.90, "Micro precision not found in second row"
        assert float(second_row[4]) == 0.88, "Micro recall not found in second row"
        assert float(second_row[5]) == 0.89, "Micro F1 not found in second row"
        assert float(second_row[6]) == 0.87, "Macro precision not found in second row"
        assert float(second_row[7]) == 0.86, "Macro recall not found in second row"
        assert float(second_row[8]) == 0.86, "Macro F1 not found in second row"

    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")
