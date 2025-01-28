"""
Tests for utils module.
"""

import os
from pathlib import Path
import pytest
from utils import save_experiment_results

def test_save_experiment_results(tmp_path: Path) -> None:
    """Test saving experiment results to CSV."""
    # Test file path
    test_file = tmp_path / "test_results.csv"
    
    # Test data
    model_name = "test-model"
    dataset_name = "test-dataset"
    paragraphs_by_call = 5
    micro_metrics = {
        "precision": 0.85,
        "recall": 0.82,
        "f1": 0.83,
    }
    macro_metrics = {
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
    assert test_file.exists()
    
    # Read contents and verify
    with open(test_file, "r", encoding="utf-8") as f:
        contents = f.read().strip().split("\n")
    
    # Check header and number of rows
    assert len(contents) == 3  # Header + 2 rows
    
    # Check all metrics from first row are present
    first_row = contents[1]
    for metric_type in ["micro", "macro"]:
        for metric_value in (micro_metrics if metric_type == "micro" else macro_metrics).values():
            assert str(metric_value) in first_row
    
    # Check model names are present
    assert "test-model" in first_row
    assert "another-model" in contents[2]
