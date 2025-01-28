"""
Utilities for handling experiment results and metrics.
"""

import csv
from pathlib import Path
from typing import TypeAlias

# Type aliases
MetricsDict: TypeAlias = dict[str, float]
ResultRow: TypeAlias = dict[str, str | float | int]


def save_experiment_results(
    results_file: str | Path,
    model_name: str,
    dataset_name: str,
    paragraphs_by_call: int,
    micro_metrics: MetricsDict,
    macro_metrics: MetricsDict,
) -> None:
    """Save experiment results to a CSV file.

    Args:
        results_file: Path to the results CSV file
        model_name: Name of the model used
        dataset_name: Name of the dataset used
        paragraphs_by_call: Number of paragraphs per API call
        micro_metrics: Dictionary containing the micro-average metrics
        macro_metrics: Dictionary containing the macro-average metrics
    """
    results_file = Path(results_file)

    # Prepare the row data
    row_data: ResultRow = {
        "model_name": model_name,
        "dataset": dataset_name,
        "paragraphs_per_call": paragraphs_by_call,
    }

    # Add metrics with prefixes to avoid column name conflicts
    row_data.update({f"micro_{k}": v for k, v in micro_metrics.items()})
    row_data.update({f"macro_{k}": v for k, v in macro_metrics.items()})

    # Get fieldnames from the first row if file exists, otherwise from row_data
    fieldnames = list(row_data.keys())
    file_exists = results_file.exists()

    if file_exists:
        with open(results_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames if reader.fieldnames else []
            # Merge existing and new fieldnames preserving order
            fieldnames = list(dict.fromkeys(existing_fieldnames + fieldnames))

    # Write to CSV file
    mode = "a" if file_exists else "w"
    with open(results_file, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
