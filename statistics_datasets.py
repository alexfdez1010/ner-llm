"""
Script for calculating statistics of the NER datasets.
"""

from typing import Dict
import numpy as np

from main import DATASETS, get_dataset_loader


def calculate_dataset_statistics() -> Dict[str, Dict[str, float]]:
    """Calculate statistics for all datasets.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary with statistics for each dataset
    """
    statistics = {}

    for dataset_name in DATASETS:
        # Load dataset
        dataset = get_dataset_loader(dataset_name)()

        # Calculate text lengths
        text_lengths = [len(instance.text) for instance in dataset.instances]

        # Calculate number of entities per instance
        entities_count = [
            len(instance.entities) if instance.entities is not None else 0
            for instance in dataset.instances
        ]

        # Calculate statistics
        stats = {
            "num_instances": len(dataset.instances),
            "avg_text_length": np.mean(text_lengths),
            "std_text_length": np.std(text_lengths),
            "avg_entities": np.mean(entities_count),
            "std_entities": np.std(entities_count),
        }

        statistics[dataset_name] = stats

    return statistics


def print_statistics(statistics: Dict[str, Dict[str, float]]) -> None:
    """Print the statistics in a formatted way.

    Args:
        statistics: Dictionary with statistics for each dataset
    """
    print("\nDataset Statistics:")
    print("-" * 80)

    for dataset_name, stats in statistics.items():
        print(f"\n{dataset_name}:")
        print(f"  Number of instances: {stats['num_instances']}")
        print(f"  Average text length: {stats['avg_text_length']:.2f}")
        print(f"  Std dev text length: {stats['std_text_length']:.2f}")
        print(f"  Average entities per instance: {stats['avg_entities']:.2f}")
        print(f"  Std dev entities per instance: {stats['std_entities']:.2f}")


def main() -> None:
    """Main function to execute the statistics script."""
    stats = calculate_dataset_statistics()
    print_statistics(stats)


if __name__ == "__main__":
    main()
