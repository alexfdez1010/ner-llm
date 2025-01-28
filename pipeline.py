"""
Pipeline for computing metrics from a dataset using a NER extractor based in LLMs.
"""

import time
from typing import NamedTuple

from ai.extractor_ner import ExtractorNER
from dataset import Dataset, Instance
from model.category import Category
from model.entity import Entity


class TokenMetrics(NamedTuple):
    """Metrics for token-level evaluation."""

    tp: float
    fp: float
    fn: float


class Pipeline:
    """Pipeline for computing metrics from a dataset using a NER extractor based in LLMs."""

    def __init__(
        self,
        extractor: ExtractorNER,
        dataset: Dataset,
        categories: list[Category],
    ):
        """Initialize the Pipeline with required components.

        Args:
            extractor (ExtractorNER): Component for extracting entities from text
            dataset (Dataset): Dataset containing training instances
            categories (list[Category]): List of categories for the dataset
        """
        self.extractor = extractor
        self.dataset = dataset
        self.categories = categories

    def compute_instance_metrics(
        self, gold_bio: list[str], pred_bio: list[str]
    ) -> TokenMetrics:
        """Compute token-level metrics for a single instance.

        Args:
            gold_bio: Gold standard BIO annotations
            pred_bio: Predicted BIO annotations

        Returns:
            TokenMetrics containing true positives, false positives, and false negatives
        """
        tp = fp = fn = 0.0

        for gold, pred in zip(gold_bio, pred_bio):
            # Both are O, nothing to count
            if gold == "O" and pred == "O":
                continue

            # Both are entity tags
            if gold != "O" and pred != "O":
                gold_tag = gold.split("-", 1)
                pred_tag = pred.split("-", 1)

                # Same entity type
                if (
                    len(gold_tag) == 2
                    and len(pred_tag) == 2
                    and gold_tag[1] == pred_tag[1]
                ):
                    # Both B or both I
                    if gold_tag[0] == pred_tag[0]:
                        tp += 1
                    # One is B and one is I - partial match
                    else:
                        tp += 0.5
                        fp += 0.5
                # Different entity types
                else:
                    fp += 1
                    fn += 1
            # One is O and one is an entity
            else:
                if gold != "O":
                    fn += 1
                if pred != "O":
                    fp += 1

        return TokenMetrics(tp=tp, fp=fp, fn=fn)

    def _calculate_f1_metrics(
        self, tp: float, fp: float, fn: float
    ) -> dict[str, float]:
        """Calculate precision, recall, and F1 score.

        Args:
            tp: True positives
            fp: False positives
            fn: False negatives

        Returns:
            Dictionary containing precision, recall, and F1 score
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    def _print_evaluation_progress(
        self,
        instance_idx: int,
        total_instances: int,
        predicted_entities: list[Entity],
        gold_entities: list[Entity],
        instance_metrics: dict[str, float],
        micro_metrics: dict[str, float],
        macro_metrics: dict[str, float],
        instance_time: float,
        total_time: float,
    ) -> None:
        """Print evaluation progress and metrics.

        Args:
            instance_idx: Current instance index
            total_instances: Total number of instances
            predicted_entities: List of predicted entities
            gold_entities: List of gold entities
            instance_metrics: Metrics for current instance
            micro_metrics: Running micro-average metrics
            macro_metrics: Running macro-average metrics
            instance_time: Time taken for current instance
            total_time: Total time taken so far
        """
        print(
            f"\nInstance {instance_idx}/{total_instances} ({(instance_idx/total_instances)*100:.1f}%)"
        )
        print("\nPredicted entities:")
        for entity in predicted_entities:
            print(
                f"{entity.category}: {entity.entity} ({entity.span[0]}, {entity.span[1]})"
            )
        print("\nGold entities:")
        for entity in gold_entities:
            print(
                f"{entity.category}: {entity.entity} ({entity.span[0]}, {entity.span[1]})"
            )
        print(f"\nInstance {instance_idx} of {total_instances}")
        print(
            f"Instance metrics:     Precision: {instance_metrics['precision']:.2f}  "
            f"Recall: {instance_metrics['recall']:.2f}  F1: {instance_metrics['f1']:.2f}"
        )
        print(
            f"Running micro-avg:    Precision: {micro_metrics['precision']:.2f}  "
            f"Recall: {micro_metrics['recall']:.2f}  F1: {micro_metrics['f1']:.2f}"
        )
        print(
            f"Running macro-avg:    Precision: {macro_metrics['precision']:.2f}  "
            f"Recall: {macro_metrics['recall']:.2f}  F1: {macro_metrics['f1']:.2f}"
        )
        print(f"Time: {instance_time:.2f}s (instance) / {total_time:.2f}s (total)")
        print("-" * 80)

    def evaluate(
        self, sentences_per_call: int = 0
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Evaluate NER performance on a dataset using BIO annotations.

        Args:
            sentences_per_call: Number of sentences to process per model call. If 0, process all text at once.

        Returns:
            Tuple containing micro-average and macro-average metrics
        """
        instances = self.dataset.get_instances()
        total_instances = len(instances)

        # Initialize counters for micro-average
        total_metrics = TokenMetrics(tp=0.0, fp=0.0, fn=0.0)

        # Initialize lists for macro-average
        all_metrics = []

        print(f"\nEvaluating {total_instances} instances...")
        print("-" * 80)

        start_time = time.time()

        for idx, instance in enumerate(instances, 1):
            if instance.entities is None:
                print(f"Skipping instance {idx} because there are no entities")
                continue

            instance_start_time = time.time()

            # Get gold BIO annotations and entities
            gold_bio = instance.get_bio_annotations()
            gold_entities = instance.entities

            # Get predicted BIO annotations by processing the sentence
            text = instance.get_sentence()
            predicted_entities = self.extractor.extract_entities(
                self.categories, text, sentences_per_call
            )

            # Create a temporary instance with predicted entities to get BIO annotations
            predicted_instance = Instance(text=text, entities=predicted_entities)
            pred_bio = predicted_instance.get_bio_annotations()

            # Compute token-level metrics for this instance
            instance_metrics = self.compute_instance_metrics(gold_bio, pred_bio)

            # Update micro-average counters
            total_metrics = TokenMetrics(
                tp=total_metrics.tp + instance_metrics.tp,
                fp=total_metrics.fp + instance_metrics.fp,
                fn=total_metrics.fn + instance_metrics.fn,
            )

            # Calculate instance-level F1 metrics
            instance_f1_metrics = self._calculate_f1_metrics(
                instance_metrics.tp, instance_metrics.fp, instance_metrics.fn
            )
            all_metrics.append(instance_f1_metrics)

            # Calculate current micro and macro metrics
            current_micro_metrics = self._calculate_f1_metrics(
                total_metrics.tp, total_metrics.fp, total_metrics.fn
            )

            current_macro_metrics = {
                metric: sum(m[metric] for m in all_metrics) / len(all_metrics)
                for metric in ["precision", "recall", "f1"]
            }

            # Print progress
            instance_time = time.time() - instance_start_time
            total_time = time.time() - start_time

            self._print_evaluation_progress(
                idx,
                total_instances,
                predicted_entities,
                gold_entities,
                instance_f1_metrics,
                current_micro_metrics,
                current_macro_metrics,
                instance_time,
                total_time,
            )

        total_time = time.time() - start_time
        print("\n\nEvaluation completed!")
        print(f"\nTotal time: {total_time:.2f}s")
        print("\nFinal Metrics:")

        # Final metrics are the same as the last running metrics
        micro_metrics = self._calculate_f1_metrics(
            total_metrics.tp, total_metrics.fp, total_metrics.fn
        )

        macro_metrics = {
            metric: sum(m[metric] for m in all_metrics) / len(all_metrics)
            for metric in ["precision", "recall", "f1"]
        }

        print(f"Micro-average metrics: {micro_metrics}")
        print(f"Macro-average metrics: {macro_metrics}")

        return micro_metrics, macro_metrics
