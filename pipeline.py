"""
Pipeline for computing metrics from a dataset using a NER extractor based in LLMs.
"""

from ai.extractor_ner import ExtractorNER
from dataset import Dataset, Instance
from model.category import Category
import time


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

    def evaluate(self, sentences_per_call: int = 0) -> None:
        """
        Evaluates NER performance on a dataset using BIO annotations.
        Computes both micro-average (weighted by sentence length) and macro-average metrics.

        Args:
            sentences_per_call (int): Number of sentences to process per model call. If 0, process all text at once.
        """
        instances = self.dataset.get_instances()
        total_instances = len(instances)

        # Initialize counters for micro-average
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Initialize counters for macro-average
        all_precisions = []
        all_recalls = []
        all_f1s = []

        print(f"\nEvaluating {total_instances} instances...")
        print("-" * 80)

        start_time = time.time()

        for idx, instance in enumerate(instances, 1):
            instance_start_time = time.time()
            
            # Get gold BIO annotations and entities
            gold_bio = instance.get_bio_annotations()
            gold_entities = instance.entities

            # Get predicted BIO annotations by processing the sentence
            text = instance.get_sentence()
            predicted_entities = self.extractor.extract_entities(self.categories, text, sentences_per_call)

            # Create a temporary instance with predicted entities to get BIO annotations
            predicted_instance = Instance(
                tokens=instance.tokens, entities=predicted_entities
            )
            pred_bio = predicted_instance.get_bio_annotations()

            # Compute token-level TP, FP, FN for this instance
            instance_tp = 0
            instance_fp = 0
            instance_fn = 0

            for gold, pred in zip(gold_bio, pred_bio):
                if gold == pred and gold != "O":
                    instance_tp += 1
                elif pred != "O" and gold != pred:
                    instance_fp += 1
                elif gold != "O" and pred != gold:
                    instance_fn += 1

            # Update micro-average counters
            total_tp += instance_tp
            total_fp += instance_fp
            total_fn += instance_fn

            # Compute instance-level metrics
            instance_precision = (
                instance_tp / (instance_tp + instance_fp)
                if (instance_tp + instance_fp) > 0
                else 0.0
            )
            instance_recall = (
                instance_tp / (instance_tp + instance_fn)
                if (instance_tp + instance_fn) > 0
                else 0.0
            )
            instance_f1 = (
                2
                * (instance_precision * instance_recall)
                / (instance_precision + instance_recall)
                if (instance_precision + instance_recall) > 0
                else 0.0
            )

            all_precisions.append(instance_precision)
            all_recalls.append(instance_recall)
            all_f1s.append(instance_f1)

            # Compute current micro-average metrics
            current_micro_precision = (
                total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            )
            current_micro_recall = (
                total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            )
            current_micro_f1 = (
                2
                * (current_micro_precision * current_micro_recall)
                / (current_micro_precision + current_micro_recall)
                if (current_micro_precision + current_micro_recall) > 0
                else 0.0
            )

            # Compute current macro-average metrics
            current_macro_precision = sum(all_precisions) / len(all_precisions)
            current_macro_recall = sum(all_recalls) / len(all_recalls)
            current_macro_f1 = sum(all_f1s) / len(all_f1s)

            # Calculate time metrics
            instance_time = time.time() - instance_start_time
            total_time = time.time() - start_time

            # Print progress with metrics
            print(f"\nInstance {idx}/{total_instances} ({(idx/total_instances)*100:.1f}%)")
            print("\nPredicted entities:")
            for entity in predicted_entities:
                print(f"{entity.category}: {entity.entity} ({entity.span[0]}, {entity.span[1]})")
            print("\nGold entities:")
            for entity in gold_entities:
                print(f"{entity.category}: {entity.entity} ({entity.span[0]}, {entity.span[1]})")
            print(f"\nInstance {idx} of {total_instances}")
            print(f"Instance metrics:     Precision: {instance_precision:.2f}  Recall: {instance_recall:.2f}  F1: {instance_f1:.2f}")
            print(f"Running micro-avg:    Precision: {current_micro_precision:.2f}  Recall: {current_micro_recall:.2f}  F1: {current_micro_f1:.2f}")
            print(f"Running macro-avg:    Precision: {current_macro_precision:.2f}  Recall: {current_macro_recall:.2f}  F1: {current_macro_f1:.2f}")
            print(f"Time: {instance_time:.2f}s (instance) / {total_time:.2f}s (total)")
            print("-" * 80)

        # Final metrics are the same as the last running metrics
        micro_metrics = {
            "precision": current_micro_precision,
            "recall": current_micro_recall,
            "f1": current_micro_f1,
        }

        macro_metrics = {
            "precision": current_macro_precision,
            "recall": current_macro_recall,
            "f1": current_macro_f1,
        }

        total_time = time.time() - start_time

        print("\n\nEvaluation completed!")
        print(f"\nTotal time: {total_time:.2f}s")
        print("\nFinal Metrics:")
        print("Micro-average metrics:", micro_metrics)
        print("Macro-average metrics:", macro_metrics)