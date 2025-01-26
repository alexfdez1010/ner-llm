import json
from math import ceil
import os
from typing import Any, Literal

import spacy

from ai.extractor_ner import ExtractorNER
from ai.rules_generator import RulesGenerator
from dataset import Dataset
from model.category import Category
from model.entity import Entity


class Pipeline:
    """Pipeline for iteratively generating NER rules from a dataset."""

    def __init__(
        self, extractor: ExtractorNER, 
        rules_generator: RulesGenerator, dataset: Dataset,
        language: str = "en",
    ):
        """Initialize the Pipeline with required components.

        Args:
            extractor (ExtractorNER): Component for extracting entities from text
            rules_generator (RulesGenerator): Component for generating rules from examples
            dataset (Dataset): Dataset containing training instances
            language (str): Language code for the dataset
        """
        self.extractor = extractor
        self.rules_generator = rules_generator
        self.dataset = dataset
        self.language = language

    def execute(
        self,
        output_file: str,
        num_iterations: int,
        categories: list[Category],
        batch_size: int,
    ) -> None:
        """Execute the pipeline to generate and store NER rules.

        Args:
            output_file (str): Path to file where rules will be stored
            num_iterations (int): Number of iterations to run
            categories (list[Category]): List of entity categories with descriptions
            batch_size (int): Number of instances to sample per iteration
        """
        # Initialize empty rules
        current_rules: list[dict[str, Any]] = []

        # Run iterations
        for iteration in range(num_iterations):
            print(f"Iteration {iteration+1}/{num_iterations}")
            # Calculate number of instances to sample
            total_instances = len(self.dataset.training)
            number_of_batches = max(1, ceil(total_instances / batch_size))

            # Get random training instances in batches
            for i, batch in enumerate(self.dataset.get_training_instances(num_instances=batch_size)):
                print(f"Batch {i+1}/{number_of_batches}")
                # Extract entities from instances
                entities_list = [
                    instance.entities for instance in batch
                ]
                
                print("Generating new rules...")

                current_rules = self.rules_generator.generate_rules(
                    categories=categories,
                    texts=[instance.get_sentence() for instance in batch],
                    entities=entities_list,
                    old_rules=current_rules,
                    language=self.language,
                )

                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(current_rules, f, indent=2)
                
                print("Evaluating new rules...")
                precision, recall, f1 = self.evaluate(current_rules, subset="validation")
                print(f"Precision: {precision}\nRecall: {recall}\nF1: {f1}")

    def evaluate(
        self,
        rules: list[dict[str, Any]],
        subset: Literal["training", "validation", "test"] = "validation",
    ) -> tuple[float, float, float]:
        """
        Evaluates NER performance on a dataset using provided spaCy rules.

        Args:
            rules: List of rules dictionaries containing pattern and label
            subset: Subset of the dataset to evaluate on (default: "validation")

        Returns:
            Tuple containing precision, recall and F1 metrics
        """
        # Get instances based on subset
        if subset == "training":
            instances = self.dataset.training
        elif subset == "validation":
            instances = self.dataset.validation
        else:
            instances = self.dataset.test

        # Initialize spaCy matcher with rules
        nlp = spacy.blank(self.language)
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns(rules)

        # Initialize counters
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Process each instance
        for instance in instances:
            # Calculate character spans for tokens
            doc = nlp(" ".join(instance.tokens))

            # Get predicted entities
            entities_detected = [
                Entity(
                    entity=entity.text,
                    category=entity.label_,
                    span=(entity.start_char, entity.end_char),
                )
                for entity in doc.ents
            ]

            # Get gold entities (now directly from instance)
            gold_entities = instance.entities if instance.entities is not None else []

            # Convert entities to set of tuples for comparison
            pred_set = {(e.span[0], e.span[1], e.category) for e in entities_detected}
            gold_set = {(e.span[0], e.span[1], e.category) for e in gold_entities}

            # Update metrics
            true_positives += len(pred_set & gold_set)
            false_positives += len(pred_set - gold_set)
            false_negatives += len(gold_set - pred_set)

        # Calculate metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return precision, recall, f1

    def load_rules(self, rules_file: str) -> list[dict[str, Any]]:
        """Load rules from a JSON file.

        Args:
            rules_file: Path to the JSON file containing the rules

        Returns:
            List of rules as dictionaries
        """
        with open(rules_file, "r") as f:
            rules = json.load(f)
        return rules
