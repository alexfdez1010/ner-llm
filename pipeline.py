from typing import Any, Literal
import json
import spacy

from ai.extractor_ner import ExtractorNER
from ai.rules_generator import RulesGenerator
from dataset import Dataset
from model.category import Category
from model.entity import Entity

import os


class Pipeline:
    """Pipeline for iteratively generating NER rules from a dataset."""

    def __init__(
        self, extractor: ExtractorNER, rules_generator: RulesGenerator, dataset: Dataset
    ):
        """Initialize the Pipeline with required components.

        Args:
            extractor (ExtractorNER): Component for extracting entities from text
            rules_generator (RulesGenerator): Component for generating rules from examples
            dataset (Dataset): Dataset containing training instances
        """
        self.extractor = extractor
        self.rules_generator = rules_generator
        self.dataset = dataset

    def execute(
        self,
        output_file: str,
        num_iterations: int,
        categories: list[Category],
        sample_percentage: float = 0.1,
    ) -> None:
        """Execute the pipeline to generate and store NER rules.

        Args:
            output_file (str): Path to file where rules will be stored
            num_iterations (int): Number of iterations to run
            categories (list[Category]): List of entity categories with descriptions
            sample_percentage (float, optional): Percentage of dataset to use in each iteration. Defaults to 0.1.
        """
        # Initialize empty rules
        current_rules: list[dict[str, Any]] = []

        # Run iterations
        for iteration in range(num_iterations):
            print(f"Iteration {iteration+1}/{num_iterations}")
            # Calculate number of instances to sample
            total_instances = len(self.dataset.training)
            num_instances = max(1, int(total_instances * sample_percentage))

            # Get random training instances
            instances = self.dataset.get_training_instances(num_instances=num_instances)

            # Extract texts and entities from instances
            texts = []
            entities_list = []

            for instance in instances:
                # Get text from tokens
                text = " ".join(instance.tokens)
                texts.append(text)

                # Get entities for this instance
                entities = instance.get_entities(self.dataset.index_to_category)
                entities_list.append(entities)

            # Generate new rules using current rules as base
            current_rules = self.rules_generator.generate_rules(
                categories=categories,
                texts=texts,
                entities=entities_list,
                old_rules=current_rules,
            )

            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # Store rules after each iteration
            with open(output_file, "w") as f:
                json.dump(current_rules, f, indent=2)

    def evaluate(
        self, 
        rules: list[dict[str, Any]], 
        subset: Literal["training", "validation", "test"] = "validation"
    ) -> tuple[float, float, float]:
        """
        Evaluates NER performance on a dataset using provided spaCy rules.
        
        Args:
            rules: List of rules dictionaries containing pattern and label
            subset: Subset of the dataset to evaluate on (default: "validation")
        
        Returns:
            Dictionary with precision, recall and F1 metrics
        """
        # Get instances based on subset
        if subset == "training":
            instances = self.dataset.training
        elif subset == "validation":
            instances = self.dataset.validation
        else:
            instances = self.dataset.test

        # Initialize spaCy matcher with rules
        nlp = spacy.blank("en")
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

            entities_detected = [
                Entity(
                    entity=entity.text,
                    category=entity.label_,
                    span=(entity.start_char, entity.end_char),
                )
                for entity in doc.ents
            ]

            # Get gold entities
            gold_entities = instance.get_entities(self.dataset.index_to_category)

            # Convert entities to set of tuples for comparison
            pred_set = {(e.span[0], e.span[1], e.category) for e in entities_detected}
            gold_set = {(e.span[0], e.span[1], e.category) for e in gold_entities}

            # Update metrics
            true_positives += len(pred_set & gold_set)
            false_positives += len(pred_set - gold_set)
            false_negatives += len(gold_set - pred_set)

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def load_rules(self, rules_file: str) -> list[dict[str, Any]]:
        with open(rules_file, "r") as f:
            rules = json.load(f)
        return rules
