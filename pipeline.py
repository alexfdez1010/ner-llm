from typing import Any
import json

from ai.extractor_ner import ExtractorNER
from ai.rules_generator import RulesGenerator
from dataset import Dataset
from model.category import Category
from model.entity import Entity


class Pipeline:
    """Pipeline for iteratively generating NER rules from a dataset."""

    def __init__(self, extractor: ExtractorNER, rules_generator: RulesGenerator, dataset: Dataset):
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
        sample_percentage: float = 0.1
    ) -> None:
        """Execute the pipeline to generate and store NER rules.
        
        Args:
            output_file (str): Path to file where rules will be stored
            num_iterations (int): Number of iterations to run
            sample_percentage (float, optional): Percentage of dataset to use in each iteration. Defaults to 0.1.
        """
        # Initialize empty rules
        current_rules: list[dict[str, Any]] = []
        
        # Get categories from dataset
        categories = [
            Category(name=category, description="")
            for category in self.dataset.index_to_category.values()
            if category is not None  # Skip None categories
        ]
        
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
                old_rules=current_rules
            )
            
            # Store rules after each iteration
            with open(output_file, 'w') as f:
                json.dump(current_rules, f, indent=2)
