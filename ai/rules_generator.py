from typing import Any
from ai.llm import LLM
from model.entity import Entity
from model.category import Category
import json
import re


class RulesGenerator:
    def __init__(self, llm: LLM):
        """Initialize the RulesGenerator with a language model.

        Args:
            llm (LLM): The language model instance for generating rules
        """
        self.llm = llm

    def generate_rules(
        self,
        categories: list[Category],
        texts: list[str],
        entities: list[list[Entity]],
        old_rules: list[dict[str, Any]] = None,
        max_attempts: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate spaCy pattern rules for NER based on example texts and entities.

        Args:
            categories (list[Category]): List of entity categories with descriptions to generate rules for
            texts (list[str]): List of example texts
            entities (list[list[Entity]]): List of lists of entities for each text
            old_rules (list[dict[str, Any]], optional): Existing rules to build upon
            max_attempts (int, optional): Maximum number of attempts to generate valid rules. Defaults to 3.

        Returns:
            list[dict[str, Any]]: List of spaCy pattern rules in JSON format

        Raises:
            ValueError: If unable to generate valid rules after max_attempts
        """
        # Prepare examples for the prompt
        examples = []
        for text, text_entities in zip(texts, entities):
            example = {
                "text": text,
                "entities": [
                    {"text": entity.entity, "category": entity.category}
                    for entity in text_entities
                ],
            }
            examples.append(example)

        # Format the prompt with examples and categories
        system_prompt = """You are an expert in Named Entity Recognition (NER) and pattern matching.
Your task is to generate spaCy pattern rules for identifying entities in text.
The rules should be in JSON format and follow this structure:
[
  {
    "pattern": [{"LOWER": "word"}, {"IS_DIGIT": true}],
    "label": "ENTITY_TYPE"
  }
]"""

        user_prompt = f"""Generate spaCy pattern rules for the following entity categories:
{[category.name for category in categories]}

Based on these examples:
{json.dumps(examples, indent=2)}

{f'Building upon these existing rules:' + json.dumps(old_rules, indent=2) if old_rules else ''}

Return ONLY the rules in valid JSON format."""

        # Try to generate valid rules up to max_attempts times
        last_error = None
        for attempt in range(max_attempts):
            try:
                completion = self.llm.generate_completion(
                    system_prompt=system_prompt, user_prompt=user_prompt
                )

                # Extract the JSON part from the completion
                pattern = r"\[[\s\S]*\]"
                match = re.search(pattern, completion)
                if not match:
                    raise ValueError("No JSON array found in the response")

                json_str = match.group()
                new_rules = json.loads(json_str)

                # Merge with old rules if provided
                if old_rules:
                    new_rules.extend(old_rules)

                return new_rules

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                continue

        # If we get here, all attempts failed
        raise ValueError(
            f"Failed to generate valid rules after {max_attempts} attempts. Last error: {str(last_error)}"
        )
