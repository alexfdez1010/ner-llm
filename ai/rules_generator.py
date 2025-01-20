from typing import Any
from ai.llm import LLM
from model.entity import Entity
from model.category import Category
import json

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
        old_rules: list[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Generate spaCy pattern rules for NER based on example texts and entities.
        
        Args:
            categories (list[Category]): List of entity categories with descriptions to generate rules for
            texts (list[str]): List of example texts
            entities (list[list[Entity]]): List of lists of entities for each text
            old_rules (list[dict[str, Any]], optional): Existing rules to build upon
            
        Returns:
            list[dict[str, Any]]: List of spaCy pattern rules in JSON format
        """
        # Prepare examples for the prompt
        examples = []
        for text, text_entities in zip(texts, entities):
            example = {
                "text": text,
                "entities": [
                    {
                        "entity": e.entity,
                        "category": e.category,
                        "span": (e.span[0], e.span[1])
                    } for e in text_entities
                ]
            }
            examples.append(example)

        # Create the system prompt with clear instructions
        system_prompt = """
You are an expert in creating pattern matching rules for spaCy's Rule-Based Matcher. Your task is to analyze text examples and their annotated entities to create precise matching patterns.

Key points about spaCy patterns:
1. Each pattern is a list of token specifications
2. Token specs can include:
   - 'LOWER': lowercase token text
   - 'TEXT': exact token text
   - 'REGEX': regex pattern
   - 'POS': part-of-speech tag
   - 'TAG': fine-grained POS tag
   - 'DEP': syntactic dependency
   - 'OP': operator ('?', '+', '*', '!')
   
Example pattern:
{
    "label": "PERSON",
    "pattern": [
        {"LOWER": "dr"},
        {"TEXT": "."},
        {"POS": "PROPN", "OP": "+"}
    ]
}

Your task:
1. Analyze the provided examples
2. Create specific patterns that would match similar entities
3. Consider context and variations
4. Include both exact matches and flexible patterns
5. Use appropriate operators for optional elements
6. Consider existing rules to avoid conflicts

Output only valid JSON patterns that spaCy can directly use."""

        # Create the user prompt with examples and existing rules
        user_prompt = f"""Generate spaCy pattern rules for the following examples:

Examples:
{json.dumps(examples, indent=2)}

Categories: {categories}

Existing rules (to consider for improvements/conflicts):
{json.dumps(old_rules, indent=2) if old_rules else "No existing rules"}

Generate a comprehensive set of pattern rules that:
1. Match the given examples
2. Can generalize to similar cases
3. Are precise enough to avoid false positives
4. Take into account the context
5. Use appropriate token attributes (LOWER, TEXT, POS, etc.)
6. Include patterns for variations of the same entity type

Return only the JSON array of pattern rules."""

        # Generate rules using the LLM
        response = self.llm.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        # Clean up response from potential markdown code blocks
        response = response.strip()
        if response.startswith("```") and response.endswith("```"):
            response = response.split("\n", 1)[1].rsplit("\n", 1)[0]
        if response.startswith("json\n"):
            response = response[5:]

        try:
            # Parse and validate the generated rules
            new_rules = json.loads(response)
            if not isinstance(new_rules, list):
                raise ValueError("Generated rules must be a list")
            
            # Merge with old rules if they exist
            if old_rules:
                # Create a set of rule patterns to avoid duplicates
                existing_patterns = {
                    json.dumps(rule.get("pattern", []))
                    for rule in old_rules
                }
                
                # Only add new rules that don't exist yet
                final_rules = old_rules.copy()
                for rule in new_rules:
                    if json.dumps(rule.get("pattern", [])) not in existing_patterns:
                        final_rules.append(rule)
                return final_rules
            
            return new_rules
            
        except json.JSONDecodeError:
            raise ValueError("Generated rules are not in valid JSON format")