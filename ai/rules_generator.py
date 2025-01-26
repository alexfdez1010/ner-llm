import json
import re
from typing import Any

from ai.llm import LLM
from model.category import Category
from model.entity import Entity


class RulesGenerator:
    """
    Generate spaCy pattern rules for the Rule-Based Matcher based on example texts and annotated entities.
    """
    def __init__(self, llm: LLM):
        """
        Initialize the RulesGenerator with a language model.

        Args:
            llm (LLM): The language model instance for generating rules.
        """
        self.llm = llm

    # Valid spaCy token attributes and operators
    VALID_TOKEN_ATTRS = {
        "TEXT", "LOWER", "NORM", "POS", "TAG", "DEP",
        "SHAPE", "LENGTH", "LEMMA", "MORPH", "ENT_TYPE",
        "IS_ALPHA", "IS_ASCII", "IS_DIGIT", "IS_LOWER", "IS_UPPER",
        "IS_TITLE", "IS_PUNCT", "IS_SPACE", "IS_STOP", "IS_SENT_START",
        "LIKE_NUM", "LIKE_URL", "LIKE_EMAIL", "SPACY", "REGEX", "FUZZY"
    }

    VALID_OPERATORS = {"!", "?", "+", "*"}  # Basic operators
    # Extended operators like {n}, {n,m} are validated through regex

    def is_valid_pattern_token(self, token_dict: dict) -> bool:
        """
        Validate a single token pattern dictionary.
        
        Args:
            token_dict: Dictionary containing token attributes
        
        Returns:
            bool: True if the token pattern is valid
        """
        if not isinstance(token_dict, dict):
            return False
        
        # Empty dict is valid (wildcard)
        if not token_dict:
            return True
        
        # Check all keys are valid
        for key in token_dict:
            key_upper = key.upper()
            if key_upper == "OP":
                value = token_dict[key]
                # Validate basic operators
                if value in self.VALID_OPERATORS:
                    continue
                # Validate extended operators like {n} or {n,m}
                if not isinstance(value, str):
                    return False
                if not re.match(r"^\{(\d+|\d+,\d*|\d*,\d+)\}$", value):
                    return False
            elif key_upper not in self.VALID_TOKEN_ATTRS:
                return False
            
        return True

    def validate_pattern(self, pattern: dict) -> bool:
        """
        Validate a complete spaCy pattern.
        
        Args:
            pattern: Dictionary containing label and pattern
        
        Returns:
            bool: True if the pattern is valid
        """
        if not isinstance(pattern, dict):
            return False
        
        if "label" not in pattern or "pattern" not in pattern:
            return False
        
        if not isinstance(pattern["pattern"], list):
            return False
        
        return all(self.is_valid_pattern_token(token) for token in pattern["pattern"])

    def filter_valid_rules(self, rules: list) -> list:
        """
        Filter out invalid spaCy rules.
        
        Args:
            rules: List of pattern dictionaries
        
        Returns:
            list: List containing only valid patterns
        """
        return [rule for rule in rules if self.validate_pattern(rule)]

    def generate_rules(
        self,
        categories: list[Category],
        texts: list[str],
        entities: list[list[Entity]],
        old_rules: list[dict[str, Any]] = None,
        max_attempts: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Generate spaCy pattern rules for the Rule-Based Matcher based on example texts and annotated entities.

        Parameters:
            categories (list[Category]): List of entity categories with their descriptions.
            texts (list[str]): List of example texts.
            entities (list[list[Entity]]): List of lists of entities for each text.
            old_rules (list[dict[str, Any]], optional): Existing rules (in JSON format)
                that should not be regenerated or duplicated. Defaults to None.
            max_attempts (int, optional): Maximum number of attempts to generate valid rules
                (in case of JSON parsing errors, etc.). Defaults to 3.

        Returns:
            list[dict[str, Any]]: A list of spaCy pattern rules in JSON format,
            ready to be used by spaCy.

        Raises:
            ValueError: If it fails to generate valid rules after max_attempts attempts.
        """

        # Prepare examples for the prompt (text + entities)
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

        # System prompt (high-level instructions to the model)
        # Emphasize the creation of NEW RULES ONLY, that are not covered by old_rules.
        system_prompt = """
You are an expert in creating pattern rules (patterns) for spaCy's Rule-Based Matcher.
Your task is to analyze the provided text examples and their annotated entities,
and then generate ONLY NEW pattern rules that are not already covered by existing rules.

Key guidelines for creating VALID spaCy rules:

1. Each pattern must have:
   - A "label" field with the entity type
   - A "pattern" field with a list of token specifications

2. Each token specification must use ONLY these valid attributes:
   - Text matching: TEXT, LOWER, NORM
   - Linguistic features: POS, TAG, DEP, LEMMA, MORPH
   - Token flags: IS_ALPHA, IS_DIGIT, IS_PUNCT, IS_SPACE, etc.
   - Shape and length: SHAPE, LENGTH

3. Token operators (OP) must be one of:
   - "!" (negation)
   - "?" (optional, 0 or 1)
   - "+" (1 or more)
   - "*" (0 or more)
   - "{n}" (exactly n)
   - "{n,m}" (between n and m)

4. Empty dict {} can be used as a wildcard to match any token

Example of valid patterns:
[
  {
    "label": "PERSON",
    "pattern": [
      {"LOWER": "dr"},
      {"TEXT": "."},
      {"POS": "PROPN", "OP": "+"}
    ]
  },
  {
    "label": "ORG",
    "pattern": [
      {"LOWER": "university"},
      {"LOWER": "of"},
      {"POS": "PROPN", "OP": "+"}
    ]
  }
]

Additional instructions:
- Carefully analyze the provided texts and their annotated entities
- Generate patterns that recognize variations of the same entity
- Use operators ('OP') for optional or repeated tokens when necessary
- Ensure you DO NOT duplicate patterns already present in old_rules
- Provide as much precision as possible to avoid false positives
- Return ONLY the JSON array with the new patterns (no extra explanations)
"""

        user_prompt = f"""Generate ONLY the new spaCy rules that are NOT present in the existing rules (old_rules),
taking into account the following data:

Training examples (text + entities):
{json.dumps(examples, indent=2, ensure_ascii=False)}

Categories to consider:
{json.dumps([cat.__dict__ for cat in categories], indent=2, ensure_ascii=False)}

Existing rules (old_rules) that MUST NOT be duplicated:
{json.dumps(old_rules, indent=2, ensure_ascii=False) if old_rules else "No existing rules"}

Goals:
1. Create new rules that match the entities from the examples.
2. Generalize to similar cases without causing false positives.
3. Use appropriate spaCy token attributes (LOWER, TEXT, POS, etc.).
4. Generate patterns ONLY for entities not already covered in old_rules or not redundant.
5. Return ONLY a JSON array of the new rules (no additional text).

Remember: We ONLY want NEW RULES (do not repeat existing ones).
"""

        last_error = None
        for _ in range(max_attempts):
            try:
                completion = self.llm.generate_completion(
                    system_prompt=system_prompt, user_prompt=user_prompt
                )

                # Extract the JSON array from the completion
                pattern = r"\[[\s\S]*\]"
                match = re.search(pattern, completion)
                if not match:
                    raise ValueError("No JSON array found in the response.")

                json_str = match.group()
                new_rules = json.loads(json_str)
                
                # Filter out invalid rules
                valid_rules = self.filter_valid_rules(new_rules)
                
                # If no valid rules were found, try again
                if not valid_rules:
                    raise ValueError("No valid rules found in the response")

                # If old_rules exist, combine them (append).
                # The LLM is instructed to avoid duplicates, but this final merge keeps all rules together.
                if old_rules:
                    valid_rules.extend(old_rules)

                return valid_rules

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                continue

        # If all attempts fail to produce valid JSON...
        raise ValueError(
            f"Failed to generate valid rules after {max_attempts} attempts. "
            f"Last error: {str(last_error)}"
        )
