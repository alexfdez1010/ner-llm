import json
import re
import spacy
from typing import Any, Optional

from ai.llm import LLM
from model.category import Category
from model.entity import Entity


class RulesGenerator:
    """
    Generate spaCy pattern rules for the Rule-Based Matcher based on example texts and annotated entities.

    This version includes:
    1. A requirement that each new rule must have a semantically valid "id" field.
    2. Logic to overwrite any existing rule that has the same "id" as a newly generated rule.
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
        "TEXT",
        "LOWER",
        "NORM",
        "LENGTH",
        "LEMMA",
        "MORPH",
        "IS_ALPHA",
        "IS_ASCII",
        "IS_DIGIT",
        "IS_LOWER",
        "IS_UPPER",
        "IS_TITLE",
        "IS_PUNCT",
        "IS_SPACE",
        "IS_STOP",
        "IS_SENT_START",
        "LIKE_NUM",
        "LIKE_URL",
        "LIKE_EMAIL",
        "SPACY",
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
                # The key is not a valid spaCy token attribute
                return False

        return True

    def validate_pattern(self, pattern: dict) -> bool:
        """
        Validate a complete spaCy pattern.

        Args:
            pattern: Dictionary containing label, pattern, and id

        Returns:
            bool: True if the pattern is valid
        """
        if not isinstance(pattern, dict):
            return False

        # Must have label and pattern
        if "label" not in pattern or "pattern" not in pattern:
            return False

        # Must have a list of tokens in "pattern"
        if not isinstance(pattern["pattern"], list):
            return False

        # Must have an "id" (for overwriting logic and reference)
        if "id" not in pattern or not isinstance(pattern["id"], str):
            return False

        # Validate each token in the pattern
        return all(self.is_valid_pattern_token(token) for token in pattern["pattern"])

    def filter_valid_rules(self, rules: list[dict], texts: list[str], language: str, categories: list[Category]) -> list[dict]:
        """
        Filter out invalid spaCy rules.

        Args:
            rules: List of pattern dictionaries
            texts: List of texts used to validate rules

        Returns:
            list: List containing only valid patterns
        """

        # Filter out invalid rules
        valid_rules = [rule for rule in rules if self.validate_pattern(rule)]

        # Filter rules that doesn't match at least one text
        valid_rules = self.filter_matching_rules(valid_rules, texts, language, categories)

        return valid_rules

    def merge_rules_with_overwrite(
        self, old_rules: list[dict[str, Any]], new_rules: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Merge old and new rules, overwriting any old rules that share an "id" with a new rule.

        Args:
            old_rules (list): Existing rules
            new_rules (list): Newly generated valid rules

        Returns:
            list: The merged list of rules, where new rules overwrite old ones by the same "id".
        """
        # Create a dictionary keyed by rule "id" for quick lookup and overwrite
        merged_dict = {}

        # Add old rules first
        for old_rule in old_rules:
            rule_id = old_rule.get("id")
            # If there's no id, generate a fallback key to preserve it
            if not rule_id:
                rule_id = f"NO_ID_{id(old_rule)}"
            merged_dict[rule_id] = old_rule

        # Overwrite or add new rules
        for new_rule in new_rules:
            rule_id = new_rule.get("id")
            if not rule_id:
                rule_id = f"NO_ID_{id(new_rule)}"
            merged_dict[rule_id] = new_rule

        # Return the merged list of rules
        return list(merged_dict.values())

    def get_entities_as_text(self, entities: list[list[Entity]]) -> list[str]:
        """
        Convert the list of entities to a list of strings sorted by type and text.

        Args:
            entities (list[list[Entity]]): A list of lists of Entity objects.

        Returns:
            list[str]: A list of strings representing the entities sorted by type and text.
        """
        entities_as_text = [
            f"{entity.category}: {entity.entity}"
            for entity_list in entities
            if entity_list
            for entity in entity_list
        ]

        entities_as_text.sort(key=lambda x: x.lower())

        return entities_as_text

    def filter_matching_rules(
        self,
        rules: list[dict[str, Any]],
        texts: list[str],
        language: str,
        categories: list[Category],
    ) -> list[dict[str, Any]]:
        """
        Filter rules that match at least one of the provided texts and validate against given entities.

        Args:
            rules (list[dict[str, Any]]): List of rules to filter.
            texts (list[str]): List of texts to match against.
            language (str): Language model to use for spaCy.
            categories (list[Category]): List of categories to validate labels against.

        Returns:
            list[dict[str, Any]]: List of rules that matched at least one text and passed validation.
        """
        matching_rules = set()
        text = "\n".join(texts)

        nlp = spacy.blank(language)
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns(rules)

        doc = nlp(text)

        categories_names = [category.name for category in categories]
        entities_ids = [entity.ent_id for entity in doc.ents]

        print(doc.ents)
        print(categories_names)
        print(entities_ids)

        for rule in rules:
            if rule["label"] not in categories_names:
                continue

            if rule["id"] in entities_ids:
                matching_rules.add(json.dumps(rule))

        return [json.loads(rule) for rule in matching_rules]

    def generate_rules(
        self,
        categories: list[Category],
        texts: list[str],
        entities: list[list[Entity]],
        language: str,
        old_rules: Optional[list[dict[str, Any]]] = None,
        max_attempts: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Generate spaCy pattern rules for the Rule-Based Matcher based on example texts and annotated entities.

        Parameters:
            categories (list[Category]): List of entity categories with their descriptions.
            texts (list[str]): List of example texts.
            entities (list[list[Entity]]): List of lists of entities for each text.
            language (str): Language model to use for spaCy.
            old_rules (list[dict[str, Any]], optional): Existing rules (in JSON format).
                New rules with the same "id" will overwrite these.
                Defaults to None.
            max_attempts (int, optional): Maximum number of attempts to generate valid rules
                (in case of JSON parsing errors, etc.). Defaults to 3.

        Returns:
            list[dict[str, Any]]: A list of spaCy pattern rules in JSON format,
            ready to be used by spaCy.

        Raises:
            ValueError: If it fails to generate valid rules after max_attempts attempts.
        """

        # Prepare examples for the prompt (text + entities)
        entities_as_text = self.get_entities_as_text(entities)
        matching_rules = self.filter_matching_rules(
            old_rules, texts, language, categories)

        print(
            f"Found {len(matching_rules)} matching rules out of {len(old_rules)} old rules"
        )

        # Extend the system prompt with the requirement for "id" and overwriting logic
        system_prompt = """
You are an expert in creating pattern rules (patterns) for spaCy's Rule-Based Matcher.
Your task is to analyze the provided text examples and their annotated entities,
and then generate ONLY NEW pattern rules that are not already covered by existing rules.

Key guidelines for creating VALID spaCy rules:

1. Each pattern must have:
   - A "label" field with the entity type
   - A "pattern" field with a list of token specifications, remember that each token is a word separated by spaces
   - An "id" field (a short, descriptive identifier for the rule)

2. Each token specification must use ONLY these valid attributes:
   - Text matching: TEXT, LOWER, NORM
   - Token flags: IS_ALPHA, IS_DIGIT, IS_PUNCT, IS_SPACE, IS_STOP, IS_TITLE, IS_UPPER, etc.
   - Like flags: LIKE_NUM, LIKE_URL, LIKE_EMAIL
   - Length: LENGTH

3. Token operators (OP) can be one of these:
   - "!" (negation)
   - "?" (optional, 0 or 1)
   - "+" (1 or more)
   - "*" (0 or more)
   - Extended quantifiers like "{n}" or "{n,m}" are also acceptable (e.g., "{1,2}")

4. If you generate a new rule that has the same "id" as an existing rule, the new rule will overwrite the old one.

Example of valid patterns:
[
  {
    "label": "PERSON",
    "pattern": [
      {"LOWER": "dr"},
      {"TEXT": "."},
      {"POS": "PROPN"}
    ],
    "id": "doctor"
  },
  {
    "label": "ORG",
    "pattern": [
      {"LOWER": "university"},
      {"LOWER": "of"},
      {"POS": "PROPN"}
    ],
    "id": "university"
  },
  {
    "label": "DATE",
    "pattern": [
      {"LIKE_NUM": true},
      {"LOWER": "jan", "OP": "?"},
      {"LOWER": "feb", "OP": "?"},
      {"LOWER": "mar", "OP": "?"}
    ],
    "id": "flexible_date_month"
  },
  {
    "label": "PRODUCT",
    "pattern": [
      {"IS_UPPER": true, "OP": "+"},
      {"IS_DIGIT": true, "OP": "+"}
    ],
    "id": "code_with_numbers"
  }
]

Additional instructions:
- Carefully analyze the provided entities 
- Generate patterns that recognize variations of the same entity.
- Use operators ('OP') for optional or repeated tokens when necessary.
- Ensure you DO NOT duplicate patterns already present in old_rules.
- Provide as much precision as possible to avoid false positives.
- Return ONLY the JSON array with the new patterns (no extra explanations).
- Include an "id" in each rule. If a new rule has the same "id" as an old rule, it overwrites the old rule.
- Never use spaces or tabs in the patterns as the token are separated by spaces and will never produce a match.
"""

        user_prompt = f"""Generate ONLY the new spaCy rules that are NOT present in the existing rules (old_rules),
taking into account the following data:

Existing rules (old_rules) that MUST NOT be duplicated (unless by overwriting with same 'id'):
{json.dumps(matching_rules, indent=2, ensure_ascii=False) if old_rules else "No existing rules"}

Categories to consider (you can only use these):
{"\n".join([f"{cat.name}: {cat.description}" for cat in categories])}

Entitites to recognize (entities sorted):
{"\n".join(entities_as_text)}

Goals:
1. Create new rules that match the entities.
2. Generalize to similar cases without causing false positives.
3. Use appropriate spaCy token attributes (LOWER, TEXT, LEMMA, etc.).
4. Generate patterns ONLY for entities not already covered in old_rules or not redundant.
5. Return ONLY a JSON array of the new rules (no additional text).
6. Each new rule MUST have an 'id'. If an id duplicates one from old_rules, it overwrites it.
7. Never use spaces or tabs in the patterns as tokens are separated by spaces and will never produce a match.

Remember: We ONLY want NEW RULES or OVERWRITTEN RULES (do not repeat existing ones unchanged).
"""

        last_error = None
        for attempt in range(max_attempts):
            try:
                print(f"Attempt {attempt + 1} to generate rules")

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

                # Filter out invalid and non-matching rules
                valid_rules = self.filter_valid_rules(new_rules, texts, language, categories)
                print(f"Removed {len(new_rules) - len(valid_rules)} invalid rules")

                # If no valid rules were found, try again
                if not valid_rules:
                    raise ValueError("No valid rules found in the response")

                if old_rules:
                    # Merge old rules with new rules, overwriting if there's a matching 'id'
                    merged_rules = self.merge_rules_with_overwrite(
                        old_rules, valid_rules
                    )
                    return merged_rules

                # If no old rules, just return the new valid rules
                return valid_rules

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                continue

        # If all attempts fail to produce valid JSON...
        raise ValueError(
            f"Failed to generate valid rules after {max_attempts} attempts. "
            f"Last error: {str(last_error)}"
        )
