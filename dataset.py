"""
Dataset of NER instances.
"""

from dataclasses import dataclass
from random import shuffle

from model.entity import Entity


@dataclass
class Instance:
    """A NER instance of the dataset."""

    text: str
    entities: list[Entity]

    def __str__(self):
        entities_str = " ".join(
            f"{entity.category}: {entity.entity}" for entity in self.entities
        )
        return f"{self.text}\n\n{entities_str}"

    def get_sentence(self) -> str:
        """Returns the text of the instance."""
        return self.text

    def get_bio_annotations(self) -> list[str]:
        """
        Returns a list of BIO annotations for the instance. The words without entities
        are tagged as "O". The first word is tagged as "B-<category>" and the rest of
        the words are tagged as "I-<category>".

        Returns:
          A list of BIO annotations.
        """
        # Split text into tokens
        tokens = self.text.split()
        annotations = ["O"] * len(tokens)

        for entity in self.entities:
            start, end = self._get_token_indexes_from_span(entity.span)
            if (
                start is not None and end is not None
            ):  # Only process if valid token indexes found
                annotations[start] = f"B-{entity.category}"
                for i in range(start + 1, end + 1):
                    annotations[i] = f"I-{entity.category}"
        return annotations

    def _get_token_indexes_from_span(
        self, span: tuple[int, int]
    ) -> tuple[int, int] | tuple[None, None]:
        """
        Returns the indexes of the first and last token in the span.

        Args:
            span: A tuple containing the start and end character indexes of the span.

        Returns:
            A tuple containing the indexes of the first and last token in the span.
            Returns (None, None) if the span is invalid.
        """
        start_char, end_char = span

        # Check if span is out of bounds
        if start_char >= len(self.text) or end_char > len(self.text):
            return (None, None)

        # Split text into tokens and get their spans
        tokens = self.text.split()
        current_pos = 0
        token_spans: list[tuple[int, int, int]] = []  # (start, end, index)

        # Calculate character spans for each token
        for i, token in enumerate(tokens):
            # Skip whitespace
            while current_pos < len(self.text) and self.text[current_pos].isspace():
                current_pos += 1
            
            token_spans.append((current_pos, current_pos + len(token), i))
            current_pos += len(token)

        # Find tokens that overlap with the span
        start_idx = None
        end_idx = None

        # Find start token (first token that overlaps with start_char)
        for token_start, token_end, idx in token_spans:
            if token_start <= start_char <= token_end:
                start_idx = idx
                break
            if start_char < token_start:
                start_idx = idx
                break

        # Find end token (last token that overlaps with end_char)
        for token_start, token_end, idx in token_spans:
            if token_start <= end_char <= token_end:
                end_idx = idx
            elif token_start > end_char:
                end_idx = idx - 1
                break

        # Handle edge cases
        if start_idx is None and token_spans:
            start_idx = 0
        if end_idx is None and token_spans:
            end_idx = len(token_spans) - 1

        # Return None if we didn't find valid boundaries
        if start_idx is None or end_idx is None or start_idx > end_idx:
            return (None, None)

        return (start_idx, end_idx)


class Dataset:
    """A dataset of NER instances."""

    def __init__(
        self,
        instances: list[Instance],
    ):
        self.instances = instances

    def get_instances(self) -> list[Instance]:
        """
        Returns a list of instances.

        Returns:
          A list of instances.
        """
        shuffle(self.instances)
        return self.instances
