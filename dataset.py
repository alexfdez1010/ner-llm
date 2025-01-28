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
            Returns (None, None) if the span doesn't align with token boundaries.
        """
        start_char, end_char = span

        # Check if span is out of bounds
        if start_char >= len(self.text) or end_char > len(self.text):
            return (None, None)

        # Get positions of all tokens and their surrounding whitespace
        tokens: list[tuple[int, int, int, int]] = (
            []
        )  # (ws_start, token_start, token_end, ws_end)
        current_pos = 0

        while current_pos < len(self.text):
            # Skip and track whitespace
            ws_start = current_pos
            while current_pos < len(self.text) and self.text[current_pos].isspace():
                current_pos += 1
            if current_pos >= len(self.text):
                break

            # Find token boundaries
            token_start = current_pos
            while current_pos < len(self.text) and not self.text[current_pos].isspace():
                current_pos += 1
            token_end = current_pos

            # Find end of whitespace after token
            ws_end = current_pos
            while current_pos < len(self.text) and self.text[current_pos].isspace():
                ws_end = current_pos + 1
                current_pos += 1

            tokens.append((ws_start, token_start, token_end, ws_end))

        if not tokens:  # No tokens found
            return (None, None)

        # Find tokens that contain or are adjacent to the span boundaries
        start_idx = None
        end_idx = None

        # Find start token
        for idx, (ws_start, token_start, token_end, _) in enumerate(tokens):
            if start_char < token_start:  # In leading whitespace
                start_idx = idx
                break
            elif start_char == token_start:  # At token start
                start_idx = idx
                break
            elif token_start < start_char < token_end:  # Inside token
                return (None, None)

        # Find end token
        for idx, (_, token_start, token_end, ws_end) in enumerate(tokens):
            if token_end < end_char <= ws_end:  # In trailing whitespace
                end_idx = idx
            elif end_char == token_end:  # At token end
                end_idx = idx
            elif token_start < end_char < token_end:  # Inside token
                return (None, None)

        # Handle case where start is before first token
        if start_idx is None and tokens and start_char <= tokens[0][1]:
            start_idx = 0

        # Handle case where end is after last token
        if end_idx is None and tokens and end_char >= tokens[-1][2]:
            end_idx = len(tokens) - 1

        # Return None if we didn't find both valid boundaries
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
