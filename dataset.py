"""
Dataset of NER instances.
"""

from dataclasses import dataclass
from random import shuffle

from model.entity import Entity


@dataclass
class Instance:
    """A NER instance of the dataset."""

    tokens: list[str]
    entities: list[Entity]

    def __str__(self):
        entities_str = " ".join(
            f"{entity.category}: {entity.entity}" for entity in self.entities
        )
        return " ".join(self.tokens) + "\n\n" + entities_str

    def get_sentence(self):
        """Returns the sentence of the instance."""
        return " ".join(self.tokens)

    def get_bio_annotations(self) -> list[str]:
        """
        Returns a list of BIO annotations for the instance. The words without entities
        are tagged as "O". The first word is tagged as "B-<category>" and the rest of
        the words are tagged as "I-<category>".

        Returns:
          A list of BIO annotations.
        """
        annotations = ["O"] * len(self.tokens)
        for entity in self.entities:
            start, end = self._get_token_indexes_from_span(entity.span)
            annotations[start] = f"B-{entity.category}"
            for i in range(start + 1, end + 1):
                annotations[i] = f"I-{entity.category}"
        return annotations

    def _get_token_indexes_from_span(self, span: tuple[int, int]) -> tuple[int, int]:
        """
        Returns the indexes of the first and last token in the span.

        Args:
            span: A tuple containing the start and end character indexes of the span.

        Returns:
            A tuple containing the indexes of the first and last token in the span.
        """
        start_char, end_char = span
        start_idx = 0
        end_idx = 0

        for idx, token in enumerate(self.tokens):
            token_start = sum(len(t) + 1 for t in self.tokens[:idx])
            token_end = token_start + len(token)

            if token_start <= start_char < token_end:
                start_idx = idx

            if token_start < end_char <= token_end:
                end_idx = idx
                break

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
