from dataclasses import dataclass
from random import shuffle

from model.entity import Entity


@dataclass
class Instance:
    """A NER instance of the dataset."""

    tokens: list[str]
    entities: list[Entity] | None

    def __str__(self):
        entities_str = " ".join(
            f"{entity.category}: {entity.entity}" for entity in self.entities
        )
        return " ".join(self.tokens) + "\n\n" + entities_str
    
    def get_sentence(self):
        """Returns the sentence of the instance."""
        return " ".join(self.tokens)


class Dataset:
    """A dataset of NER instances."""

    def __init__(
        self,
        training: list[Instance],
        validation: list[Instance],
        test: list[Instance]
    ):
        self.training = training
        self.validation = validation
        self.test = test

    def get_training_instances(
        self, num_instances: int | None = None
    ) -> list[Instance]:
        """
        Returns a list of training instances randomly shuffled.

        Args:
          num_instances: The number of instances to return. If None, then all instances are returned.

        Returns:
          A list of training instances with num_instances.
        """
        shuffle(self.training)

        if num_instances is None:
            return self.training

        return self.training[:num_instances]

    def get_validation_instances(self) -> list[Instance]:
        """
        Returns a list of validation instances.

        Returns:
          A list of validation instances.
        """
        return self.validation

    def get_test_instances(self) -> list[Instance]:
        """
        Returns a list of test instances.

        Returns:
          A list of test instances.
        """
        return self.test
