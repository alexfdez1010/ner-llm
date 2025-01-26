from dataclasses import dataclass


@dataclass
class Entity:
    """A named entity in a sentence."""
    category: str
    entity: str
    span: tuple[int, int]
