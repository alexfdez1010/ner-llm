from dataclasses import dataclass


@dataclass
class Entity:
    category: str
    entity: str
    span: tuple[int, int]
