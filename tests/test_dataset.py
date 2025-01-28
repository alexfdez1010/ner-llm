import pytest
from dataset import Dataset, Instance
from model.entity import Entity


def test_instance_creation():
    # Test instance creation with entities
    entities = [
        Entity(category="PERSON", entity="John", span=(0, 4)),
        Entity(category="LOCATION", entity="New York", span=(14, 22))
    ]
    instance = Instance(tokens=["hello", "world"], entities=entities)
    assert instance.tokens == ["hello", "world"]
    assert instance.entities == entities

    # Test instance creation without entities
    instance = Instance(tokens=["hello", "world"], entities=None)
    assert instance.tokens == ["hello", "world"]
    assert instance.entities is None


def test_instance_to_string():
    entities = [
        Entity(category="PERSON", entity="John", span=(0, 4)),
        Entity(category="LOCATION", entity="New York", span=(14, 22))
    ]
    instance = Instance(tokens=["John", "lives", "in", "New", "York"], entities=entities)
    assert str(instance) == "John lives in New York\n\nPERSON: John LOCATION: New York"


def test_instance_get_sentence():
    instance = Instance(tokens=["hello", "world"], entities=None)
    assert instance.get_sentence() == "hello world"


def test_get_token_indexes_from_span():
    # Test case 1: Single token span
    instance = Instance(tokens=["John", "lives", "in", "New", "York"], entities=[])
    assert instance._get_token_indexes_from_span((0, 4)) == (0, 0)  # "John"
    
    # Test case 2: Multi-token span
    assert instance._get_token_indexes_from_span((14, 22)) == (3, 4)  # "New York"
    
    # Test case 3: Span in the middle
    assert instance._get_token_indexes_from_span((5, 10)) == (1, 1)  # "lives"
    
    # Test case 4: Span with spaces
    instance = Instance(tokens=["The", "United", "States", "of", "America"], entities=[])
    assert instance._get_token_indexes_from_span((4, 17)) == (1, 2)  # "United States"


def test_get_bio_annotations():
    # Test case 1: Single entity
    instance = Instance(
        tokens=["John", "lives", "in", "New", "York"],
        entities=[Entity(category="PERSON", entity="John", span=(0, 4))]
    )
    expected = ["B-PERSON", "O", "O", "O", "O"]
    assert instance.get_bio_annotations() == expected
    
    # Test case 2: Multiple entities
    instance = Instance(
        tokens=["John", "lives", "in", "New", "York"],
        entities=[
            Entity(category="PERSON", entity="John", span=(0, 4)),
            Entity(category="LOCATION", entity="New York", span=(14, 22))
        ]
    )
    expected = ["B-PERSON", "O", "O", "B-LOCATION", "I-LOCATION"]
    assert instance.get_bio_annotations() == expected
    
    # Test case 3: No entities
    instance = Instance(tokens=["Hello", "world"], entities=[])
    expected = ["O", "O"]
    assert instance.get_bio_annotations() == expected
    
    # Test case 4: Adjacent entities
    instance = Instance(
        tokens=["Visit", "New", "York", "City"],
        entities=[
            Entity(category="LOCATION", entity="New York", span=(6, 14)),
            Entity(category="LOCATION", entity="City", span=(15, 19))
        ]
    )
    expected = ["O", "B-LOCATION", "I-LOCATION", "B-LOCATION"]
    assert instance.get_bio_annotations() == expected