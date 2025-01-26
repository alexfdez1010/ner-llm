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


def test_dataset_creation():
    training = [Instance(tokens=["train1"], entities=None)]
    validation = [Instance(tokens=["val1"], entities=None)]
    test = [Instance(tokens=["test1"], entities=None)]

    dataset = Dataset(
        training=training,
        validation=validation,
        test=test,
    )
    assert dataset.training == training
    assert dataset.validation == validation
    assert dataset.test == test


def test_get_training_instances():
    training = [
        Instance(tokens=["train1"], entities=None),
        Instance(tokens=["train2"], entities=None),
        Instance(tokens=["train3"], entities=None),
    ]
    dataset = Dataset(
        training=training,
        validation=[],
        test=[],
    )

    # Test getting all instances
    all_instances = dataset.get_training_instances()
    assert len(all_instances) == 3
    assert all(isinstance(inst, Instance) for inst in all_instances)

    # Test getting limited number of instances
    limited_instances = dataset.get_training_instances(num_instances=2)
    assert len(limited_instances) == 2
    assert all(isinstance(inst, Instance) for inst in limited_instances)


def test_get_validation_instances():
    validation = [
        Instance(tokens=["val1"], entities=None),
        Instance(tokens=["val2"], entities=None),
    ]
    dataset = Dataset(
        training=[],
        validation=validation,
        test=[],
    )

    instances = dataset.get_validation_instances()
    assert len(instances) == 2
    assert all(isinstance(inst, Instance) for inst in instances)


def test_get_test_instances():
    test = [
        Instance(tokens=["test1"], entities=None),
        Instance(tokens=["test2"], entities=None),
    ]
    dataset = Dataset(
        training=[],
        validation=[],
        test=test,
    )

    instances = dataset.get_test_instances()
    assert len(instances) == 2
    assert all(isinstance(inst, Instance) for inst in instances)
