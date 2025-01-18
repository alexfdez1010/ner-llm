import pytest
from dataset import Dataset, Instance

def test_instance_creation():
    # Test instance creation with labels
    instance = Instance(tokens=["hello", "world"], labels=[0, 1])
    assert instance.tokens == ["hello", "world"]
    assert instance.labels == [0, 1]
    
    # Test instance creation without labels
    instance = Instance(tokens=["hello", "world"], labels=None)
    assert instance.tokens == ["hello", "world"]
    assert instance.labels is None

def test_instance_to_string():
    instance = Instance(tokens=["hello", "world"], labels=[0, 1])
    assert str(instance) == "hello world"

def test_instance_get_entities():
    # Test with all valid categories
    instance = Instance(tokens=["John", "lives", "in", "New", "York"], labels=[1, 0, 0, 2, 2])
    index_to_category = {0: None, 1: "PERSON", 2: "LOCATION"}
    entities = instance.get_entities(index_to_category)
    assert entities == ["PERSON", "LOCATION", "LOCATION"]
    
    # Test with some invalid categories
    instance = Instance(tokens=["The", "cat", "sleeps"], labels=[0, 1, 0])
    index_to_category = {0: None, 2: "ACTION"}  # Missing category 1
    entities = instance.get_entities(index_to_category)
    assert entities == []
    
    # Test with no labels
    instance = Instance(tokens=["Simple", "text"], labels=None)
    index_to_category = {0: None, 1: "CATEGORY"}
    entities = instance.get_entities(index_to_category)
    assert entities == []

def test_dataset_creation():
    training = [Instance(tokens=["train1"], labels=[0])]
    validation = [Instance(tokens=["val1"], labels=[1])]
    test = [Instance(tokens=["test1"], labels=[2])]
    index_to_category = {0: None, 1: "PERSON", 2: "LOCATION"}
    
    dataset = Dataset(training=training, validation=validation, test=test, index_to_category=index_to_category)
    assert dataset.training == training
    assert dataset.validation == validation
    assert dataset.test == test
    assert dataset.index_to_category == index_to_category

def test_get_training_instances():
    training = [
        Instance(tokens=["train1"], labels=[0]),
        Instance(tokens=["train2"], labels=[1]),
        Instance(tokens=["train3"], labels=[2])
    ]
    dataset = Dataset(
        training=training,
        validation=[],
        test=[],
        index_to_category={0: None, 1: "PERSON", 2: "LOCATION"}
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
        Instance(tokens=["val1"], labels=[0]),
        Instance(tokens=["val2"], labels=[1])
    ]
    dataset = Dataset(
        training=[],
        validation=validation,
        test=[],
        index_to_category={0: None, 1: "PERSON"}
    )
    
    instances = dataset.get_validation_instances()
    assert instances == validation
    assert len(instances) == 2

def test_get_test_instances():
    test = [
        Instance(tokens=["test1"], labels=[0]),
        Instance(tokens=["test2"], labels=[1])
    ]
    dataset = Dataset(
        training=[],
        validation=[],
        test=test,
        index_to_category={0: None, 1: "PERSON"}
    )
    
    instances = dataset.get_test_instances()
    assert instances == test
    assert len(instances) == 2
