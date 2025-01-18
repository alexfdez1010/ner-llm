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
    assert instance.to_string() == "hello world"

def test_dataset_creation():
    training = [Instance(tokens=["train1"], labels=[0])]
    validation = [Instance(tokens=["val1"], labels=[1])]
    test = [Instance(tokens=["test1"], labels=[2])]
    
    dataset = Dataset(training=training, validation=validation, test=test)
    assert dataset.training == training
    assert dataset.validation == validation
    assert dataset.test == test

def test_get_training_instances():
    training = [
        Instance(tokens=["train1"], labels=[0]),
        Instance(tokens=["train2"], labels=[1]),
        Instance(tokens=["train3"], labels=[2])
    ]
    dataset = Dataset(
        training=training,
        validation=[],
        test=[]
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
        test=[]
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
        test=test
    )
    
    instances = dataset.get_test_instances()
    assert instances == test
    assert len(instances) == 2
