from dataclasses import dataclass
from random import shuffle
from typing import Optional

@dataclass
class Instance:
  """A NER instance of the dataset."""
  tokens: list[str]
  labels: Optional[list[int]]   

  def __str__(self):
    return " ".join(self.tokens)
  
  def get_entities(self, index_to_category: dict[int, str]):
    if self.labels is None:
      return []
    return list(
      filter(lambda x: x is not None, 
        map(index_to_category.get, self.labels)
      )
    )


class Dataset:
  """A dataset of NER instances."""
  def __init__(
    self, 
    training: list[Instance],
    validation: list[Instance],
    test: list[Instance],
    index_to_category: dict[int, str]
  ):
    self.training = training
    self.validation = validation
    self.test = test
    self.index_to_category = index_to_category

  def get_training_instances(self, num_instances: Optional[int] = None): 
    """
    Returns a list of training instances randomly shuffled.

    Args:
      num_instances: The number of instances to return. If None, then all instances are returned.
    """
    shuffle(self.training)
    
    if num_instances is None:
      return self.training
    
    return self.training[:num_instances]
  
  def get_validation_instances(self): 
    """
    Returns a list of validation instances.
    """
    return self.validation
  
  def get_test_instances(self): 
    """
    Returns a list of test instances.
    """
    return self.test
