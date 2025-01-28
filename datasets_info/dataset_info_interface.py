from abc import ABC, abstractmethod
from typing import List

from dataset import Dataset
from model.category import Category


class DatasetInfo(ABC):
    """Interface for dataset information."""
    
    @abstractmethod
    def load_dataset(self) -> Dataset:
        """Load the dataset.
        
        Returns:
            Dataset: The loaded dataset
        """
    
    @abstractmethod
    def categories(self) -> List[Category]:
        """Get the categories of the dataset.
        
        Returns:
            List[Category]: List of categories in the dataset
        """
    
    @abstractmethod
    def language(self) -> str:
        """Get the language of the dataset.
        
        Returns:
            str: Language code of the dataset
        """
    
    @abstractmethod
    def example_prompt(self) -> str:
        """Get an example prompt for the dataset.
        
        Returns:
            str: Example prompt for the dataset
        """
