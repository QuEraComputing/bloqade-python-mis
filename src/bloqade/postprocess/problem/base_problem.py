# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:10:30 2023

@author: jwurtz
"""

from abc import ABC, abstractmethod, abstractstaticmethod
import bloqade.postprocess as postprocess

class Problem(ABC):

    """
    The `Problem` class is an abstract base class that defines the interface for a generic optimization problem.
    
    Derived classes should implement specific problem definitions and the associated methods to compute costs, 
    load data, and serialize the problem instance.
    
    Attributes:
    (Since this is an abstract class, there are no directly defined attributes. Any specific attributes 
    should be defined in the derived classes.)
    """

    
    @abstractmethod
    def cost(self,solution)->float:

        """
        Computes the cost of a given candidate solution.
        
        Derived classes should provide a concrete implementation of this method that computes and returns 
        the cost associated with the provided solution based on the specific problem definition.
        
        Args:
        - solution: The candidate solution whose cost needs to be evaluated.
        
        Returns:
        - float: The computed cost of the candidate solution.
        """

        pass
    
    @abstractstaticmethod
    def from_file(self, recorder:postprocess.record) -> None:

        """
        Loads problem data from a serialized string stored in a file.
        
        Derived classes should provide a concrete implementation of this method that reads the serialized 
        problem data from the specified file and updates the problem instance accordingly.
        
        Args:
        - recorder : a postprocess.record class which saves data locally or remotely
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """
        Serializes the problem instance into a string representation.
        
        Derived classes should provide a concrete implementation of this method that returns a string 
        representation of the problem instance, suitable for storage or debugging purposes.
        
        Returns:
        - str: The string representation of the problem instance.
        """
    
    
    def to_file(self, recorder:postprocess.record) -> bool:
        """
        Saves the serialized string representation of the problem instance to a record.
        
        Args:
        - recorder : a postprocess.record class which saves data locally or remotely
        """
        recorder.put(self)
    