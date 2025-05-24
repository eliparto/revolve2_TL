"""An Abstraction Layer for a population/body analyzer in an Evolutionary Porocess"""

from abc import ABC, abstractmethod
from typing import Any

TPopulation = (
    Any  # An alias for Any signifying that a population can vary depending on use-case.
)

class Morpho(ABC):
    """An analyzer object enabling us to visualize bodies and find a robot's frontal orientation."""
    
    @abstractmethod
    def findNose(self, population: TPopulation) -> TPopulation:
        """
        Find the noses of all robots in a population.

        :param population: Population of robots
        :return: Population with nose orientation stored in 'Individual'
        """