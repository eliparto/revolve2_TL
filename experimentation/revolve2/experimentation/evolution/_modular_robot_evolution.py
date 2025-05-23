from typing import Any

from .abstract_elements import Evaluator, Evolution, Learner, Reproducer, Selector, Morpho

TPopulation = (
    Any  # An alias for Any signifying that a population can vary depending on use-case.
)


class ModularRobotEvolution(Evolution):
    """An object to encapsulate the general functionality of an evolutionary process for modular robots."""

    _parent_selection: Selector
    _survivor_selection: Selector
    _learner: Learner
    _reproducer: Reproducer
    _morpho: Morpho

    def __init__(
        self,
        parent_selection: Selector,
        survivor_selection: Selector,
        reproducer: Reproducer,
        learner: Learner,
        morpho: Morpho,
    ) -> None:
        """
        Initialize the ModularRobotEvolution object to make robots evolve.

        :param parent_selection: Selector object for the parents for reproduction.
        :param survivor_selection: Selector object for the survivor selection.
        :param evaluator: Evaluator object for evaluation.
        :param reproducer: The reproducer object.
        :param learner: Learning object for learning.
        """
        self._parent_selection = parent_selection
        self._survivor_selection = survivor_selection
        self._learner = learner
        self._reproducer = reproducer
        self._morpho = morpho

    def step(self, population: TPopulation, **kwargs: Any) -> TPopulation:
        """
        Step the current evolution by one iteration.

        This implementation follows the following schedule:

            [Parent Selection] ------------------> [Reproduction]
            
                    ^                                     |
                    |                        [Learning (children only)]
                    |                                     |
                    |                                     ⌄

            [Survivor Selection] <----------- [Evaluation of Children]

        The schedule can be easily adapted and reorganized for your needs.

        :param population: The current population.
        :param kwargs: Additional keyword arguments to use in the step.
        :return: The population resulting from the step
        """
        parent_pairs = self._parent_selection.select(population)        # Generate parent pairs (parents have already learned)
        children = self._reproducer.reproduce(parent_pairs, population) # Parents reproduce and make children
        children = self._morpho.findNose(children)                      # Find frontal orientation of children
        children = self._learner.learn(children)                        # Put children through learning process
        survivors = self._survivor_selection.select(                    # Find survivors for the next generation
            population=population,
            children=children,
        )
        
        return survivors
