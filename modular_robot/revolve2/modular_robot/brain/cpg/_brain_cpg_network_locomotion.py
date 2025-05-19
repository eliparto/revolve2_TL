from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...body.base import ActiveHinge
from .._brain import Brain
from .._brain_instance import BrainInstance
from ._brain_cpg_instance_locomotion import BrainCpgInstanceLocomotion
from ._cpg_network_structure import CpgNetworkStructure


class BrainCpgNetworkLocomotion(Brain):
    """
    A CPG (central pattern generator) brain with CPGs and connections for targeted locomotion.

    A state vector is integrated over time using a weight matrix which multiplication with the state vector sum defines the derivative of the state vector.
    I.e X' = WX

    The first `num_output_neurons` in the state vector are the outputs for the controller created by this brain.
    """

    _initial_state: npt.NDArray[np.float_]
    _weight_tensor: npt.NDArray[np.float_]
    _output_mapping: list[tuple[int, ActiveHinge]]
    _targets: list[list[float]]

    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        weight_tensor: npt.NDArray[np.float_],
        output_mapping: list[tuple[int, ActiveHinge]],
        targets: list[list[float]],
        nose: int,
    ) -> None:
        """
        Initialize this object.

        :param initial_state: The initial state of the neural network.
        :param all_weights: Tensor containing the 3 two-dimensional weight matrices.
        :param output_mapping: Marks neurons as controller outputs and map them to the correct active hinge.
        :param targets: List of targets for the robot to reach
        """
        self._initial_state = initial_state
        self._weight_tensor = weight_tensor
        self._output_mapping = output_mapping
        self._targets = targets
        self._nose = nose

    @classmethod
    def uniform_from_params(
        cls,
        params: npt.NDArray[np.float_], # 1x3n solution vector
        cpg_network_structure: CpgNetworkStructure,
        initial_state_uniform: float,
        output_mapping: list[tuple[int, ActiveHinge]],
        targets: list[list[float]],
        nose: int,
        ) -> BrainCpgNetworkLocomotion:
        """
        Create and initialize an instance of this brain from the provided parameters, assuming uniform initial state.

        :param params: Parameters for the weight matrix to be created.
        :param cpg_network_structure: The cpg network structure.
        :param initial_state_uniform: Initial state to use for all neurons.
        :param output_mapping: Marks neurons as controller outputs and map them to the correct active hinge.
        :returns: The created brain.
        """        
        initial_state = cpg_network_structure.make_uniform_state(initial_state_uniform)
        params = np.reshape(params, (3, int(len(params)/3)))
        weight_tensor = np.zeros((3, 2*cpg_network_structure.num_cpgs,
                                  2*cpg_network_structure.num_cpgs))
        
        # Fill weights tensor with weight matrices
        for i in range(3):
            weight_matrix = (
                cpg_network_structure.make_connection_weights_matrix_from_params(
                    list(params[i]))
                )
            weight_tensor[i] = weight_matrix
                    
        return BrainCpgNetworkLocomotion(
            initial_state=initial_state,
            weight_tensor=weight_tensor,
            output_mapping=output_mapping,
            targets=targets,
            nose=nose,
        )

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return BrainCpgInstanceLocomotion(
            initial_state=self._initial_state,
            weight_tensor=self._weight_tensor,
            output_mapping=self._output_mapping,
            targets=self._targets,
            nose=self._nose,
        )
