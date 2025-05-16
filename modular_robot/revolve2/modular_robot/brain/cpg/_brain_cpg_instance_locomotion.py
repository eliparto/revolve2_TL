import numpy as np
import numpy.typing as npt

from ..._modular_robot_control_interface import ModularRobotControlInterface
from ...body.base import ActiveHinge
from ...sensor_state import ModularRobotSensorState
from .._brain_instance import BrainInstance


class BrainCpgInstanceLocomotion(BrainInstance):
    """
    CPG network brain for targeted locomotion.

    A state array that is integrated over time following the differential equation `X'=WX`.
    W is a weight matrix that is multiplied by the state array.
    The outputs of the controller are defined by the `outputs`, a list of indices for the state array.
    """

    _initial_state: npt.NDArray[np.float_]  # Square matrix of cpg states
    _weight_tensor: npt.NDArray[np.float_]  # 3xnxn tensor matching number of neurons
    _output_mapping: list[tuple[int, ActiveHinge]]
    _target: list[list[float]]
    _nose: int # Frontal orientation of robot

    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        weight_tensor: npt.NDArray[np.float_],
        output_mapping: list[tuple[int, ActiveHinge]],
        targets:list[list[float]],
        nose: int
    ) -> None:
        # TODO: Functionality when a target is reached
        """
        Initialize this CPG Brain Instance.

        :param initial_state: The initial state of the neural network.
        :param weight_matrix: The weight matrix used during integration.
        :param output_mapping: Marks neurons as controller outputs and map them to the correct active hinge.
        """
        #TODO: Determine if the weight matrices need their own initial states
        assert initial_state.ndim == 1
        assert weight_tensor.ndim == 3
        assert weight_tensor.shape[1] == weight_tensor.shape[2]
        assert initial_state.shape[0] == weight_tensor.shape[1]
        assert all([i >= 0 and i < len(initial_state) for i, _ in output_mapping])
        assert nose >= 0

        self._state = initial_state
        self._weight_tensor = weight_tensor
        self._output_mapping = output_mapping
        self._targets = targets
        self._nose = nose

    @staticmethod
    def _rk45(
        state: npt.NDArray[np.float_], A: npt.NDArray[np.float_], dt: float
    ) -> npt.NDArray[np.float_]:
        """
        Calculate the next state using the RK45 method.

        This implementation of the Runge–Kutta–Fehlberg method allows us to improve accuracy of state calculations by comparing solutions at different step sizes.
        For more info see: See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method.
        RK45 is a method of order 4 with an error estimator of order 5 (Fehlberg, E. (1969). Low-order classical Runge-Kutta formulas with stepsize control. NASA Technical Report R-315.).

        :param state: The current state of the network.
        :param A: The weights matrix of the network.
        :param dt: The step size (elapsed simulation time).
        :return: The new state.
        """
        A1: npt.NDArray[np.float_] = np.matmul(A, state)
        A2: npt.NDArray[np.float_] = np.matmul(A, (state + dt / 2 * A1))
        A3: npt.NDArray[np.float_] = np.matmul(A, (state + dt / 2 * A2))
        A4: npt.NDArray[np.float_] = np.matmul(A, (state + dt * A3))
        state = state + dt / 6 * (A1 + 2 * (A2 + A3) + A4)
        return np.clip(state, a_min=-1, a_max=1)
    
    def action(self, data) -> None:
        """
        Calculate the angle to the target and determine which action to take.
        """
        # robot_pose = sensor_state._simulation_state.get_pose()
        # robot_pos = np.array([robot_pose.position_x,
        #                       robot_pose.position_y])
        # robot_quat = robot_pose.orientation
        
        # print(robot_pos)
        # print(robot_quat)
        
        return 0

    def control(
        self,
        dt: float,
        sensor_state: ModularRobotSensorState,
        control_interface: ModularRobotControlInterface,
        data,
    ) -> None:
        # TODO: Add data param to description.  Also in handles code.
        """
        Control the modular robot.

        Sets the active hinge targets to the values in the state array as defined by the mapping provided in the constructor.

        :param dt: Elapsed seconds since last call to this function.
        :param sensor_state: Interface for reading the current sensor state.
        :param control_interface: Interface for controlling the robot.
        """
        # Choose the weight matrix for the requested movement
        idx = self.action(data)
        weight_matrix = self._weight_tensor[idx]
        
        # Integrate ODE to obtain new state.
        self._state = self._rk45(self._state, weight_matrix, dt)

        # Set active hinge targets to match newly calculated state.
        for state_index, active_hinge in self._output_mapping:
            control_interface.set_active_hinge_target(
                active_hinge, float(self._state[state_index]) * active_hinge.range
            )
