import concurrent.futures
import logging
import os

from revolve2.simulation.scene import SimulationState
from revolve2.simulation.simulator import Batch, Simulator

from ._simulate_manual_scene import simulate_manual_scene
from ._simulate_scene import simulate_scene
from .viewers import ViewerType


class LocalSimulator(Simulator):
    """Simulator using MuJoCo."""

    _headless: bool
    _start_paused: bool
    _num_simulators: int
    _cast_shadows: bool
    _fast_sim: bool
    _manual_control: bool
    _viewer_type: ViewerType

    def __init__(
        self,
        headless: bool = False,
        start_paused: bool = False,
        num_simulators: int = 1,
        cast_shadows: bool = False,
        fast_sim: bool = False,
        manual_control: bool = False,
        viewer_type: ViewerType | str = ViewerType.CUSTOM,
    ):
        """
        Initialize this object.

        :param headless: If True, the simulation will not be rendered. This drastically improves performance.
        :param start_paused: If True, start the simulation paused. Only possible when not in headless mode.
        :param num_simulators: The number of simulators to deploy in parallel. They will take one core each but will share space on the main python thread for calculating control.
        :param cast_shadows: Whether shadows are cast in the simulation.
        :param fast_sim: Whether more complex rendering prohibited.
        :param manual_control: Whether the simulation should be controlled manually.
        :param viewer_type: The viewer-implementation to use in the local simulator.
        """
        assert (
            headless or num_simulators == 1
        ), "Cannot have parallel simulators when visualizing."

        assert not (
            headless and start_paused
        ), "Cannot start simulation paused in headless mode."

        self._headless = headless
        self._start_paused = start_paused
        self._num_simulators = num_simulators
        self._cast_shadows = cast_shadows
        self._fast_sim = fast_sim
        self._manual_control = manual_control
        self._viewer_type = (
            ViewerType.from_string(viewer_type)
            if isinstance(viewer_type, str)
            else viewer_type
        )

    def simulate_batch(self, batch: Batch) -> list[list[SimulationState]]:
        """
        Simulate the provided batch by simulating each contained scene.

        :param batch: The batch to run.
        :returns: List of simulation states in ascending order of time.
        :raises Exception: If manual control is selected, but headless is enabled.
        """
        logging.debug("Starting simulation batch with MuJoCo.")

        control_step = 1.0 / batch.parameters.control_frequency
        sample_step = (
            None
            if batch.parameters.sampling_frequency is None
            else 1.0 / batch.parameters.sampling_frequency
        )

        if batch.record_settings is not None:
            os.makedirs(
                batch.record_settings.video_directory,
                exist_ok=batch.record_settings.overwrite,
            )

        if self._manual_control:
            if self._headless:
                raise Exception("Manual control only works with rendered simulations.")
            for scene in batch.scenes:
                simulate_manual_scene(scene=scene)
            return [[]]

        if self._num_simulators > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self._num_simulators
            ) as executor:
                futures = [
                    executor.submit(
                        simulate_scene,  # This is the function to call, followed by the parameters of the function
                        scene_index,
                        scene,
                        self._headless,
                        batch.record_settings,
                        self._start_paused,
                        control_step,
                        sample_step,
                        batch.parameters.simulation_time,
                        batch.parameters.simulation_timestep,
                        self._cast_shadows,
                        self._fast_sim,
                        self._viewer_type,
                    )
                    for scene_index, scene in enumerate(batch.scenes)
                ]
                results = [future.result() for future in futures]
        else:
            results = [
                simulate_scene(
                    scene_index,  # This is the function to call, followed by the parameters of the function
                    scene,
                    self._headless,
                    batch.record_settings,
                    self._start_paused,
                    control_step,
                    sample_step,
                    batch.parameters.simulation_time,
                    batch.parameters.simulation_timestep,
                    self._cast_shadows,
                    self._fast_sim,
                    self._viewer_type,
                )
                for scene_index, scene in enumerate(batch.scenes)
            ]

        logging.debug("Finished batch.")

        return results
