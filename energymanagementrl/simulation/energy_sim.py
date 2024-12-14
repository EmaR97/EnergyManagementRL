import numpy as np

from .base_sim import BaseSim
from .utils import min5, day, shuffle_array_blocks
from abc import abstractmethod


class EnergySim(BaseSim):
    """
    EnergySim simulates energy production or consumption over time based on a provided power series.
    It calculates energy values for each time step and provides sliding 24-hour windowed sums to represent
    recent energy activity.

    Attributes:
        energy_series (list[float]): Series of energy values based on input power series, adjusted by `min5` factor.
        step_index (int): Index of the current step in the energy series.
    """

    def __init__(self,
                 power_series: list[float],
                 daily_sample: int = 24,
                 forecast_steps: int = 24, seed=None
                 ) -> None:
        """
        Initializes the EnergySim instance with given parameters.

        Parameters:
            power_series (list[float]): List of power values for each time step.
        """
        super().__init__(seed)
        self.orig_energy_series: list[float] = [x * min5 for x in power_series]
        self.energy_series: list[float] = self.orig_energy_series
        self.sample_size: int = day // daily_sample
        self.forecast_steps = forecast_steps
        self.forecast_range = [i * self.sample_size for i in range(self.forecast_steps)]

    def check_max_steps(self, max_steps):
        allowed_max_steps = self.get_allowed_max_steps()
        if max_steps > allowed_max_steps:
            raise ValueError(f"Allowed max steps: {allowed_max_steps}")

    def get_allowed_max_steps(self):
        return len(self.orig_energy_series)

    def reset(self, seed=None, **kwargs):
        super().reset(seed)

        if kwargs.get('shuffle', 0) > 0:
            self.energy_series = shuffle_array_blocks(
                array=np.array(self.orig_energy_series),
                block_size=288,
                max_shift=2,
                mix_probability=1,
                random_state=self.random_state
            ).tolist()

    def get_energy(self) -> int:
        """
        Retrieves the energy at the current time step.

        Returns:
            int: Current energy in the simulation.
        """
        return int(self.energy_series[self.step_index])

    @abstractmethod
    def get_energy_sample(self) -> list[int]:
        pass

    def get_state(self):
        state = {f"energy_sample_{i}": value for i, value in enumerate(self.get_energy_sample())}
        state['energy'] = self.get_energy()
        return state
