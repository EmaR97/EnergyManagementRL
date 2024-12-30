import numpy as np

from .energy_sim import EnergySim
from .utils import SmoothedHistory, shuffle_array_blocks


class ConsumptionSim(EnergySim):
    """
    EnergySim simulates energy production or consumption over time based on a provided power series.
    It calculates energy values for each time step and provides sliding 24-hour windowed sums to represent
    recent energy activity.

    Attributes:
        energy_series (list[float]): Series of energy values based on input power series, adjusted by `min5` factor.
        step_index (int): Index of the current step in the energy series.
        energy_samples (list[float]): Sliding 24-hour energy sums calculated from `energy_series`.
    """

    def __init__(self, power_series: list[float],
                 daily_sample: int = 24, forecast_steps: int = 24, seed=None) -> None:
        """
        Initializes the EnergySim instance with given parameters.

        Parameters:
            power_series (list[float]): List of power values for each time step.
        """
        super().__init__(power_series, daily_sample, forecast_steps, seed)
        history = SmoothedHistory(12, self.forecast_steps * self.sample_size)
        self.max_shift = 2
        self.mix_probability = .25
        self.energy_samples = [history.get_smoothed_history(sample, self.forecast_range) for sample in
                               self.energy_series]

    def step(self, **inputs):
        super().step(**inputs)
        return self.get_energy()

    def reset(self, seed=None, **kwargs):
        super().reset(seed, **kwargs)
        if kwargs.get('shuffle', 0) > 0:
            self.energy_series = shuffle_array_blocks(
                arrays=[np.array(self.orig_energy_series)],
                block_size=288,
                max_shift=self.max_shift,
                mix_probability=self.mix_probability,
                random_state=self.random_state
            )[0].tolist()

    def get_energy_sample(self) -> list[int]:
        """
        Predicts the total energy for the upcoming 24-hour window.

        Returns:
            int: Estimated energy for the next 24 hours.
        """
        return self.energy_samples[self.step_index]
