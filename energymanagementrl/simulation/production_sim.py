import numpy as np

from .utils import sparse_matrix
from .energy_sim import EnergySim
from .weather_sim import WeatherSim


class ProductionSim(EnergySim):
    def __init__(self,
                 power_series: list[float],
                 daily_sample: int = 24,
                 forecast_steps: int = 24,
                 w_sim: WeatherSim = None,
                 seed=None
                 ) -> None:
        """
        Initializes the EnergySim instance with given parameters.

        Parameters:
            power_series (list[float]): List of power values for each time step.
        """
        super().__init__(power_series, daily_sample, forecast_steps, seed)
        self.forecast_steps = forecast_steps
        self.forecast_range = [i * self.sample_size for i in range(self.forecast_steps)]
        self.energy_samples = self._get_sliding_sum()
        self.w_sim: WeatherSim = w_sim

    def step(self, **inputs):
        super().step(**inputs)
        if self.w_sim is not None:
            self.w_sim.step(**inputs)
        return self.get_energy()

    def reset(self, seed=None):
        super().reset(seed)
        if self.w_sim is not None:
            self.w_sim.reset(seed)

    def get_energy(self) -> int:
        current_energy = super().get_energy()
        if self.w_sim is not None:
            current_energy *= (1 - self.w_sim.get_attenuation())
        return current_energy

    def _get_sliding_sum(self) -> list[float]:
        """
        Calculates the 24-hour sliding energy sums for the energy series.

        Returns:
            list[float]: List of 24-hour sliding average energy values.
        """
        window = self.sample_size
        expanded_series = self.energy_series + ([0] * ((self.forecast_steps + 1) * window))
        return [sum(expanded_series[i:i + window]) / window
                for i in range(len(expanded_series) - window + 1)]

    def get_energy_sample(self) -> list[int]:
        """
        Predicts the total energy for the upcoming 24-hour window.

        Returns:
            int: Estimated energy for the next 24 hours.
        """
        return [int(self.energy_samples[self.step_index + i]) for i in self.forecast_range]

    def get_state(self):
        state = {f"energy_sample_{i}": value for i, value in
                 enumerate(np.array(self.get_energy_sample()) @ sparse_matrix)}
        state['energy'] = self.get_energy()
        if self.w_sim is not None:
            state['w_sim'] = self.w_sim.get_state()
        return state
