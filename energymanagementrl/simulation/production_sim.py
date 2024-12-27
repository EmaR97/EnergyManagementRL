import numpy as np

from .utils import sparse_matrix, min5, shuffle_array_blocks
from .energy_sim import EnergySim


class ProductionSim(EnergySim):
    def __init__(self,
                 power_series: list[float],
                 daily_sample: int = 24,
                 forecast_steps: int = 24,
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

    def get_allowed_max_steps(self):
        return len(self.orig_energy_series) - self.forecast_steps * self.sample_size

    def step(self, **inputs):
        super().step(**inputs)
        return self.get_energy()

    def reset(self, seed=None, **kwargs):
        super().reset(seed, **kwargs)

    def get_energy(self) -> int:
        return super().get_energy()

    def get_energy_sample(self) -> list[int]:
        sample = [int(self.energy_series[self.step_index + i]) for i in self.forecast_range]
        return np.array(sample) @ sparse_matrix

    def get_state(self):
        state = {f"energy_sample_{i}": value for i, value in enumerate(self.get_energy_sample())}
        state['energy'] = self.get_energy()
        return state

class ProductionSimFromReal(ProductionSim):
    def __init__(self,
                 power_series: list[float],
                 optimal_power_series: list[float],
                 weather_power_series: list[float],
                 daily_sample: int = 24,
                 forecast_steps: int = 24,
                 seed=None
                 ) -> None:
        super().__init__(power_series, daily_sample, forecast_steps, seed)
        self.optimal_power_series = [x * min5 for x in optimal_power_series]
        self.weather_power_series = [x * min5 for x in weather_power_series]
        self.residual_series = []
        self.precompute_residual()
        self.original_residual_series = self.residual_series
        self.original_weather_power_series = self.weather_power_series

    def get_state(self):
        state = super().get_state()
        state.update({f"residual_sample_{i}": value for i, value in enumerate(self.get_residual_sample())})
        return state

    def reset(self, seed=None, **kwargs):
        super().reset(seed, **kwargs)
        shuffle = kwargs.get('shuffle', 0)
        if shuffle > 0:
            self.energy_series, self.weather_power_series, self.residual_series = [
                array.tolist() for array in shuffle_array_blocks(
                    arrays=[
                        np.array(self.orig_energy_series),
                        np.array(self.original_weather_power_series),
                        np.array(self.original_residual_series)
                    ],
                    block_size=288,
                    max_shift=2,
                    mix_probability=.25,
                    random_state=self.random_state
                )]

    def precompute_residual(self):
        self.residual_series = []
        for i in range(len(self.optimal_power_series)):
            residual = abs(self.optimal_power_series[i] - self.weather_power_series[i])
            self.residual_series.append(residual)

    def get_residual_sample(self) -> list[int]:
        sample = [int(self.residual_series[self.step_index + i]) for i in self.forecast_range]
        return np.array(sample) @ sparse_matrix

    def get_energy_sample(self) -> list[int]:
        sample = [int(self.weather_power_series[self.step_index + i]) for i in self.forecast_range]
        return np.array(sample) @ sparse_matrix
