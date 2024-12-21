import numpy as np

from .utils import sparse_matrix, min5, shuffle_array_blocks
from .energy_sim import EnergySim
from .weather_sim import WeatherSim


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


# Derived version of ProductionSim (requires w_sim)
class ProductionSimWithWeather(ProductionSim):
    def __init__(self,
                 power_series: list[float],
                 daily_sample: int = 24,
                 forecast_steps: int = 24,
                 w_sim: WeatherSim = None,
                 seed=None
                 ) -> None:
        """
        Initializes the EnergySim instance with given parameters, including WeatherSim.

        Parameters:
            power_series (list[float]): List of power values for each time step.
            w_sim (WeatherSim): Weather simulation instance.
        """
        super().__init__(power_series, daily_sample, forecast_steps, seed)
        self.w_sim: WeatherSim = w_sim

    def step(self, **inputs):
        super().step(**inputs)
        if self.w_sim is not None:
            self.w_sim.step(**inputs)
        return self.get_energy()

    def reset(self, seed=None, **kwargs):
        super().reset(seed)
        if self.w_sim is not None:
            self.w_sim.reset(seed)
            self.w_sim.time_steps = len(self.energy_series)

    def get_energy(self) -> int:
        current_energy = super().get_energy()
        if self.w_sim is not None:
            current_energy *= (1 - self.w_sim.get_attenuation())
        return current_energy

    def get_state(self):
        state = super().get_state()
        if self.w_sim is not None:
            state['w_sim'] = self.w_sim.get_state()
        return state


class ProductionSimWithError(ProductionSim):
    def __init__(self,
                 power_series: list[float],
                 optimal_power_series: list[float],
                 daily_sample: int = 24,
                 forecast_steps: int = 24,
                 error_divisor: int = 3,
                 seed=None
                 ) -> None:
        super().__init__(power_series, daily_sample, forecast_steps, seed)
        self.original_optimal_power_series = [x * min5 for x in optimal_power_series]
        self.optimal_power_series = self.original_optimal_power_series
        self.error_divisor = error_divisor
        self.precomputed_energy = []
        self.residual_series = []
        self.precompute_energy()

    def get_state(self):
        state = super().get_state()
        state.update({f"residual_sample_{i}": value for i, value in enumerate(self.get_residual_sample())})
        return state

    def reset(self, seed=None, **kwargs):
        super().reset(seed, **kwargs)

    def precompute_energy(self):
        """Precompute updated energy values based on optimal_power_series."""
        self.precomputed_energy = []
        self.residual_series = []
        for i in range(len(self.optimal_power_series)):
            residual = abs(self.optimal_power_series[i] - self.energy_series[i])
            self.residual_series.append(residual)
            noise = self.random_state.normal(
                0,
                max(0.1, min(residual, self.energy_series[i]) / self.error_divisor)
            )
            updated_energy = min(self.optimal_power_series[i], max(0., self.energy_series[i] + noise))
            self.precomputed_energy.append(updated_energy)

    def get_energy(self):
        energy = self.precomputed_energy[self.step_index]
        return energy

    def get_residual_sample(self):
        sample = [int(self.residual_series[self.step_index + i]) for i in self.forecast_range]
        return np.array(sample) @ sparse_matrix


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
            state = self.random_state.get_state()
            self.energy_series = shuffle_array_blocks(
                array=np.array(self.orig_energy_series),
                block_size=288,
                max_shift=2,
                mix_probability=.25,
                random_state=self.random_state
            ).tolist()
            self.random_state.set_state(state)
            self.weather_power_series = shuffle_array_blocks(
                array=np.array(self.original_weather_power_series),
                block_size=288,
                max_shift=2,
                mix_probability=.25,
                random_state=self.random_state
            ).tolist()
            self.random_state.set_state(state)
            self.residual_series = shuffle_array_blocks(
                array=np.array(self.original_residual_series),
                block_size=288,
                max_shift=2,
                mix_probability=.25,
                random_state=self.random_state
            ).tolist()

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
