from . import EnergySim
from scipy.ndimage import gaussian_filter1d


class ConsumptionSim(EnergySim):
    """
    EnergySim simulates energy production or consumption over time based on a provided power series.
    It calculates energy values for each time step and provides sliding 24-hour windowed sums to represent
    recent energy activity.

    Attributes:
        current_energy (int): Energy at the current time step.
        energy_series (list[float]): Series of energy values based on input power series, adjusted by `min5` factor.
        step_index (int): Index of the current step in the energy series.
        max_step (int): Maximum energy per step, capped at the highest value in `energy_series` if not specified.
        energy_samples (list[float]): Sliding 24-hour energy sums calculated from `energy_series`.
        max_24h (int): Maximum 24-hour energy, capped at the highest value in `sliding_sum` if not specified.
    """

    def __init__(self, power_series: list[float], max_step: int = None, max_24h: int = None,
                 daily_sample: int = 6) -> None:
        """
        Initializes the EnergySim instance with given parameters.

        Parameters:
            power_series (list[float]): List of power values for each time step.
            max_step (int, optional): Maximum energy allowed per step. Defaults to the max of energy series.
            max_24h (int, optional): Maximum allowed energy over 24 hours. Defaults to the max of sliding sum.
        """
        super().__init__(power_series, max_step, max_24h, daily_sample)
        self.energy_samples = [get_smoothed_history(sample, self.forecast_range, self.forecast_steps*self.sample_size) for sample in
                               self.energy_series]

    def get_energy_sample(self) -> list[int]:
        """
        Predicts the total energy for the upcoming 24-hour window.

        Returns:
            int: Estimated energy for the next 24 hours.
        """

        return self.energy_samples[self.step_index]


buffer = []
sigma = 12


def get_smoothed_history(new_point, _range, steps):
    global buffer
    if not len(buffer):
        buffer = [new_point] * steps
    buffer.append(new_point)
    buffer.pop(0)
    return [gaussian_filter1d(buffer, sigma=sigma).tolist()[i] for i in _range]
