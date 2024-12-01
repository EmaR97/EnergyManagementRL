from .utils import min5, day


class EnergySim:
    """
    EnergySim simulates energy production or consumption over time based on a provided power series.
    It calculates energy values for each time step and provides sliding 24-hour windowed sums to represent
    recent energy activity.

    Attributes:
        current_energy (int): Energy at the current time step.
        energy_series (list[float]): Series of energy values based on input power series, adjusted by `min5` factor.
        step_index (int): Index of the current step in the energy series.
        energy_type (str): Type of energy represented ("production" or "consumption").
        max_step (int): Maximum energy per step, capped at the highest value in `energy_series` if not specified.
        sliding_sum (list[float]): Sliding 24-hour energy sums calculated from `energy_series`.
        max_24h (int): Maximum 24-hour energy, capped at the highest value in `sliding_sum` if not specified.
    """

    def __init__(self, power_series: list[float], max_step: int = None, max_24h: int = None,
                 energy_type: str = "production", daily_sample: int = 6) -> None:
        """
        Initializes the EnergySim instance with given parameters.

        Parameters:
            power_series (list[float]): List of power values for each time step.
            max_step (int, optional): Maximum energy allowed per step. Defaults to the max of energy series.
            max_24h (int, optional): Maximum allowed energy over 24 hours. Defaults to the max of sliding sum.
            energy_type (str, optional): Type of energy, either "production" or "consumption" (default: "production").
        """
        self.current_energy: int = 0
        self.energy_series: list[float] = [x * min5 for x in power_series]
        self.step_index: int = 0
        self.energy_type: str = energy_type
        self.max_step: int = int(max_step or max(self.energy_series))
        self.sample_size: int = day // daily_sample
        self.forecast_range = range(0, day * 2, self.sample_size)
        self.sliding_sum = self._get_sliding_sum()
        self.max_24h: int = int(max_24h or max(self.sliding_sum))

    def _get_sliding_sum(self) -> list[float]:
        """
        Calculates the 24-hour sliding energy sums for the energy series.

        Returns:
            list[float]: List of 24-hour sliding average energy values.
        """
        window = self.sample_size
        expanded_series = self.energy_series + ([0] * (day * 2 + window))
        sliding_sum = [sum(expanded_series[i:i + window]) / window
                       for i in range(len(expanded_series) - window + 1)]

        return sliding_sum  # Remove initial padding

    def reset(self) -> None:
        """
        Resets the simulation to the starting conditions.
        """
        self.step_index: int = 0
        self.current_energy: int = 0

    def step(self) -> int:
        """
        Advances the simulation by one time step and updates the current energy.

        Returns:
            int: Energy for the current time step.
        """
        self.current_energy = self.energy_series[self.step_index]
        self.step_index += 1
        return int(self.current_energy)

    def get_energy(self) -> int:
        """
        Retrieves the energy at the current time step.

        Returns:
            int: Current energy in the simulation.
        """
        return int(self.current_energy)

    def get_energy_forecast(self) -> list[int]:
        """
        Predicts the total energy for the upcoming 24-hour window.

        Returns:
            int: Estimated energy for the next 24 hours.
        """

        return [int(self.sliding_sum[self.step_index + i]) for i in self.forecast_range]
