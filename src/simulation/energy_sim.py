from .utils import min5, day


class EnergySim:
    def __init__(self, power_series: list[float], max_step: int = None, max_24h: int = None,
                 energy_type: str = "production") -> None:
        self.current_energy: int = 0
        self.energy_series: list[float] = [x * min5 for x in power_series]
        self.step_index: int = 0
        self.energy_type: str = energy_type
        self.max_step: int = int(max_step or max(self.energy_series))
        self.sliding_sum = self._get_sliding_sum()
        self.max_24h: int = int(max_24h or max(self.sliding_sum))

    def _get_sliding_sum(self):
        h24_window = day
        # Extend the list with zeros before and after for a rolling sum
        zeros_before = [0] * h24_window
        zeros_after = [0] * (h24_window * 3)
        expanded_series = zeros_before + self.energy_series + zeros_after

        # Compute the sliding sum using a moving window approach
        sliding_sum = [sum(expanded_series[i:i + h24_window]) for i in range(len(expanded_series) - h24_window + 1)]

        return sliding_sum[h24_window:]  # Trim the initial zero padding

    def reset(self, seed=0):
        self.step_index: int = 0
        self.current_energy: int = 0

    def step(self) -> int:
        self.current_energy = self.energy_series[self.step_index]
        self.step_index += 1
        return int(self.current_energy)

    def get_energy(self) -> int:
        return int(self.current_energy)

    def get_energy_last_24h(self) -> int:
        return self.sliding_sum[self.step_index]

    def get_energy_next_24h(self) -> int:
        return self.sliding_sum[self.step_index + day]

    def get_energy_24h_following_next_24h(self) -> int:
        return self.sliding_sum[self.step_index + day + day]
