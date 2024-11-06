from .utils import min5


class GridSim:
    def __init__(self, feed_in_max, feed_in_min, voltage_max, voltage_min, max_taken_from, energy_price_sell,
                 energy_price_buy, voltage_series):
        self.feed_in_max_known = feed_in_max
        self.feed_in_min = feed_in_min
        self.voltage_max = voltage_max
        self.voltage_min_known = voltage_min
        self.max_taken_from = max_taken_from
        self.energy_price_sell = energy_price_sell
        self.energy_price_buy = energy_price_buy
        self.current_feed_to_grid = 0.
        self.current_taken_from_grid = 0.
        self.voltage_series = voltage_series
        self.step_index = 0
        self.power_for_voltage = (
                (self.feed_in_max_known - self.feed_in_min)
                / (self.voltage_max - self.voltage_min_known)
        )

    def reset(self, seed=0):
        self.step_index: int = 0
        self.current_feed_to_grid: int = 0
        self.current_taken_from_grid: int = 0

    def step(self, energy_balance: int) -> int:
        self.current_feed_to_grid = 0
        self.current_taken_from_grid = 0
        self.step_index += 1
        if energy_balance > 0:  # Excess energy: try to feed into the grid
            self.current_feed_to_grid = min(energy_balance, int(self.get_grid_acceptance()))
            energy_balance -= self.current_feed_to_grid
        elif energy_balance < 0:  # Energy deficit: try to take from the grid
            self.current_taken_from_grid = min(-energy_balance, int(self.max_taken_from * min5))
            energy_balance += self.current_taken_from_grid

        return int(energy_balance)

    def get_grid_acceptance(self):
        current_voltage = self.voltage_series[self.step_index]
        grid_acceptance_capacity = self._map_voltage_to_power(current_voltage) * min5
        return grid_acceptance_capacity

    def _map_voltage_to_power(self, voltage):
        if voltage >= self.voltage_max:
            return self.feed_in_min

        power_acceptance = self.feed_in_min + (self.voltage_max - voltage) * self.power_for_voltage
        return power_acceptance

    def get_feed_to(self):
        return self.current_feed_to_grid

    def get_taken_from(self):
        return self.current_taken_from_grid
