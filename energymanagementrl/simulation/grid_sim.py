from .utils import min5


class GridSim:
    """
    GridSim simulates grid interaction by managing power flow to and from the grid based on voltage levels,
    energy balance, and grid acceptance limits. It enables modeling of energy exchange between a system
    and the power grid.

    Attributes:
        feed_in_max_known (float): Maximum power that can be fed into the grid.
        feed_in_min (float): Minimum power fed into the grid when voltage is at maximum.
        voltage_max (float): Maximum grid voltage level.
        voltage_min_known (float): Minimum grid voltage level corresponding to max feed-in capacity.
        max_taken_from (float): Maximum power that can be taken from the grid.
        energy_price_sell (float): Price per unit of energy sold to the grid.
        energy_price_buy (float): Price per unit of energy bought from the grid.
        current_feed_to_grid (float): Current power fed into the grid.
        current_taken_from_grid (float): Current power taken from the grid.
        voltage_series (list[float]): Series of voltage levels over time.
        step_index (int): Index for tracking the current time step.
        power_for_voltage (float): Power per voltage unit for calculating grid acceptance based on voltage.
    """

    def __init__(self, feed_in_max, feed_in_min, voltage_max, voltage_min, max_taken_from, energy_price_sell,
                 energy_price_buy, voltage_series):
        """
        Initializes the GridSim instance with the given parameters.

        Parameters:
            feed_in_max (float): Maximum power allowed for feeding into the grid.
            feed_in_min (float): Minimum feed-in power level when voltage is at maximum.
            voltage_max (float): Maximum voltage level for the grid.
            voltage_min (float): Minimum voltage level corresponding to max feed-in capacity.
            max_taken_from (float): Maximum power that can be drawn from the grid.
            energy_price_sell (float): Price per unit energy for selling to the grid.
            energy_price_buy (float): Price per unit energy for buying from the grid.
            voltage_series (list[float]): Series of voltage levels over time.
        """
        self.feed_in_max_known = feed_in_max
        self.feed_in_min = feed_in_min
        self.voltage_max = voltage_max
        self.voltage_min_known = voltage_min
        self.max_taken_from = max_taken_from
        self.energy_price_sell = energy_price_sell
        self.energy_price_buy = energy_price_buy
        self.current_feed_to_grid = 0.0
        self.current_taken_from_grid = 0.0
        self.voltage_series = voltage_series
        self.step_index = 0
        self.power_for_voltage = (
                (self.feed_in_max_known - self.feed_in_min)
                / (self.voltage_max - self.voltage_min_known)
        )

    def reset(self):
        """
        Resets the simulation, clearing the current feed and draw values and setting the step index to zero.
        """
        self.step_index = 0
        self.current_feed_to_grid = 0
        self.current_taken_from_grid = 0

    def step(self, energy_balance: float) -> int:
        """
        Simulates one step of energy balance interaction with the grid.
        Positive energy balance indicates surplus energy to feed into the grid, while
        negative balance indicates a deficit to be covered by drawing from the grid.

        Parameters:
            energy_balance (float): Net energy balance (positive for surplus, negative for deficit).

        Returns:
            int: Remaining energy balance after adjusting for feed-in or draw from the grid.
        """
        self.current_feed_to_grid = 0
        self.current_taken_from_grid = 0
        self.step_index += 1

        if energy_balance > 0:  # Surplus energy
            self.current_feed_to_grid = min(energy_balance, int(self.get_grid_acceptance()))
            energy_balance -= self.current_feed_to_grid
        elif energy_balance < 0:  # Deficit
            self.current_taken_from_grid = min(-energy_balance, int(self.max_taken_from * min5))
            energy_balance += self.current_taken_from_grid

        return int(energy_balance)

    def get_grid_acceptance(self) -> float:
        """
        Determines the current grid's acceptance capacity based on the voltage at the current step.

        Returns:
            float: Maximum power the grid can accept at the current voltage level.
        """
        current_voltage = self.voltage_series[self.step_index]
        grid_acceptance_capacity = self._map_voltage_to_power(current_voltage) * min5
        return grid_acceptance_capacity

    def _map_voltage_to_power(self, voltage: float) -> float:
        """
        Maps a given voltage to a power acceptance level, decreasing acceptance as voltage approaches maximum.

        Parameters:
            voltage (float): Current voltage level.

        Returns:
            float: Power acceptance level based on the voltage.
        """
        if voltage >= self.voltage_max:
            return self.feed_in_min
        power_acceptance = self.feed_in_min + (self.voltage_max - voltage) * self.power_for_voltage
        return power_acceptance

    def get_feed_to(self) -> float:
        """
        Returns the current amount of power being fed into the grid.

        Returns:
            float: Power currently fed into the grid.
        """
        return self.current_feed_to_grid

    def get_taken_from(self) -> float:
        """
        Returns the current amount of power being drawn from the grid.

        Returns:
            float: Power currently drawn from the grid.
        """
        return self.current_taken_from_grid
