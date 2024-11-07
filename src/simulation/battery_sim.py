from .utils import min5


class BatterySim:
    """
    BatterySim models a battery with basic charging, discharging, and efficiency properties.
    It provides methods to simulate charging/discharging steps based on an energy balance input
    and calculates the allowed charge/discharge rates for a given time interval.

    Attributes:
        max_charge_rate (int): Maximum charge power in kW.
        max_discharge_rate (int): Maximum discharge power in kW.
        capacity (int): Total battery capacity in kWh.
        battery_wear_rate (float): Rate at which battery wears down, unitless.
        efficiency (float): Efficiency of charging and discharging (0 to 1), defaults to 0.95.
        starting_charge (float): Initial energy stored in the battery in kWh.
        current_charge (float): Current energy stored in the battery in kWh.
        current_charge_rate (float): Power charged in the current step in kW.
        current_discharge_rate (float): Power discharged in the current step in kW.
    """

    def __init__(self, max_charge_rate: int, max_discharge_rate: int, capacity: int, battery_wear_rate: float,
                 efficiency: float = 0.95, current_charge: float = 0):
        """
        Initializes the BatterySim instance with given parameters.

        Parameters:
            max_charge_rate (int): Maximum charge rate (kW).
            max_discharge_rate (int): Maximum discharge rate (kW).
            capacity (int): Battery's total energy capacity in kWh.
            battery_wear_rate (float): Wear rate of the battery over time (unitless).
            efficiency (float): Efficiency of charge/discharge processes (default: 0.95).
            current_charge (float): Initial stored energy in the battery (kWh).
        """
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.capacity = capacity
        self.battery_wear_rate = battery_wear_rate
        self.efficiency = efficiency
        self.starting_charge = current_charge
        self.current_charge = current_charge
        self.current_charge_rate = 0.0
        self.current_discharge_rate = 0.0

    def reset(self):
        """
        Resets the battery to its starting charge and clears charge and discharge rates.
        """
        self.current_charge = int(self.starting_charge)
        self.current_charge_rate = 0
        self.current_discharge_rate = 0

    def step(self, energy_balance: int) -> int:
        """
        Simulates a single step of battery behavior based on an energy balance input.

        Parameters:
            energy_balance (int): The net energy input/output for the step (positive for charge, negative for discharge).

        Returns:
            int: Remaining energy balance after charge or discharge has been applied.
        """
        self.current_charge_rate = 0.0
        self.current_discharge_rate = 0.0

        if energy_balance < 0:
            self.current_discharge_rate = min(float(-energy_balance), self.get_allowed_discharge())
            energy_balance += self.current_discharge_rate
            self.current_charge = max(
                0.0,
                self.current_charge - (self.current_discharge_rate / self.efficiency)
            )
        elif energy_balance > 0:
            self.current_charge_rate = min(float(energy_balance), self.get_allowed_charge())
            energy_balance -= self.current_charge_rate
            self.current_charge = min(
                float(self.capacity),
                self.current_charge + (self.current_charge_rate * self.efficiency)
            )

        return int(energy_balance)

    def get_allowed_charge(self) -> float:
        """
        Calculates the maximum allowed charge rate for the current state.

        Returns:
            float: Allowed charge rate in kW based on maximum charge rate, remaining capacity, and efficiency.
        """
        return min(
            self.max_charge_rate * min5,
            (self.capacity - self.current_charge) / self.efficiency
        )

    def get_allowed_discharge(self) -> float:
        """
        Calculates the maximum allowed discharge rate for the current state.

        Returns:
            float: Allowed discharge rate in kW based on maximum discharge rate, current charge, and efficiency.
        """
        return min(
            self.max_discharge_rate * min5,
            self.current_charge * self.efficiency
        )

    def get_charge_rate(self) -> int:
        """
        Returns the charge rate in the current step.

        Returns:
            int: Charge rate in kW.
        """
        return int(self.current_charge_rate)

    def get_discharge_rate(self) -> int:
        """
        Returns the discharge rate in the current step.

        Returns:
            int: Discharge rate in kW.
        """
        return int(self.current_discharge_rate)

    def get_stored(self) -> int:
        """
        Returns the current stored energy in the battery.

        Returns:
            int: Stored energy in kWh.
        """
        return int(self.current_charge)
