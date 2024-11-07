from .utils import min5


class BatterySim:
    def __init__(self, max_charge_rate: int, max_discharge_rate: int, capacity: int, battery_wear_rate: float,
                 efficiency: float = 0.95, current_charge: float = 0):
        self.max_charge_rate = max_charge_rate  # kW, max power for charging
        self.max_discharge_rate = max_discharge_rate  # kW, max power for discharging
        self.capacity = capacity  # kWh, max stored energy
        self.battery_wear_rate = battery_wear_rate  # Wear rate (unit-less)
        self.efficiency = efficiency  # Charging/discharging efficiency
        self.starting_charge = current_charge
        self.current_charge = current_charge  # kWh, current stored energy in the battery
        self.current_charge_rate = 0.0  # kW, power charged in this step
        self.current_discharge_rate = 0.0  # kW, power discharged in this step

    def reset(self, seed=0):
        self.current_charge: int = int(self.starting_charge)
        self.current_charge_rate: int = 0
        self.current_discharge_rate: int = 0

    def step(self, energy_balance: int) -> int:
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

    def get_allowed_charge(self):
        return min(
            self.max_charge_rate * min5,
            (self.capacity - self.current_charge) / self.efficiency
        )

    def get_allowed_discharge(self):
        return min(
            self.max_discharge_rate * min5,
            self.current_charge * self.efficiency
        )

    def get_charge_rate(self) -> int:
        return int(self.current_charge_rate)

    def get_discharge_rate(self) -> int:
        return int(self.current_discharge_rate)

    def get_stored(self) -> int:
        return int(self.current_charge)
