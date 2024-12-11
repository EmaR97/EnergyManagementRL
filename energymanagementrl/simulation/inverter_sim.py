import numpy as np
import pandas as pd

from .base_sim import BaseSim
from .battery_sim import BatterySim
from .consumption_sim import ConsumptionSim
from .energy_sim import EnergySim
from .grid_sim import GridSim
from .production_sim import ProductionSim

MODE_A = 1
MODE_B = 0


class InverterSim(BaseSim):
    """
    InverterSim models the operation of an energy inverter system with energy production,
    consumption, battery storage, and grid interaction. The simulation offers two operation modes:
    Mode A (Max-Self-Consumption) and Mode B (Full-Feed-to-Grid). It uses precomputed sine and cosine
    values for each timestamp to simulate time-dependent behavior.

    Attributes:
        prod_sim (EnergySim): Simulation of energy production.
        cons_sim (EnergySim): Simulation of energy consumption.
        batt_sim (BatterySim): Battery simulation for energy storage.
        grid_sim (GridSim): Simulation of grid interaction.
        timestamps (pd.Series): Series of timestamps for each simulation step.
        precomputed_time_steps (np.ndarray): Array of precomputed sine and cosine values for each timestep.
    """

    def __init__(
            self,
            prod_sim: ProductionSim,
            cons_sim: ConsumptionSim,
            batt_sim: BatterySim,
            grid_sim: GridSim,
            timestamps: pd.Series,
            seed=None,
    ):
        """
        Initializes the InverterSim with energy production, consumption, battery, grid simulations, and timestamps.

        Parameters:
            prod_sim (EnergySim): Instance for simulating energy production.
            cons_sim (EnergySim): Instance for simulating energy consumption.
            batt_sim (BatterySim): Instance for managing battery operations.
            grid_sim (GridSim): Instance for handling grid interactions.
            timestamps (pd.Series): Series of timestamps corresponding to each simulation step.
        """
        super().__init__(seed)
        self.energy_balance = 0
        self.prod_sim = prod_sim
        self.cons_sim = cons_sim
        self.batt_sim = batt_sim
        self.grid_sim = grid_sim
        self.timestamps = timestamps

        # Precompute sine and cosine values for each timestep
        self.precomputed_time_steps = self._precompute_time_steps()

    def _precompute_time_steps(self) -> np.ndarray:
        """
        Precomputes normalized sine and cosine values for each timestamp to simulate time-dependent behavior.

        Returns:
            np.ndarray: Array of precomputed sine and cosine values for each timestamp.
        """
        time_steps = []
        for timestamp in self.timestamps:
            timestep = (timestamp.hour * 12 + timestamp.minute / 5) / 288 * 2 * np.pi
            sin_cos = (np.array([np.sin(timestep), np.cos(timestep)]) / 2) + 0.5
            time_steps.append(sin_cos)
        return np.array(time_steps)

    def reset(self, seed=None) -> None:
        """
        Resets the simulation state to the starting conditions, resetting all components.
        """
        super().reset(seed)
        self.batt_sim.reset(seed)
        self.grid_sim.reset(seed)
        self.prod_sim.reset(seed)
        self.cons_sim.reset(seed)

    def step(self, action: int, **inputs) -> None:
        """
        Advances the simulation by one step and adjusts energy balance based on the chosen operation mode.

        Parameters:
            action (int): Operation mode (1 for Max-Self-Consumption, otherwise Full-Feed-to-Grid).

        Returns:
            int: Remaining energy balance after the step.
        """
        super().step(**inputs)
        energy_balance = self.prod_sim.step() - self.cons_sim.step()  # Net energy (production - consumption)
        if action == MODE_A:  # Mode A (Max-Self-Consumption)
            energy_balance = self._manage_energy_mode_a(energy_balance)
        elif action == MODE_B:  # Mode B (Full-Feed-to-Grid)
            energy_balance = self._manage_energy_mode_b(energy_balance)
        self.energy_balance = energy_balance

    def _manage_energy_mode_a(self, energy_balance: int):
        """
        Manages energy flow in Mode A (Max-Self-Consumption), prioritizing:
        1. Balancing production and consumption.
        2. Charging/discharging the battery.
        3. Feeding excess or drawing deficit from the grid.

        Parameters:
            energy_balance (int): Current net energy balance.

        Returns:
            int: Adjusted energy balance after managing battery and grid interaction.
        """
        return self.grid_sim.step(self.batt_sim.step(energy_balance))

    def _manage_energy_mode_b(self, energy_balance: int):
        """
        Manages energy flow in Mode B (Full-Feed-to-Grid), prioritizing:
        1. Balancing production and consumption.
        2. Feeding as much as possible to the grid.
        3. Storing any remaining surplus or deficit in the battery.

        Parameters:
            energy_balance (int): Current net energy balance.

        Returns:
            int: Adjusted energy balance after managing grid feed-in and battery usage.
        """
        # Consider grid as a load in Mode B
        grid_acceptance = self.grid_sim.get_grid_acceptance()
        energy_balance -= grid_acceptance  # Feed as much as possible to the grid
        energy_balance_after_batt = self.batt_sim.step(energy_balance)  # Battery handles excess or deficit

        if energy_balance_after_batt >= 0:  # If additional load can be satisfied
            return self.grid_sim.step(grid_acceptance)
        else:  # If load can't be fully satisfied, adjust balance to provide available energy
            adjusted_balance = grid_acceptance + energy_balance_after_batt
            return self.grid_sim.step(adjusted_balance)

    def get_state(self):
        state = {
            'prod_sim': self.prod_sim.get_state(),
            'cons_sim': self.cons_sim.get_state(),
            'batt_sim': self.batt_sim.get_state(),
            'grid_sim': self.grid_sim.get_state()
        }
        state['prod_sim']['energy'] -= self.energy_balance
        return state
