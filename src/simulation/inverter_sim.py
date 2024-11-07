import numpy as np
import pandas as pd

from .battery_sim import BatterySim
from .energy_sim import EnergySim
from .grid_sim import GridSim


class InverterSim:

    def __init__(
            self,
            prod_sim: EnergySim,
            cons_sim: EnergySim,
            batt_sim: BatterySim,
            grid_sim: GridSim,
            timestamps: pd.Series,
    ):
        self.prod_sim = prod_sim
        self.cons_sim = cons_sim
        self.batt_sim = batt_sim
        self.grid_sim = grid_sim
        self.timestamps = timestamps
        self.current_step = 0

        # Precompute sine and cosine values for each timestep
        self.precomputed_time_steps = self._precompute_time_steps()

    def _precompute_time_steps(self):
        time_steps = []
        for timestamp in self.timestamps:
            timestep = (timestamp.hour * 12 + timestamp.minute / 5) / 288 * 2 * np.pi
            sin_cos = (np.array([np.sin(timestep), np.cos(timestep)]) / 2) + 0.5
            time_steps.append(sin_cos)
        return np.array(time_steps)

    def get_timestep(self):
        # Retrieve the precomputed value for the current step
        return self.precomputed_time_steps[self.current_step]

    def reset(
            self,
    ):
        self.current_step = 0
        self.batt_sim.reset()
        self.grid_sim.reset()
        self.prod_sim.reset()
        self.cons_sim.reset()

    def step(
            self,
            action
    ):
        self.current_step += 1
        # Calculate energy balance (production - consumption)
        energy_balance = self.prod_sim.step() - self.cons_sim.step()

        if action == 1:  # Max-self-consumption (Mode A)
            energy_balance = self._manage_energy_mode_a(energy_balance)
        else:  # Full-feed-to-grid (Mode B)
            energy_balance = self._manage_energy_mode_b(energy_balance)

        return energy_balance

    def _manage_energy_mode_a(
            self,
            energy_balance
    ):
        """
        Manage energy in Mode A (max-self-consumption): prioritize balance, battery, then grid.
        """

        return self.grid_sim.step(self.batt_sim.step(energy_balance))

    def _manage_energy_mode_b(
            self,
            energy_balance
    ):
        """
        Manage energy in Mode B (full-feed-to-grid): prioritize balance, grid, then battery.
        """

        grid_acceptance = self.grid_sim.get_grid_acceptance()
        energy_balance -= grid_acceptance
        energy_balance_after_batt = self.batt_sim.step(energy_balance)  # Battery handles excess or deficit
        if energy_balance_after_batt >= 0:
            energy_balance_after_grid = self.grid_sim.step(grid_acceptance)
        else:
            energy_balance_after_batt = grid_acceptance + energy_balance_after_batt
            energy_balance_after_grid = self.grid_sim.step(energy_balance_after_batt)
        return energy_balance_after_grid
