import numpy as np
import pandas as pd

from .battery_sim import BatterySim
from .energy_sim import EnergySim
from .grid_sim import GridSim
from .utils import week
import gymnasium as gym
from gymnasium import spaces


class InverterEnv(gym.Env):

    def __init__(
            self,
            prod_sim: EnergySim,
            cons_sim: EnergySim,
            batt_sim: BatterySim,
            grid_sim: GridSim,
            timestamps: pd.Series,
            max_steps: int = week,
    ):
        super(InverterEnv, self).__init__()

        self.prod_sim = prod_sim
        self.cons_sim = cons_sim
        self.batt_sim = batt_sim
        self.grid_sim = grid_sim
        self.timestamps = timestamps
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(low=0, high=1, shape=(15,), dtype=np.float64)

        self.state = np.zeros(15)
        self.current_step = 0

        # Precompute sine and cosine values for each timestep
        self.precomputed_timesteps = self._precompute_timesteps()

        self.inv_factors = np.array([
            1.0 / self.prod_sim.max_step,
            1.0 / self.cons_sim.max_step,
            1.0 / self.batt_sim.max_charge_rate,
            1.0 / self.batt_sim.max_discharge_rate,
            1.0 / self.batt_sim.capacity,
            1.0 / self.grid_sim.feed_in_max_known * 2,
            1.0 / self.grid_sim.max_taken_from,
            1.0 / self.prod_sim.max_24h,
            1.0 / self.cons_sim.max_24h,
            1.0 / self.prod_sim.max_24h,
            1.0 / self.cons_sim.max_24h,
            1.0 / self.prod_sim.max_24h,
            1.0 / self.cons_sim.max_24h
        ])

    def _precompute_timesteps(self):
        timesteps = []
        for timestamp in self.timestamps:
            timestep = (timestamp.hour * 12 + timestamp.minute / 5) / 288 * 2 * np.pi
            sin_cos = (np.array([np.sin(timestep), np.cos(timestep)]) / 2) + 0.5
            timesteps.append(sin_cos)
        return np.array(timesteps)

    def _get_timestep(self):
        # Retrieve the precomputed value for the current step
        return self.precomputed_timesteps[self.current_step]

    def reset(
            self,
            seed=0,
            **kwargs
    ):
        self.state = np.zeros(15)
        self.current_step = 0
        self.batt_sim.reset()
        self.grid_sim.reset()
        self.prod_sim.reset()
        self.cons_sim.reset()
        return self.state, {}

    def step(
            self,
            action
    ):
        self.current_step += 1

        # Calculate energy balance (production - consumption)
        energy_balance = self.prod_sim.step() - self.cons_sim.step()
        # logging.info(f'Step {self.current_step}: Calculated energy balance = {energy_balance:.2f}')

        # Apply action logic (energy management mode)
        if action == 1:  # Max-self-consumption (Mode A)
            _ = self._manage_energy_mode_a(energy_balance)
        else:  # Full-feed-to-grid (Mode B)
            _ = self._manage_energy_mode_b(energy_balance)

        self._update_state()
        reward, _, _, _ = self.calc_reward()
        done = False if self.current_step <= self.max_steps else True
        truncated = False

        return self.state, reward, done, truncated, {}

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
        # logging.info(
        #     f'Mode A: Balance after battery = {energy_balance_after_batt:.2f}, '
        #     f'after grid = {energy_balance_after_grid:.2f}')
        return energy_balance_after_grid

    def _update_state(self):
        values = np.array([
            self.prod_sim.get_energy(),
            self.cons_sim.get_energy(),
            self.batt_sim.get_charge_rate(),
            self.batt_sim.get_discharge_rate(),
            self.batt_sim.get_stored(),
            self.grid_sim.get_feed_to(),
            self.grid_sim.get_taken_from(),
            self.prod_sim.get_energy_last_24h(),
            self.cons_sim.get_energy_last_24h(),
            self.prod_sim.get_energy_next_24h(),
            self.cons_sim.get_energy_next_24h(),
            self.prod_sim.get_energy_24h_following_next_24h(),
            self.cons_sim.get_energy_24h_following_next_24h()
        ])

        # Perform element-wise multiplication
        scaled_values = values * self.inv_factors

        # Include additional timestep values if needed
        self.state = np.concatenate([scaled_values, self._get_timestep()])

    def calc_reward(
            self
    ):
        feed_in_reward = self.grid_sim.get_feed_to() * self.grid_sim.energy_price_sell

        energy_deficit = self.cons_sim.get_energy() - (self.prod_sim.get_energy() + self.batt_sim.max_discharge_rate)

        avoidable_grid_purchase = max(0, self.grid_sim.get_taken_from() - max(0., energy_deficit))
        purchase_penalty = avoidable_grid_purchase * self.grid_sim.energy_price_buy

        battery_wear_penalty = ((
                                        self.batt_sim.get_charge_rate() + self.batt_sim.get_discharge_rate()) *
                                self.batt_sim.battery_wear_rate)

        total_reward = feed_in_reward - purchase_penalty - battery_wear_penalty

        return total_reward, feed_in_reward, purchase_penalty, battery_wear_penalty


class InverterEnvSimple(InverterEnv):

    def __init__(
            self,
            prod_sim: EnergySim,
            cons_sim: EnergySim,
            batt_sim: BatterySim,
            grid_sim: GridSim,
            timestamps: pd.Series,
            max_steps: int = week,

    ):
        super().__init__(
            prod_sim,
            cons_sim,
            batt_sim,
            grid_sim,
            timestamps,
            max_steps

        )

        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float64)

        # Initialize internal states
        self.state = np.zeros(7)

    def reset(
            self,
            seed=0,
            **kwargs
    ):
        """Resets the environment for a new episode.
        """
        super().reset()
        self.state = np.zeros(7)

        return self.state, {}

    def step(
            self,
            action
    ):
        """
        Executes one time step in the environment based on the action taken.

        Args:
            action (int): 0 (Mode B) for full-feed-to-grid or 1 (Mode A) for max-self-consumption.

        Returns:
            tuple: observation (state), reward, done (bool), and additional info (empty dict).
        """
        self.state, reward, done, truncated, _ = super().step(action)
        #         self.state= np.array([np.sin(2 * np.pi *self.state[-1]),np.cos(2 * np.pi *self.state[-1])])
        self.state = np.array([self.state[4], *self.state[-6:]])
        return self.state, reward, done, truncated, {}

