import numpy as np
from gymnasium import spaces
import gymnasium as gym
from ..simulation import InverterSim, week


class InverterEnv(gym.Env):

    def __init__(
            self,
            inverter_sim: InverterSim,
            max_steps: int = week,
    ):
        super(InverterEnv, self).__init__()

        self.inverter_sim = inverter_sim
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(low=0, high=1000, shape=(15,), dtype=np.float64)

        self.state = np.zeros(15)

        # Precompute sine and cosine values for each timestep

        # self.inv_factors = np.array([
        #     1.0 / self.inverter_sim.prod_sim.max_step,
        #     1.0 / self.inverter_sim.cons_sim.max_step,
        #     1.0 / self.inverter_sim.batt_sim.max_charge_rate,
        #     1.0 / self.inverter_sim.batt_sim.max_discharge_rate,
        #     1.0 / self.inverter_sim.batt_sim.capacity,
        #     1.0 / self.inverter_sim.grid_sim.feed_in_max_known * 2,
        #     1.0 / self.inverter_sim.grid_sim.max_taken_from,
        #     1.0 / self.inverter_sim.prod_sim.max_24h,
        #     1.0 / self.inverter_sim.cons_sim.max_24h,
        #     1.0 / self.inverter_sim.prod_sim.max_24h,
        #     1.0 / self.inverter_sim.cons_sim.max_24h,
        #     1.0 / self.inverter_sim.prod_sim.max_24h,
        #     1.0 / self.inverter_sim.cons_sim.max_24h
        # ])
        self.inv_factors = np.array([1 / 1000] * 13)

    def reset(
            self,
            seed=0,
            **kwargs
    ):
        self.state = np.zeros(15)
        self.current_step = 0
        self.inverter_sim.reset()
        return self.state, {}

    def step(
            self,
            action
    ):
        self.current_step += 1
        self.inverter_sim.step(action)
        self._update_state()
        reward, _, _, _ = self.calc_reward()
        done = False if self.current_step <= self.max_steps else True
        truncated = False

        return self.state, reward, done, truncated, {}

    def _update_state(self):
        values = np.array([
            self.inverter_sim.prod_sim.get_energy(),
            self.inverter_sim.cons_sim.get_energy(),
            self.inverter_sim.batt_sim.get_charge_rate(),
            self.inverter_sim.batt_sim.get_discharge_rate(),
            self.inverter_sim.batt_sim.get_stored(),
            self.inverter_sim.grid_sim.get_feed_to(),
            self.inverter_sim.grid_sim.get_taken_from(),
            self.inverter_sim.prod_sim.get_energy_last_24h(),
            self.inverter_sim.cons_sim.get_energy_last_24h(),
            self.inverter_sim.prod_sim.get_energy_next_24h(),
            self.inverter_sim.cons_sim.get_energy_next_24h(),
            self.inverter_sim.prod_sim.get_energy_24h_following_next_24h(),
            self.inverter_sim.cons_sim.get_energy_24h_following_next_24h()
        ])

        # Perform element-wise multiplication
        scaled_values = values * self.inv_factors

        # Include additional timestep values if needed
        self.state = np.concatenate([scaled_values, self.inverter_sim._get_timestep()])

    def calc_reward(
            self
    ):
        feed_in_reward = self.inverter_sim.grid_sim.get_feed_to() * self.inverter_sim.grid_sim.energy_price_sell

        energy_deficit = self.inverter_sim.cons_sim.get_energy() - (
                self.inverter_sim.prod_sim.get_energy() + self.inverter_sim.batt_sim.max_discharge_rate)

        avoidable_grid_purchase = max(0, self.inverter_sim.grid_sim.get_taken_from() - max(0., energy_deficit))
        purchase_penalty = avoidable_grid_purchase * self.inverter_sim.grid_sim.energy_price_buy

        battery_wear_penalty = ((
                                        self.inverter_sim.batt_sim.get_charge_rate() + self.inverter_sim.batt_sim.get_discharge_rate()) *
                                self.inverter_sim.batt_sim.battery_wear_rate)

        total_reward = feed_in_reward - purchase_penalty - battery_wear_penalty

        return total_reward, feed_in_reward, purchase_penalty, battery_wear_penalty


class InverterEnvSimple(InverterEnv):

    def __init__(
            self,
            inverter_sim: InverterSim,
            max_steps: int = week,

    ):
        super().__init__(
            inverter_sim,
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
        self.state, reward, done, truncated, _ = super().step(action)
        #         self.state= np.array([np.sin(2 * np.pi *self.state[-1]),np.cos(2 * np.pi *self.state[-1])])
        self.state = np.array([self.state[4], *self.state[-6:]])
        return self.state, reward, done, truncated, {}
