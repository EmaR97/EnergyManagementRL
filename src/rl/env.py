import numpy as np
from gymnasium import spaces
import gymnasium as gym
from ..simulation import InverterSim, week


class InverterEnv(gym.Env):
    state_names = [
        "prod_energy",
        "cons_energy",
        "batt_charge_rate",
        "batt_discharge_rate",
        "batt_stored",
        "grid_feed_to",
        "grid_taken_from",
        "prod_energy_last_24h",
        "cons_energy_last_24h",
        "prod_energy_next_24h",
        "cons_energy_next_24h",
        "prod_energy_following_24h",
        "cons_energy_following_24h",
        "cos_time",
        "sin_time",
        'last_action',
        'reward',
        'reward_energy_sold',
        'penalty_energy_purchase',
        'penalty_battery_wear'
    ]

    def __init__(
            self,
            inverter_sim: InverterSim,
            max_steps: int = week,
            normalize: bool = False,
    ):
        super(InverterEnv, self).__init__()

        self.inverter_sim = inverter_sim
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(15,), dtype=np.float64)
        self.state = np.zeros(15)
        self.last_action = 0
        self.reward_energy_sold = 0
        self.penalty_energy_purchase = 0
        self.penalty_battery_wear = 0
        self.reward = 0

        # Precompute normalization factors
        if normalize:
            self.inv_factors = np.array([
                1.0 / self.inverter_sim.prod_sim.max_step,
                1.0 / self.inverter_sim.cons_sim.max_step,
                1.0 / self.inverter_sim.batt_sim.max_charge_rate,
                1.0 / self.inverter_sim.batt_sim.max_discharge_rate,
                1.0 / self.inverter_sim.batt_sim.capacity,
                1.0 / self.inverter_sim.grid_sim.feed_in_max_known * 2,
                1.0 / self.inverter_sim.grid_sim.max_taken_from,
                1.0 / self.inverter_sim.prod_sim.max_24h,
                1.0 / self.inverter_sim.cons_sim.max_24h,
                1.0 / self.inverter_sim.prod_sim.max_24h,
                1.0 / self.inverter_sim.cons_sim.max_24h,
                1.0 / self.inverter_sim.prod_sim.max_24h,
                1.0 / self.inverter_sim.cons_sim.max_24h
            ])
        else:
            self.inv_factors = np.array([1 / 1000] * 13)

    def reset(self, seed=0, **kwargs):
        self.state = np.zeros(15)
        self.current_step = 0
        self.inverter_sim.reset()
        self.last_action = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        self.last_action = action
        self.inverter_sim.step(action)
        self._update_state()
        reward = self.set_reward()
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
        self.state = np.concatenate([scaled_values, self.inverter_sim.get_timestep()])

    def set_reward(self):
        self.set_reward_energy_sold()
        self.set_penalty_energy_purchase()
        self.set_penalty_battery_wear()
        self.reward = self.reward_energy_sold - self.penalty_energy_purchase - self.penalty_battery_wear

        return self.reward

    def set_reward_energy_sold(self):
        self.reward_energy_sold = self.inverter_sim.grid_sim.get_feed_to() * self.inverter_sim.grid_sim.energy_price_sell

    def set_penalty_energy_purchase(self):
        energy_deficit = self.inverter_sim.cons_sim.get_energy() - (
                self.inverter_sim.prod_sim.get_energy() + self.inverter_sim.batt_sim.max_discharge_rate)
        avoidable_grid_purchase = max(0, self.inverter_sim.grid_sim.get_taken_from() - max(0., energy_deficit))
        self.penalty_energy_purchase = avoidable_grid_purchase * self.inverter_sim.grid_sim.energy_price_buy

    def set_penalty_battery_wear(self):
        if self.inverter_sim.batt_sim.battery_wear_rate:
            self.penalty_battery_wear = (
                    (self.inverter_sim.batt_sim.get_charge_rate() + self.inverter_sim.batt_sim.get_discharge_rate()) *
                    self.inverter_sim.batt_sim.battery_wear_rate)
        else:
            self.penalty_battery_wear = 0

    def get_state_dict(self):
        """Helper function to get state as a dictionary with names."""
        return dict(
            zip(self.state_names,
                self.state.tolist() + [self.last_action, self.reward, self.reward_energy_sold,
                                       self.penalty_energy_purchase, self.penalty_battery_wear]))


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

