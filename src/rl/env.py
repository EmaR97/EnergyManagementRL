import numpy as np
from gymnasium import spaces
import gymnasium as gym
from ..simulation import InverterSim, week


class InverterEnv(gym.Env):
    """
    InverterEnv is a reinforcement learning environment for simulating an inverter's operation.
    It provides a state space for training agents to optimize energy usage, battery wear, and grid interaction.

    Attributes:
        state_names (list): List of state variable names for tracking various energy and time-related metrics.
        inverter_sim (InverterSim): Instance of the InverterSim simulation model.
        max_steps (int): Maximum number of steps per episode.
        current_step (int): Index for tracking the current simulation step.
        action_space (spaces.Discrete): Action space for the environment (binary: 0 for grid-feeding, 1 for self-consumption).
        observation_space (spaces.Box): Observation space defining the range of possible state values.
        state (np.ndarray): Array storing the current state representation.
        last_action (int): Stores the last action taken by the agent.
        reward_energy_sold (float): Reward component from energy sold to the grid.
        penalty_energy_purchase (float): Penalty for energy bought from the grid.
        penalty_battery_wear (float): Penalty for battery usage affecting wear.
        reward (float): Total reward for the current step.
        inv_factors (np.ndarray): Scaling factors for normalizing state values.
    """

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
        """
        Initializes the environment for inverter simulation.

        Parameters:
            inverter_sim (InverterSim): Simulation model of an inverter system.
            max_steps (int): Maximum steps allowed in an episode (default: one week).
            normalize (bool): Whether to normalize state values.
        """
        super(InverterEnv, self).__init__()

        self.inverter_sim = inverter_sim
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = spaces.Discrete(2)  # Two actions: grid-feeding or self-consumption
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
        """
        Resets the environment state, simulation, and metrics.

        Parameters:
            seed (int): Optional random seed.

        Returns:
            tuple: Initial state and an empty info dictionary.
        """
        self.state = np.zeros(15)
        self.current_step = 0
        self.inverter_sim.reset()
        self.last_action = 0
        return self.state, {}

    def step(self, action: int):
        """
        Executes one simulation step based on the agent's action.

        Parameters:
            action (int): Action to take (0 for grid-feeding, 1 for self-consumption).

        Returns:
            tuple: Updated state, reward, done flag, truncated flag, and info dictionary.
        """
        self.current_step += 1
        self.last_action = action
        self.inverter_sim.step(action)
        self._update_state()
        reward = self.set_reward()
        done = self.current_step > self.max_steps
        truncated = False

        return self.state, reward, done, truncated, {}

    def _update_state(self):
        """
        Updates the state by retrieving current values from the inverter simulation
        and applying normalization if specified.
        """
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

        # Apply scaling factors for normalization
        scaled_values = values * self.inv_factors
        # Append precomputed sine and cosine values
        self.state = np.concatenate([scaled_values, self.inverter_sim.get_timestep()])

    def set_reward(self) -> float:
        """
        Calculates the reward for the current step, considering energy sales, grid purchase penalties,
        and battery wear penalties.

        Returns:
            float: The total reward for the current step.
        """
        self.set_reward_energy_sold()
        self.set_penalty_energy_purchase()
        self.set_penalty_battery_wear()
        self.reward = self.reward_energy_sold - self.penalty_energy_purchase - self.penalty_battery_wear

        return self.reward

    def set_reward_energy_sold(self) -> None:
        """
        Sets the reward component from energy sold to the grid.
        """
        self.reward_energy_sold = self.inverter_sim.grid_sim.get_feed_to() * self.inverter_sim.grid_sim.energy_price_sell

    def set_penalty_energy_purchase(self) -> None:
        """
        Sets the penalty for energy purchased from the grid, accounting for avoidable purchases.
        """
        energy_deficit = self.inverter_sim.cons_sim.get_energy() - (
                self.inverter_sim.prod_sim.get_energy() + self.inverter_sim.batt_sim.max_discharge_rate)
        avoidable_grid_purchase = max(.0, self.inverter_sim.grid_sim.get_taken_from() - max(0., energy_deficit))
        self.penalty_energy_purchase = avoidable_grid_purchase * self.inverter_sim.grid_sim.energy_price_buy

    def set_penalty_battery_wear(self) -> None:
        """
        Sets the penalty for battery wear based on charge and discharge rates.
        """
        if self.inverter_sim.batt_sim.battery_wear_rate:
            self.penalty_battery_wear = (
                    (self.inverter_sim.batt_sim.get_charge_rate() + self.inverter_sim.batt_sim.get_discharge_rate()) *
                    self.inverter_sim.batt_sim.battery_wear_rate)
        else:
            self.penalty_battery_wear = 0

    def get_state_dict(self) -> dict:
        """
        Returns the current state as a dictionary with named state variables for better readability.

        Returns:
            dict: Dictionary with state variable names and corresponding values.
        """
        return dict(
            zip(self.state_names,
                self.state.tolist() + [self.last_action, self.reward, self.reward_energy_sold,
                                       self.penalty_energy_purchase, self.penalty_battery_wear])
        )


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
