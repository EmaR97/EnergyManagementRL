import numpy as np
from gymnasium import spaces
import gymnasium as gym

from .utils import extract_values_gen, flatten_dict
from ..simulation import InverterSim, week


class InverterEnv(gym.Env):
    """
    InverterEnv is a reinforcement learning environment for simulating an inverter's operation.
    It provides a state space for training agents to optimize energy usage, battery wear, and grid interaction.

    Attributes:
        inverter_sim (InverterSim): Instance of the InverterSim simulation model.
        _max_steps (int): Maximum number of steps per episode.
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

    def __init__(
            self,
            inverter_sim: InverterSim,
            max_steps: int = week,
    ):
        """
        Initializes the environment for inverter simulation.

        Parameters:
            inverter_sim (InverterSim): Simulation model of an inverter system.
            max_steps (int): Maximum steps allowed in an episode (default: one week).
        """
        super(InverterEnv, self).__init__()
        self.inverter_sim = inverter_sim
        self.state_size = len(list(extract_values_gen(self.inverter_sim.get_state())))
        self._max_steps = None
        self.set_max_steps(max_steps)
        self.current_step = 0
        self.action_space = spaces.Discrete(2)  # Two actions: grid-feeding or self-consumption
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(self.state_size,), dtype=np.float64)
        self.state = np.zeros(self.state_size)
        self.last_action = 0
        self.reward_energy_sold = 0
        self.penalty_energy_purchase = 0
        self.penalty_battery_wear = 0
        self.reward = 0
        self.shuffle = 0
        # Precompute normalization factors
        self.inv_factors = np.array([1 / 1000] * self.state_size)

    def set_max_steps(self, max_steps):
        self.inverter_sim.check_max_steps(max_steps)
        self._max_steps = max_steps

    def reset(self, seed=0, shuffle=None, **kwargs):
        self.state = np.zeros(self.state_size)
        self.current_step = 0
        if shuffle is None:
            shuffle = self.shuffle
        self.inverter_sim.reset(seed if seed != 0 else None, shuffle=shuffle, **kwargs)
        self.last_action = 0
        self.inverter_sim.random_start(self._max_steps)

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
        done = self.current_step > self._max_steps
        truncated = False

        return self.state, reward, done, truncated, {}

    def _update_state(self):
        """
        Updates the state by retrieving current values from the inverter simulation
        and applying normalization if specified.
        """
        values = np.array(list(extract_values_gen(self.inverter_sim.get_state())))
        # Apply scaling factors for normalization
        self.state = values * self.inv_factors

    def get_state_dict(self):
        state = dict(flatten_dict(self.inverter_sim.get_state()))
        state = {key: value / 1000 for key, value in state.items()}
        state['reward_energy_sold'] = self.reward_energy_sold
        state['penalty_energy_purchase'] = self.penalty_energy_purchase
        state['penalty_battery_wear'] = self.penalty_battery_wear
        state['reward'] = self.reward
        return state

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


class InverterEnvBatteryMgmt(InverterEnv):
    """
    A custom environment for managing inverter behavior with a focus on optimized battery usage.

    This class extends `InverterEnv` by introducing a customized reward function that encourages
    efficient battery charging and discharging behaviors based on solar production forecasts and
    the battery's state of charge (SOC). The reward logic aims to:
    - Penalize premature discharging during early night hours when solar production is forecasted
      to remain zero for the next 1 to 3 hours.
    - Encourage maintaining an optimal SOC during the last hour of daily solar production to
      ensure sufficient reserves for nighttime consumption.

    Parameters:
        inverter_sim (InverterSim): The inverter simulation instance.
        max_steps (int): The maximum number of steps in an episode.
        reward_near_full (float): The reward multiplier for maintaining near-full SOC.
        penalty_early_discharge (float): The penalty multiplier for premature discharging.
    """

    def __init__(
            self,
            inverter_sim: InverterSim,
            max_steps: int = week,
            reward_near_full: float = .01,
            penalty_below_night_reserve: float = .01,
            penalty_below_min_reserve: float = .01,
            penalty_early_discharge: float = 1,
            night_reserver: float = .9,
            near_full: float = .98,
            min_reserve: float = .1,
    ):
        super().__init__(
            inverter_sim,
            max_steps
        )
        self.reward_near_full = reward_near_full
        self.penalty_below_night_reserve = penalty_below_night_reserve
        self.penalty_early_discharge = penalty_early_discharge
        self.penalty_below_min_reserve = penalty_below_min_reserve
        self.night_reserver = night_reserver
        self.near_full = near_full
        self.min_reserve = min_reserve

    def set_reward(self) -> float:
        """
        Calculates the reward for the current step based on battery usage and forecasted solar production.
        This method adjusts the base reward to encourage efficient energy management, including
        optimal battery discharging and charging behaviors.

        Specifically:
        - Penalizes premature battery discharging during early night hours (when no solar production
         is forecasted for the next 1 to 3 hours), while allowing discharging closer to dawn.
        - Rewards or penalizes the battery's state of charge (SOC) during the last hour of daily solar
         production, encouraging sufficient reserves for nighttime consumption.

        Returns:
           float: The total adjusted reward for the current step.
        """
        super().set_reward()

        # Penalize discharging the battery during the early night, defined as hours with no forecasted production,
        # while allowing discharging closer to dawn (if the sun is expected to rise within the next 3 hours).
        # `self.state[1:4]` represents the forecasted solar production for the next 1 to 3 hours.
        # If there is no forecasted production in this timeframe and the last action was discharging,
        # it indicates premature battery use, so a penalty is applied.
        if (not any(state > 0 for state in self.state[1:4])) and self.last_action == 0:
            self.reward -= self.penalty_early_discharge

        # Calculate the State of Charge (SOC) of the battery
        soc = self.inverter_sim.batt_sim.current_charge / self.inverter_sim.batt_sim.capacity

        # Check if we are in the last hour of daily production (`self.state[0] > 0` and `self.state[1] == 0`)
        # Apply penalties or rewards based on the battery's SOC to optimize usage for the next cycle
        if self.state[0] > 0 and self.state[1] == 0:
            if soc < self.night_reserver:  # Penalize if SOC is too low to ensure enough reserve for night consumption
                self.reward -= self.penalty_below_night_reserve
            elif soc < self.near_full:  # Reward maintaining a near-full SOC for efficient utilization
                self.reward += self.reward_near_full
        if soc < self.min_reserve:
            self.reward -= self.penalty_below_min_reserve
        return self.reward
