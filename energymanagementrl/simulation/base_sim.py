from abc import abstractmethod

import numpy as np


class BaseSim:
    def __init__(self, seed=None):
        """
        Initialize the simulation.

        Args:
            seed (int, optional): An optional seed for randomization or state initialization.
        """
        self.step_index = 0
        self.random_seed = seed
        self.random_state = np.random.RandomState(self.random_seed)

    def reset(self, seed=None):
        """
        Reset the simulation to its initial state.

        Args:
            seed (int, optional): An optional seed for randomization or state initialization.
        """
        self.random_seed = seed
        self.random_state = np.random.RandomState(self.random_seed)
        self.step_index = 0

    def step(self, **inputs):
        """
        Advance the simulation by one step.

        Args:
            inputs (any): Inputs affecting the simulation state.
        """
        self.step_index += 1

    @abstractmethod
    def get_state(self):
        pass
