import numpy as np


class ConservativeModel:
    @staticmethod
    def predict(*args, **kwargs):
        action = 1
        return action, {}


class GreedyModel:
    @staticmethod
    def predict(*args, **kwargs):
        action = 0
        return action, {}


class SimpleModel:
    def __init__(self, lowerbound=60, upperbound=210):
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def predict(self, obs, *args, **kwargs):
        sin, cos = (obs[-2:] - .5) * 2
        timestep = np.arctan2(sin, cos) / np.pi / 2 * 288
        timestep = int(timestep if timestep > 0 else timestep + 288)
        action = 0 if self.lowerbound < timestep < self.upperbound else 1
        return action, {}
