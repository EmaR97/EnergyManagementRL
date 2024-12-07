import numpy as np
from scipy.ndimage._filters import _gaussian_kernel1d, correlate1d

min5 = 1 / 12
day = 24 * 12
week = day * 7
month = week * 4
full_period = month * 5


class SmoothedHistory:
    def __init__(self, sigma, steps):
        self.steps = steps
        self.buffer = []
        self.kernel = _gaussian_kernel1d(sigma, 0, int(4 * float(sigma) + 0.5))[::-1]

    def get_smoothed_history(self, new_point, _range):
        if not self.buffer:
            self.buffer = [new_point] * self.steps
        self.buffer.append(new_point)
        self.buffer.pop(0)
        smoothed: np.ndarray = correlate1d(self.buffer, self.kernel, -1, None, "reflect", .0, 0)
        return [smoothed[i] for i in _range]
