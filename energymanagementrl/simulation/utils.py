import numpy as np
from scipy.ndimage._filters import _gaussian_kernel1d, correlate1d
from scipy.sparse import csr_matrix

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


def create_gaussian_under_sampling_matrix(input_length, under_sampling_factor):
    sigma = under_sampling_factor // 2
    x = np.arange(input_length)
    centers = np.arange(sigma, input_length, under_sampling_factor)
    offsets = x[np.newaxis, :] - centers[:, np.newaxis]
    gaussian = np.exp(-0.5 * (offsets / sigma) ** 2)
    gaussian[gaussian < 1 / 1000] = 0  # Threshold for negligible values
    gaussian /= gaussian.sum(axis=1, keepdims=True)
    return gaussian


# Define intervals for different resolutions
identity_steps = 6
mid_steps = 6
far_steps = 6

# Dynamically compute rows for Gaussian matrices
mid_gaussian = create_gaussian_under_sampling_matrix(48, 2)
far_gaussian = create_gaussian_under_sampling_matrix(48, 5)

mixed_resampling_matrix = np.concatenate(
    (
        np.eye(48)[:identity_steps],
        mid_gaussian[(identity_steps // 2):((identity_steps // 2) + mid_steps)],
        far_gaussian[-far_steps:]
    )
)
sparse_matrix = csr_matrix(mixed_resampling_matrix).T
