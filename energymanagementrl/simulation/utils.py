import numpy as np
from scipy.ndimage._filters import _gaussian_kernel1d, correlate1d
from scipy.sparse import csr_matrix

min5 = 1 / 12
day = 24 * 12
week = day * 7
month = week * 4
full_period = month * 5


def shuffle_array_blocks(array: np.ndarray, block_size: int, max_shift: int, mix_probability: float,
                         random_state: np.random.RandomState = None):
    """
    Shuffle the blocks of an array with controlled randomness and shifting.

    Parameters:
        array (np.ndarray): The input array to shuffle.
        block_size (int): The size of each block to be shuffled.
        max_shift (int): The maximum number of blocks to shift a block by.
        mix_probability (float): The probability (0 to 1) of shifting a block.
        random_state
    Returns:
        np.ndarray: A new array with the blocks shuffled according to the specified rules.

    Raises:
        ValueError: If block_size is not a positive integer or does not evenly divide the array length.
        ValueError: If max_shift is negative.
        ValueError: If mix_probability is not in the range [0, 1].
    """
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if max_shift < 0:
        raise ValueError("max_shift must be a non-negative integer.")
    if not (0 <= mix_probability <= 1):
        raise ValueError("mix_probability must be between 0 and 1.")
    if not random_state:
        random_state = np.random

    num_blocks = len(array) // block_size
    remainder = len(array) % block_size

    # Generate shifts with probability
    shifts = random_state.randint(-max_shift, max_shift + 1, num_blocks)
    shifts = np.where(random_state.random(num_blocks) < mix_probability, shifts, 0)

    # Determine new block positions
    new_positions = np.clip(np.arange(num_blocks) + shifts, 0, num_blocks - 1)

    # Shuffle blocks
    reshaped_array = array[:num_blocks * block_size].reshape(num_blocks, block_size)
    shuffled_blocks = reshaped_array[new_positions]

    # Flatten the shuffled blocks and append any remainder
    if remainder:
        shuffled_array = np.concatenate((shuffled_blocks.ravel(), array[-remainder:]))
    else:
        shuffled_array = shuffled_blocks.ravel()

    return shuffled_array


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
