from .base_sim import BaseSim

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


class WeatherSim(BaseSim):

    def __init__(
            self,
            cloud_transition_matrices,
            noise_transition_matrix,
            resolution=12,
            uncertainty_factors=(0.2, 0.1, 0.05),
            weights=(0.7, 0.4, 0.15),
            time_steps=24 * 7 * 20
    ):
        super().__init__()
        self.cloud_transition_matrices = cloud_transition_matrices
        self.noise_transition_matrix = noise_transition_matrix
        self.resolution = resolution
        self.uncertainty_factors = uncertainty_factors
        self.weights = weights
        self.time_steps = time_steps
        self.attenuation_series = None
        self.cloud_coverage_series = None
        self.reset()  # 20 weeks

    def reset(self):
        super().reset()
        self.cloud_coverage_series = self._get_cloud_coverage_series()
        self.attenuation_series = self._get_attenuation_series()

    def step(self):
        super().step()
        return (
            self.cloud_coverage_series[:, self.step_index // self.resolution],
            self.attenuation_series[self.step_index]
        )

    def _get_cloud_coverage_series(self):
        """
        Simulates cloud coverage states for multiple layers over time.

        Parameters:
        - transition_matrix_3d (np.ndarray): 3D transition matrix for cloud states.
        - time_steps (int): Number of time steps to simulate.

        Returns:
        - list[np.ndarray]: List of cloud states for each layer.
        """
        return np.array([
            simulate_markov_chain(
                initial_state=np.random.randint(0, 100),
                trans_matrix=self.cloud_transition_matrices[layer],
                steps=self.time_steps
            ) / 100  # Normalize to [0, 1]
            for layer in range(self.cloud_transition_matrices.shape[0])
        ])

    def _get_attenuation_series(self):
        """
        Optimized calculation of solar attenuation.
        """
        smoothed_clouds = [
            interpolate_and_smooth(layer, self.resolution) for layer in self.cloud_coverage_series
        ]
        uncertainties = [
            calc_uncertainty(layer, scale=self.uncertainty_factors[idx])
            for idx, layer in enumerate(smoothed_clouds)
        ]
        noise_states = simulate_markov_chain(
            np.random.randint(0, 100),
            self.noise_transition_matrix,
            len(smoothed_clouds[0])
        ) / 50 - 1

        # Efficiently aggregate attenuation components
        attenuation_components = [
            smoothed * (noise_states * uncertainty + self.weights[idx])
            for idx, (smoothed, uncertainty) in enumerate(zip(smoothed_clouds, uncertainties))
        ]
        return np.clip(sum(attenuation_components), 0, 1)


def simulate_markov_chain(initial_state, trans_matrix, steps=100):
    """
    Simulates a Markov chain using cumulative probabilities for speed.
    """
    cumulative_probs = np.cumsum(trans_matrix, axis=1)  # Precompute cumulative probabilities
    states = np.zeros(steps, dtype=int)
    states[0] = initial_state

    random_values = np.random.random(size=steps - 1)  # Pre-generate random values
    for i in range(1, steps):
        states[i] = np.searchsorted(cumulative_probs[states[i - 1]], random_values[i - 1])

    return states


def interpolate_and_smooth(data, resolution=12):
    """
    Interpolates and smooths data using Gaussian filtering.

    Parameters:
    - data (np.ndarray): Array of data to process.
    - resolution (int): Number of intervals per hour for interpolation.

    Returns:
    - np.ndarray: Smoothed data.
    """
    hours = np.arange(len(data))
    finer_intervals = np.linspace(0, len(data) - 1, len(data) * resolution)

    interpolated_data = interp1d(hours, data, kind='linear')(finer_intervals)
    return gaussian_filter1d(interpolated_data, sigma=resolution / 2)


def calc_uncertainty(cloud_fraction, scale=0.2, exp_factor=1):
    """
    Computes uncertainty scaling based on cloud fraction.

    Parameters:
    - cloud_fraction (float or np.ndarray): Cloud coverage fraction (0 to 1).
    - scale (float): Base scale for uncertainty.
    - exp_factor (float): Exponential scaling factor.

    Returns:
    - float or np.ndarray: Adjusted uncertainty value.
    """
    return scale * np.exp(-2 * exp_factor * (1 - cloud_fraction) ** exp_factor)
