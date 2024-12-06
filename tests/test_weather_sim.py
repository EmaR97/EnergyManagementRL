from unittest import TestCase

import numpy as np

from energymanagementrl.simulation.weather_sim import WeatherSim


class TestWeatherSim(TestCase):
    def setUp(self):
        """
        Set up test parameters and initialize a WeatherSim instance.
        """
        self.cloud_transition_matrices = np.load('/home/emanuele/IdeaProjects/EnergyManagementRL/data/cloud_coverage_transition_matrix_3d.npy')
        self.noise_transition_matrix = np.load('/home/emanuele/IdeaProjects/EnergyManagementRL/data/noise_transition_matrix.npy')
        self.sim = WeatherSim(
            cloud_transition_matrices=self.cloud_transition_matrices,
            noise_transition_matrix=self.noise_transition_matrix,
            resolution=12,
            uncertainty_factors=(0.2, 0.1, 0.05),
            weights=(0.7, 0.4, 0.15),
            time_steps=100
        )


    def test_initialization(self):
        """
        Test the initialization and initial state of WeatherSim.
        """
        self.assertIsNotNone(self.sim.cloud_coverage_series)
        self.assertIsNotNone(self.sim.attenuation_series)
        self.assertEqual(len(self.sim.cloud_coverage_series), 3)  # Matches number of cloud layers
        self.assertEqual(len(self.sim.attenuation_series), 100 * 12)  # Matches time_steps * resolution


    def test_reset(self):
        """
        Test if the reset method reinitializes the cloud and attenuation series.
        """
        initial_cloud_series = self.sim.cloud_coverage_series
        initial_attenuation_series = self.sim.attenuation_series
        self.sim.reset()
        self.assertFalse(np.array_equal(initial_cloud_series, self.sim.cloud_coverage_series))
        self.assertFalse(np.array_equal(initial_attenuation_series, self.sim.attenuation_series))


    def test_step(self):
        """
        Test the step method.
        """
        print(self.sim.cloud_coverage_series[:,0])
        initial_step_index = self.sim.step_index
        cloud_coverage, attenuation = self.sim.step()
        self.assertEqual(self.sim.step_index, initial_step_index + 1)
        self.assertIn(cloud_coverage, self.sim.cloud_coverage_series[:,0])
        self.assertIn(attenuation, self.sim.attenuation_series)

    def test_step_multiple(self):
        """
        Test the step method over multiple steps to ensure consistent updates.
        """
        num_steps = 10  # Number of steps to test
        cloud_coverages = []
        attenuations = []

        for _ in range(num_steps):
            cloud_coverage, attenuation = self.sim.step()
            cloud_coverages.append(cloud_coverage)
            attenuations.append(attenuation)

        # Verify step_index increments correctly
        self.assertEqual(self.sim.step_index, num_steps)

        # Verify cloud coverage and attenuation values are within expected ranges
        for i in range(num_steps):
            self.assertTrue(
                (0 <= cloud_coverages[i]).all() and (cloud_coverages[i] <= 1).all(),
                f"Cloud coverage out of range at step {i}: {cloud_coverages[i]}"
            )
            self.assertTrue(
                0 <= attenuations[i] <= 1,
                f"Attenuation out of range at step {i}: {attenuations[i]}"
            )
