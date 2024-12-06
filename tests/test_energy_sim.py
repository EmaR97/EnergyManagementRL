from unittest import TestCase

import unittest

import pandas as pd

from energymanagementrl.simulation import EnergySim, min5, day


class TestEnergySim(TestCase):

    def setUp(self):
        """
        Set up a sample power series for testing the EnergySim class.
        """
        complete_series_csv = '../data/complete_series.csv'
        df = pd.read_csv(complete_series_csv, parse_dates=['timestamp'])
        self.power_series = df.production_w
        self.min5 = min5
        self.day = day
        self.energy_sim = EnergySim(self.power_series)

    def test_initialization(self):
        """
        Test if the EnergySim initializes correctly.
        """
        self.assertEqual(self.energy_sim.current_energy, 0)
        self.assertEqual(self.energy_sim.step_index, 0)
        expected_energy_series = [x * self.min5 for x in self.power_series]
        self.assertEqual(self.energy_sim.energy_series, expected_energy_series)
        self.assertEqual(self.energy_sim.max_step, int(max(expected_energy_series)))
        self.assertEqual(self.energy_sim.max_24h, int(max(self.energy_sim.energy_samples)))

    def test_step(self):
        """
        Test stepping through the simulation.
        """
        for i, value in enumerate(self.power_series):
            expected_energy = int(value * self.min5)
            self.assertEqual(self.energy_sim.step(), expected_energy)
            self.assertEqual(int(self.energy_sim.current_energy), expected_energy)
            self.assertEqual(self.energy_sim.step_index, i + 1)

    def test_reset(self):
        """
        Test resetting the simulation.
        """
        self.energy_sim.step()
        self.energy_sim.step()
        self.energy_sim.reset()
        self.assertEqual(self.energy_sim.current_energy, 0)
        self.assertEqual(self.energy_sim.step_index, 0)

    def test_get_energy(self):
        """
        Test retrieving the current energy.
        """
        self.energy_sim.step()
        expected_energy = int(self.power_series[0] * self.min5)
        self.assertEqual(self.energy_sim.get_energy(), expected_energy)

    def test_get_energy_sample(self):
        """
        Test retrieving the energy samples for the upcoming 24-hour window.
        """
        samples = self.energy_sim.get_energy_sample()
        self.assertEqual(len(samples), len(self.energy_sim.forecast_range))
        for i, value in enumerate(samples):
            self.assertEqual(value, int(self.energy_sim.energy_samples[self.energy_sim.step_index + i*self.energy_sim.sample_size]))

    def test_sliding_sum(self):
        """
        Test if the sliding sum calculation works correctly.
        """
        sliding_sums = self.energy_sim._get_sliding_sum()
        self.assertEqual(len(sliding_sums), len(self.energy_sim.energy_samples))
        # Verify the first sliding sum is as expected
        window_size = self.energy_sim.sample_size
        expected_sum = sum(self.energy_sim.energy_series[:window_size]) / window_size
        self.assertAlmostEqual(sliding_sums[0], expected_sum, places=5)


if __name__ == "__main__":
    unittest.main()
