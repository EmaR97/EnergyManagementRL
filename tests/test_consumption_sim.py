import unittest

import pandas as pd

from energymanagementrl.simulation import min5, ConsumptionSim
from energymanagementrl.simulation.consumption_sim import get_smoothed_history


class TestConsumptionSim(unittest.TestCase):

    def setUp(self):
        """
        Set up a sample power series for testing the ConsumptionSim class.
        """
        complete_series_csv = '../data/complete_series.csv'
        df = pd.read_csv(complete_series_csv, parse_dates=['timestamp'])
        self.power_series = df.consumption_w[:2000]
        self.min5 = min5
        self.daily_sample = 6
        self.forecast_steps = 24
        self.sim = ConsumptionSim(self.power_series)

    def test_initialization(self):
        """
        Test if the ConsumptionSim initializes correctly.
        """
        self.assertEqual(self.sim.current_energy, 0)
        self.assertEqual(self.sim.step_index, 0)
        expected_energy_series = [x * self.min5 for x in self.power_series]
        self.assertEqual(self.sim.energy_series, expected_energy_series)
        self.assertEqual(self.sim.max_step, int(max(expected_energy_series)))
        self.assertTrue(len(self.sim.energy_samples) > 0)

    def test_get_energy_sample(self):
        """
        Test retrieving smoothed energy samples.
        """
        sample = self.sim.get_energy_sample()
        print(sample)
        self.assertIsInstance(sample, list)
        self.assertTrue(all(isinstance(x, float) for x in sample))
        self.assertEqual(len(sample), len(self.sim.forecast_range))

    def test_step_and_sample_smoothness(self):
        """
        Test if the energy samples update smoothly with each step.
        """
        previous_sample = self.sim.get_energy_sample()
        for _ in range(5):
            self.sim.step()
            current_sample = self.sim.get_energy_sample()
            self.assertEqual(len(current_sample), len(previous_sample))
            # Assert that the smoothed values do not vary drastically
            for prev, curr in zip(previous_sample, current_sample):
                self.assertLessEqual(abs(curr - prev), 2.0)
            previous_sample = current_sample

    def test_reset(self):
        """
        Test resetting the simulation.
        """
        self.sim.step()
        self.sim.reset()
        self.assertEqual(self.sim.current_energy, 0)
        self.assertEqual(self.sim.step_index, 0)

    def test_get_smoothed_history(self):
        """
        Test the smoothing function used by ConsumptionSim.
        """
        global buffer
        buffer = []  # Ensure a fresh buffer for each test
        steps = 10
        _range = range(steps)
        sigma = 12
        results = []
        for i in range(steps):
            results.append(get_smoothed_history(i, _range, steps))
        self.assertEqual(len(results[-1]), steps)
        # Check for smoothness in the final result
        for i in range(1, steps):
            self.assertLessEqual(abs(results[-1][i] - results[-1][i - 1]), 2.0)

    def test_energy_sim_integration(self):
        """
        Test if the ConsumptionSim integrates correctly with EnergySim features.
        """
        for _ in range(3):
            self.sim.step()
        current_energy = self.sim.get_energy()
        self.assertEqual(current_energy, int(self.sim.energy_series[self.sim.step_index - 1]))


if __name__ == "__main__":
    unittest.main()
