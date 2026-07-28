import pytest

from energymanagementrl.simulation.grid_sim import GridSim
from energymanagementrl.simulation.utils import min5


class TestGridSim:
    def test_surplus_feeds_grid(self, grid_sim):
        grid_sim.step_index = 0
        remaining = grid_sim.step(1000)
        assert grid_sim.get_feed_to() > 0

    def test_deficit_takes_from_grid(self, grid_sim):
        grid_sim.step_index = 0
        grid_sim.step(-1000)
        assert grid_sim.get_taken_from() > 0

    def test_zero_balance(self, grid_sim):
        grid_sim.step_index = 0
        grid_sim.step(0)
        assert grid_sim.get_feed_to() == 0
        assert grid_sim.get_taken_from() == 0

    def test_voltage_affects_acceptance(self):
        voltage_low = [230.0] * 100
        voltage_high = [250.0] * 100

        sim_low = GridSim(
            feed_in_max=3500,
            feed_in_min=0,
            voltage_max=250,
            voltage_min=230,
            max_taken_from=6000,
            energy_price_sell_per_kwh=0.1 / 1000,
            energy_price_buy_per_kwh=0.4 / 1000,
            voltage_series=voltage_low,
        )
        sim_high = GridSim(
            feed_in_max=3500,
            feed_in_min=0,
            voltage_max=250,
            voltage_min=230,
            max_taken_from=6000,
            energy_price_sell_per_kwh=0.1 / 1000,
            energy_price_buy_per_kwh=0.4 / 1000,
            voltage_series=voltage_high,
        )

        sim_low.step_index = 0
        sim_high.step_index = 0

        acceptance_low = sim_low.get_grid_acceptance()
        acceptance_high = sim_high.get_grid_acceptance()

        assert acceptance_low > acceptance_high

    def test_grid_limited_by_max_taken(self, grid_sim):
        grid_sim.step_index = 0
        grid_sim.step(-10000)
        assert grid_sim.get_taken_from() <= 6000 * min5

    def test_reset(self, grid_sim):
        grid_sim.step_index = 0
        grid_sim.step(1000)
        grid_sim.reset()
        assert grid_sim.step_index == 0
        assert grid_sim.current_feed_to_grid == 0
        assert grid_sim.current_taken_from_grid == 0

    def test_get_state(self, grid_sim):
        state = grid_sim.get_state()
        assert "feed_to_grid" in state
        assert "taken_from_grid" in state

    def test_check_max_steps_raises(self, grid_sim):
        with pytest.raises(ValueError, match="Allowed max steps"):
            grid_sim.check_max_steps(len(grid_sim.voltage_series) + 100)

    def test_map_voltage_to_power(self, grid_sim):
        power_at_min = grid_sim._map_voltage_to_power(230)
        power_at_max = grid_sim._map_voltage_to_power(250)
        assert power_at_min > power_at_max
        assert power_at_max == 0
