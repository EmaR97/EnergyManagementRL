import pytest

from energymanagementrl.simulation.energy_sim import EnergySim
from energymanagementrl.simulation.utils import min5, day


class ConcreteEnergySim(EnergySim):
    def step(self, **inputs):
        super().step(**inputs)
        return self.get_energy()

    def get_energy_sample(self):
        return [self.energy_series[self.step_index]]


class TestEnergySim:
    def test_energy_series_conversion(self, sample_power_series):
        sim = ConcreteEnergySim(power_series=sample_power_series)
        expected = [x * min5 for x in sample_power_series]
        assert sim.energy_series == pytest.approx(expected)

    def test_get_energy(self, sample_power_series):
        sim = ConcreteEnergySim(power_series=sample_power_series)
        expected = int(sample_power_series[0] * min5)
        assert sim.get_energy() == expected

    def test_step_returns_energy(self, sample_power_series):
        sim = ConcreteEnergySim(power_series=sample_power_series)
        result = sim.step()
        assert result == int(sample_power_series[1] * min5)

    def test_step_advances_index(self, sample_power_series):
        sim = ConcreteEnergySim(power_series=sample_power_series)
        sim.step()
        assert sim.step_index == 1

    def test_reset(self, sample_power_series):
        sim = ConcreteEnergySim(power_series=sample_power_series)
        sim.step()
        sim.step()
        sim.reset()
        assert sim.step_index == 0

    def test_sample_size(self, sample_power_series):
        sim = ConcreteEnergySim(power_series=sample_power_series, daily_sample=6)
        assert sim.sample_size == day // 6

    def test_forecast_range(self, sample_power_series):
        sim = ConcreteEnergySim(
            power_series=sample_power_series, daily_sample=6, forecast_steps=12
        )
        assert len(sim.forecast_range) == 12
        assert sim.forecast_range[0] == 0
        assert sim.forecast_range[1] == sim.sample_size

    def test_check_max_steps_raises(self, sample_power_series):
        sim = ConcreteEnergySim(power_series=sample_power_series)
        with pytest.raises(ValueError, match="Allowed max steps"):
            sim.check_max_steps(len(sample_power_series) + 100)

    def test_check_max_steps_ok(self, sample_power_series):
        sim = ConcreteEnergySim(power_series=sample_power_series)
        sim.check_max_steps(len(sample_power_series))

    def test_get_state(self, sample_power_series):
        sim = ConcreteEnergySim(power_series=sample_power_series)
        state = sim.get_state()
        assert "energy_sample_0" in state
        assert "energy" in state

    def test_get_allowed_max_steps(self, sample_power_series):
        sim = ConcreteEnergySim(power_series=sample_power_series)
        assert sim.get_allowed_max_steps() == len(sample_power_series)
