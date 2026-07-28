import numpy as np

from energymanagementrl.simulation.production_sim import ProductionSim
from energymanagementrl.simulation.utils import sparse_matrix


class TestProductionSim:
    def test_get_energy(self, production_sim):
        energy = production_sim.get_energy()
        assert isinstance(energy, int)
        assert energy >= 0

    def test_step_returns_energy(self, production_sim):
        result = production_sim.step()
        assert isinstance(result, int)

    def test_energy_sample_length(self, production_sim):
        sample = production_sim.get_energy_sample()
        expected_len = sparse_matrix.shape[1]
        assert len(sample) == expected_len

    def test_get_state_keys(self, production_sim):
        state = production_sim.get_state()
        assert "energy" in state
        assert any(k.startswith("energy_sample_") for k in state)

    def test_max_steps_reduction(self, sample_power_series):
        sim = ProductionSim(power_series=sample_power_series, forecast_steps=48)
        assert sim.get_allowed_max_steps() < len(sample_power_series)

    def test_step_advances(self, production_sim):
        production_sim.step()
        assert production_sim.step_index == 1

    def test_reset(self, production_sim):
        production_sim.step()
        production_sim.step()
        production_sim.reset()
        assert production_sim.step_index == 0


class TestProductionSimFromReal:
    def test_residual_computed(self, production_sim_from_real):
        assert len(production_sim_from_real.residual_series) > 0

    def test_residual_non_negative(self, production_sim_from_real):
        assert all(r >= 0 for r in production_sim_from_real.residual_series)

    def test_get_residual_sample_length(self, production_sim_from_real):
        sample = production_sim_from_real.get_residual_sample()
        expected_len = sparse_matrix.shape[1]
        assert len(sample) == expected_len

    def test_get_state_includes_residual(self, production_sim_from_real):
        state = production_sim_from_real.get_state()
        assert any(k.startswith("residual_sample_") for k in state)

    def test_energy_sample_uses_weather(self, production_sim_from_real):
        sample = production_sim_from_real.get_energy_sample()
        assert len(sample) > 0
        assert all(isinstance(v, (int, float, np.integer, np.floating)) for v in sample)

    def test_shuffle_on_reset(self, production_sim_from_real):
        production_sim_from_real.step()
        production_sim_from_real.step()
        original_energy = list(production_sim_from_real.energy_series)
        production_sim_from_real.reset(seed=1, shuffle=2)
        may_differ = list(production_sim_from_real.energy_series)
        assert len(original_energy) == len(may_differ)
