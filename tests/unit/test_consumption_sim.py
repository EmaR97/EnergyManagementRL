from energymanagementrl.simulation.consumption_sim import ConsumptionSim


class TestConsumptionSim:
    def test_step_returns_energy(self, consumption_sim):
        result = consumption_sim.step()
        assert isinstance(result, int)

    def test_step_advances(self, consumption_sim):
        consumption_sim.step()
        assert consumption_sim.step_index == 1

    def test_reset(self, consumption_sim):
        consumption_sim.step()
        consumption_sim.step()
        consumption_sim.reset()
        assert consumption_sim.step_index == 0

    def test_energy_sample_length(self, consumption_sim):
        sample = consumption_sim.get_energy_sample()
        assert len(sample) == 12

    def test_get_state_keys(self, consumption_sim):
        state = consumption_sim.get_state()
        assert "energy" in state
        assert any(k.startswith("energy_sample_") for k in state)

    def test_shuffle_changes_series(self, sample_consumption_series):
        sim = ConsumptionSim(power_series=sample_consumption_series, daily_sample=6)
        sim.step()
        original = list(sim.energy_series)
        sim.reset(seed=42, shuffle=2)
        new = list(sim.energy_series)
        assert len(original) == len(new)


class TestSmoothedHistory:
    def test_smoothing(self):
        from energymanagementrl.simulation.utils import SmoothedHistory

        history = SmoothedHistory(sigma=3, steps=10)
        for i in range(20):
            result = history.get_smoothed_history(i, range(10))
            assert len(result) == 10

    def test_smoothing_stabilizes(self):
        from energymanagementrl.simulation.utils import SmoothedHistory

        history = SmoothedHistory(sigma=3, steps=10)
        for i in range(50):
            result = history.get_smoothed_history(100, range(10))
        assert all(abs(v - 100) < 10 for v in result)
