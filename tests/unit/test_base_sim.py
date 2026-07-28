import numpy as np

from energymanagementrl.simulation.base_sim import BaseSim


class ConcreteSim(BaseSim):
    def get_state(self):
        return {"step": self.step_index}


class TestBaseSim:
    def test_initial_step_index(self):
        sim = ConcreteSim()
        assert sim.step_index == 0

    def test_step_increments(self):
        sim = ConcreteSim()
        sim.step()
        assert sim.step_index == 1
        sim.step()
        assert sim.step_index == 2

    def test_reset(self):
        sim = ConcreteSim()
        sim.step()
        sim.step()
        sim.reset()
        assert sim.step_index == 0

    def test_reset_with_seed(self):
        sim = ConcreteSim(seed=42)
        sim.step()
        sim.reset(seed=99)
        assert sim.random_seed == 99
        assert isinstance(sim.random_state, np.random.RandomState)

    def test_random_state_reproducibility(self):
        sim1 = ConcreteSim(seed=42)
        sim2 = ConcreteSim(seed=42)
        vals1 = [sim1.random_state.random() for _ in range(5)]
        vals2 = [sim2.random_state.random() for _ in range(5)]
        assert vals1 == vals2

    def test_get_state(self):
        sim = ConcreteSim()
        state = sim.get_state()
        assert state == {"step": 0}
