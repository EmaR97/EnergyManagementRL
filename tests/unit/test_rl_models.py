import numpy as np

from energymanagementrl.rl.models import GreedyModel, ConservativeModel, SimpleModel


class TestGreedyModel:
    def test_always_returns_0(self):
        model = GreedyModel()
        for _ in range(10):
            action, _ = model.predict(np.zeros(55))
            assert action == 0


class TestConservativeModel:
    def test_always_returns_1(self):
        model = ConservativeModel()
        for _ in range(10):
            action, _ = model.predict(np.zeros(55))
            assert action == 1


class TestSimpleModel:
    def test_daytime_action(self):
        model = SimpleModel(lowerbound=60, upperbound=210)
        obs = np.zeros(55)
        sin_val = np.sin(2 * np.pi * 150 / 288)
        cos_val = np.cos(2 * np.pi * 150 / 288)
        obs[-2] = (sin_val + 1) / 2
        obs[-1] = (cos_val + 1) / 2
        action, _ = model.predict(obs)
        assert action == 0

    def test_nighttime_action(self):
        model = SimpleModel(lowerbound=60, upperbound=210)
        obs = np.zeros(55)
        sin_val = np.sin(2 * np.pi * 10 / 288)
        cos_val = np.cos(2 * np.pi * 10 / 288)
        obs[-2] = (sin_val + 1) / 2
        obs[-1] = (cos_val + 1) / 2
        action, _ = model.predict(obs)
        assert action == 1
