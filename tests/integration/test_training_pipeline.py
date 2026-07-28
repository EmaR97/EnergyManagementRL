import copy

import numpy as np
import pytest

from energymanagementrl.rl.env import InverterEnvBatteryMgmt
from energymanagementrl.rl.models import GreedyModel, ConservativeModel
from energymanagementrl.simulation import (
    ProductionSimFromReal,
    ConsumptionSim,
    BatterySim,
    GridSim,
    InverterSim,
    week,
)


def _make_training_env(max_steps=None):
    rng = np.random.RandomState(42)
    n = 288 * 14
    prod = list(rng.uniform(0, 5000, size=n))
    prod_weather = list(rng.uniform(0, 4500, size=n))
    prod_optimal = list(rng.uniform(0, 5000, size=n))
    cons = list(rng.uniform(100, 3000, size=n))
    volt = list(rng.uniform(230, 250, size=n))

    i_sim = InverterSim(
        prod_sim=ProductionSimFromReal(
            power_series=prod,
            optimal_power_series=prod_optimal,
            weather_power_series=prod_weather,
            forecast_steps=48,
        ),
        cons_sim=ConsumptionSim(power_series=cons, daily_sample=6, forecast_steps=12),
        batt_sim=BatterySim(
            max_charge_rate=5000, max_discharge_rate=5000, capacity=9000, battery_wear_rate=0, starting_charge=4500
        ),
        grid_sim=GridSim(
            feed_in_max=3500,
            feed_in_min=0,
            voltage_max=250,
            voltage_min=230,
            max_taken_from=6000,
            energy_price_sell_per_kwh=0.1 / 1000,
            energy_price_buy_per_kwh=0.4 / 1000,
            voltage_series=volt,
        ),
    )
    return InverterEnvBatteryMgmt(
        inverter_sim=i_sim,
        max_steps=max_steps or week,
        reward_near_full=0.06,
        penalty_below_night_reserve=0.06,
        penalty_below_min_reserve=0.02,
        night_reserver=0.85,
        near_full=0.98,
        min_reserve=0.1,
    )


class TestTrainingPipeline:
    def test_env_creation(self):
        env = _make_training_env()
        assert env.observation_space.shape[0] > 0

    def test_baselines_run(self):
        env = _make_training_env(max_steps=288)
        for model in [GreedyModel(), ConservativeModel()]:
            obs, _ = env.reset()
            total_reward = 0
            for _ in range(288):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                if done:
                    break
            assert isinstance(total_reward, float)

    def test_training_loop_with_sb3(self):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import DQN
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = _make_training_env(max_steps=288)
        vec_env = DummyVecEnv([lambda: copy.deepcopy(env)])

        model = DQN("MlpPolicy", vec_env, verbose=0, learning_rate=0.001, batch_size=64, device="cpu")
        model.learn(total_timesteps=500)
        assert model is not None

        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape[0] == 1
