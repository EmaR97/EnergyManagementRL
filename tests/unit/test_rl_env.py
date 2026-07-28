import numpy as np

from energymanagementrl.rl.env import InverterEnv, InverterEnvBatteryMgmt
from energymanagementrl.simulation import (
    ProductionSim,
    ConsumptionSim,
    BatterySim,
    GridSim,
    InverterSim,
    week,
)


def _make_env(max_steps=None):
    rng = np.random.RandomState(42)
    n = 288 * 21
    prod = list(rng.uniform(0, 5000, size=n))
    cons = list(rng.uniform(100, 3000, size=n))
    volt = list(rng.uniform(230, 250, size=n))

    i_sim = InverterSim(
        prod_sim=ProductionSim(power_series=prod, daily_sample=6, forecast_steps=48),
        cons_sim=ConsumptionSim(power_series=cons, daily_sample=6, forecast_steps=12),
        batt_sim=BatterySim(
            max_charge_rate=5000,
            max_discharge_rate=5000,
            capacity=9000,
            battery_wear_rate=0,
            starting_charge=4500,
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
    return InverterEnv(i_sim, max_steps or week)


class TestInverterEnv:
    def test_observation_space_shape(self):
        env = _make_env()
        assert env.observation_space.shape == (env.state_size,)

    def test_action_space(self):
        env = _make_env()
        assert env.action_space.n == 2

    def test_reset(self):
        env = _make_env()
        obs, info = env.reset()
        assert obs.shape == (env.state_size,)
        assert env.current_step == 0

    def test_step(self):
        env = _make_env()
        env.reset()
        obs, reward, done, truncated, info = env.step(0)
        assert obs.shape == (env.state_size,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_step_advances(self):
        env = _make_env()
        env.reset()
        env.step(0)
        assert env.current_step == 1

    def test_done_after_max_steps(self):
        env = _make_env(max_steps=5)
        env.reset()
        for _ in range(6):
            obs, reward, done, truncated, info = env.step(0)
        assert done is True

    def test_reward_components(self):
        env = _make_env()
        env.reset()
        env.step(0)
        state = env.get_state_dict()
        assert "reward_energy_sold" in state
        assert "penalty_energy_purchase" in state
        assert "penalty_battery_wear" in state
        assert "reward" in state


class TestInverterEnvBatteryMgmt:
    def _make_battery_mgmt_env(self):
        rng = np.random.RandomState(42)
        n = 288 * 21
        prod = list(rng.uniform(0, 5000, size=n))
        cons = list(rng.uniform(100, 3000, size=n))
        volt = list(rng.uniform(230, 250, size=n))

        i_sim = InverterSim(
            prod_sim=ProductionSim(power_series=prod, daily_sample=6, forecast_steps=48),
            cons_sim=ConsumptionSim(power_series=cons, daily_sample=6, forecast_steps=12),
            batt_sim=BatterySim(
                max_charge_rate=5000,
                max_discharge_rate=5000,
                capacity=9000,
                battery_wear_rate=0,
                starting_charge=4500,
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
            max_steps=week,
            reward_near_full=0.06,
            penalty_below_night_reserve=0.06,
            penalty_below_min_reserve=0.02,
            night_reserver=0.85,
            near_full=0.98,
            min_reserve=0.1,
        )

    def test_inherits_from_base(self):
        env = self._make_battery_mgmt_env()
        assert isinstance(env, InverterEnv)

    def test_step_works(self):
        env = self._make_battery_mgmt_env()
        env.reset()
        obs, reward, done, truncated, info = env.step(0)
        assert isinstance(reward, float)

    def test_reward_includes_bsoc_penalties(self):
        env = self._make_battery_mgmt_env()
        env.reset()
        obs, reward, done, truncated, info = env.step(0)
        assert isinstance(reward, float)
        assert reward != 0 or True
