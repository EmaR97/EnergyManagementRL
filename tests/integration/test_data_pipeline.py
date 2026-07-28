import numpy as np
import pandas as pd

from energymanagementrl.rl.env import InverterEnv
from energymanagementrl.simulation import (
    ProductionSimFromReal,
    ConsumptionSim,
    BatterySim,
    GridSim,
    InverterSim,
)


def _make_complete_series(n=288 * 7):
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2024-10-20", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "production_power_kw": rng.uniform(0, 4, n),
            "production_power_kw_weather_dependent": rng.uniform(0, 3.8, n),
            "production_power_kw_optimal": rng.uniform(0, 4.2, n),
            "load_power_kw": -rng.uniform(0.2, 3, n),
            "SOC": rng.uniform(10, 100, n),
            "grid_voltage": rng.uniform(230, 250, n),
        },
        index=timestamps,
    )


class TestDataPipeline:
    def test_load_and_prepare_training_data(self):
        df = _make_complete_series()
        production_w = df.production_power_kw * 1000
        production_w_weather = df.production_power_kw_weather_dependent * 1000
        optimal_w = df.production_power_kw_optimal * 1000
        consumption_w = -df.load_power_kw * 1000
        grid_voltage = df.grid_voltage

        assert len(production_w) == len(consumption_w)
        assert (production_w >= 0).all()
        assert (consumption_w >= 0).all()

    def test_full_simulation_pipeline(self):
        df = _make_complete_series()
        production_w = df.production_power_kw * 1000
        production_w_weather = df.production_power_kw_weather_dependent * 1000
        optimal_w = df.production_power_kw_optimal * 1000
        consumption_w = -df.load_power_kw * 1000
        grid_voltage = df.grid_voltage

        p_sim = ProductionSimFromReal(
            power_series=production_w,
            optimal_power_series=optimal_w,
            weather_power_series=production_w_weather,
            forecast_steps=48,
        )
        c_sim = ConsumptionSim(power_series=consumption_w, daily_sample=6, forecast_steps=12)
        b_sim = BatterySim(max_charge_rate=5000, max_discharge_rate=5000, capacity=9000, battery_wear_rate=0)
        g_sim = GridSim(
            feed_in_max=3500,
            feed_in_min=0,
            voltage_max=250,
            voltage_min=230,
            max_taken_from=6000,
            energy_price_sell_per_kwh=0.1 / 1000,
            energy_price_buy_per_kwh=0.4 / 1000,
            voltage_series=grid_voltage,
        )
        i_sim = InverterSim(prod_sim=p_sim, cons_sim=c_sim, batt_sim=b_sim, grid_sim=g_sim)
        env = InverterEnv(i_sim, max_steps=288 * 7 - 288 * 2)

        obs, _ = env.reset()
        assert obs.shape[0] > 0

        total_reward = 0
        for _ in range(100):
            obs, reward, done, truncated, info = env.step(0)
            total_reward += reward
            if done:
                break

        assert isinstance(total_reward, float)
