import numpy as np
import pandas as pd
import pytest

from energymanagementrl.simulation import (
    ProductionSim,
    ProductionSimFromReal,
    ConsumptionSim,
    BatterySim,
    GridSim,
    InverterSim,
)


@pytest.fixture
def sample_power_series():
    rng = np.random.RandomState(42)
    return list(rng.uniform(0, 5000, size=288 * 14))


@pytest.fixture
def sample_voltage_series():
    rng = np.random.RandomState(42)
    return list(rng.uniform(230, 250, size=288 * 14))


@pytest.fixture
def sample_consumption_series():
    rng = np.random.RandomState(42)
    return list(rng.uniform(100, 3000, size=288 * 14))


@pytest.fixture
def production_sim(sample_power_series):
    return ProductionSim(
        power_series=sample_power_series,
        daily_sample=6,
        forecast_steps=48,
    )


@pytest.fixture
def production_sim_from_real(sample_power_series):
    rng = np.random.RandomState(42)
    optimal = list(rng.uniform(0, 5000, size=288 * 14))
    weather = list(rng.uniform(0, 4500, size=288 * 14))
    return ProductionSimFromReal(
        power_series=sample_power_series,
        optimal_power_series=optimal,
        weather_power_series=weather,
        forecast_steps=48,
    )


@pytest.fixture
def consumption_sim(sample_consumption_series):
    return ConsumptionSim(
        power_series=sample_consumption_series,
        daily_sample=6,
        forecast_steps=12,
    )


@pytest.fixture
def battery_sim():
    return BatterySim(
        max_charge_rate=5000,
        max_discharge_rate=5000,
        capacity=9000,
        battery_wear_rate=0,
        efficiency=0.95,
        starting_charge=4500,
    )


@pytest.fixture
def grid_sim(sample_voltage_series):
    return GridSim(
        feed_in_max=3500,
        feed_in_min=0,
        voltage_max=250,
        voltage_min=230,
        max_taken_from=6000,
        energy_price_sell_per_kwh=0.1 / 1000,
        energy_price_buy_per_kwh=0.4 / 1000,
        voltage_series=sample_voltage_series,
    )


@pytest.fixture
def inverter_sim(production_sim, consumption_sim, battery_sim, grid_sim):
    return InverterSim(
        prod_sim=production_sim,
        cons_sim=consumption_sim,
        batt_sim=battery_sim,
        grid_sim=grid_sim,
    )


@pytest.fixture
def sample_config():
    return {
        "log_level": "INFO",
        "solar_plant": {
            "timezone": "Europe/Rome",
            "panel_model": {
                "pdc0": 0.42,
                "temp_model_a": -3.56,
                "temp_model_b": -0.075,
                "delta_t": 3,
                "gamma_pdc": -0.004,
            },
            "num_panels": 14,
            "arrays": [
                {"name": "sud_east", "tilt_angle": 25, "azimuth": 110},
                {"name": "nord_west", "tilt_angle": 18, "azimuth": 290},
            ],
            "inverter": {"pdc0": 8, "id": "NE=150331722", "huawei_subdomain": "uni004eu5"},
        },
        "battery": {
            "max_charge_rate": 5000,
            "max_discharge_rate": 5000,
            "capacity": 9000,
            "battery_wear_rate": 0,
        },
        "grid": {
            "feed_in_max": 3500,
            "feed_in_min": 0,
            "voltage_max": 250,
            "voltage_min": 230,
            "max_taken_from": 6000,
            "energy_price_sell_per_kwh": 0.1,
            "energy_price_buy_per_kwh": 0.4,
        },
        "simulation": {
            "timesteps_per_day": 288,
            "interval_minutes": 5,
            "forecast_steps": {"production": 48, "consumption": 12},
            "daily_sample": 6,
        },
        "training": {
            "algorithm": "DQN",
            "policy": "MlpPolicy",
            "device": "cpu",
            "learning_rate": 0.0003,
            "batch_size": 2048,
            "num_envs": 4,
            "train_periods": 10,
            "shuffle": 2,
            "rewards": {
                "energy_price_buy_per_kwh": 1.0,
                "reward_near_full": 0.06,
                "penalty_below_night_reserve": 0.06,
                "penalty_below_min_reserve": 0.02,
            },
            "battery_reserves": {
                "min_reserve": 0.1,
                "night_reserver": 0.85,
                "near_full": 0.98,
            },
            "checkpoint": {
                "save_freq": 50000,
                "save_replay_buffer": True,
                "save_vecnormalize": True,
                "name_prefix": "latest_model",
            },
            "eval": {"eval_freq_multiplier": 10, "train_max_steps_multiplier": 2},
        },
        "data_paths": {
            "base": "../data",
            "simulation_inputs": "../data/simulation_inputs",
            "trained_models": "../data/trained_models",
            "logs": "../data/logs",
        },
        "kaggle": {"datasets": {}, "sheet_key": "kaggle"},
    }


@pytest.fixture
def complete_series_df():
    rng = np.random.RandomState(42)
    n = 288 * 7
    timestamps = pd.date_range("2024-10-20", periods=n, freq="5min")
    df = pd.DataFrame(
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
    df.index.name = "index"
    return df


def pytest_collection_modifyitems(items):
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path:
            item.add_marker(pytest.mark.run(order=0))
        elif "/integration/" in path:
            item.add_marker(pytest.mark.run(order=2))
        else:
            item.add_marker(pytest.mark.run(order=1))
