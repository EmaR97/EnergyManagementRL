import numpy as np
import pytest

from energymanagementrl.simulation.inverter_sim import InverterSim, MODE_A, MODE_B
from energymanagementrl.simulation import (
    ProductionSim,
    ConsumptionSim,
    BatterySim,
    GridSim,
)


def _make_inverter_sim():
    rng = np.random.RandomState(42)
    n = 288 * 21
    prod = list(rng.uniform(0, 5000, size=n))
    cons = list(rng.uniform(100, 3000, size=n))
    volt = list(rng.uniform(230, 250, size=n))

    return InverterSim(
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


class TestInverterSim:
    def test_mode_a(self):
        sim = _make_inverter_sim()
        sim.step(MODE_A)
        assert sim.step_index == 1

    def test_mode_b(self):
        sim = _make_inverter_sim()
        sim.step(MODE_B)
        assert sim.step_index == 1

    def test_invalid_action_raises(self):
        sim = _make_inverter_sim()
        with pytest.raises(ValueError, match="Invalid operation mode"):
            sim.step(2)

    def test_reset(self):
        sim = _make_inverter_sim()
        sim.step(MODE_A)
        sim.step(MODE_A)
        sim.reset()
        assert sim.step_index == 0

    def test_get_state_has_all_keys(self):
        sim = _make_inverter_sim()
        state = sim.get_state()
        assert any(k.startswith("prod_sim.") for k in state)
        assert any(k.startswith("cons_sim.") for k in state)
        assert any(k.startswith("batt_sim.") for k in state)
        assert any(k.startswith("grid_sim.") for k in state)

    def test_energy_balance_after_step(self):
        sim = _make_inverter_sim()
        sim.step(MODE_A)
        assert isinstance(sim.energy_balance, (int, float))

    def test_random_start(self):
        sim = _make_inverter_sim()
        initial_step = sim.prod_sim.step_index
        sim.random_start(100)
        assert sim.prod_sim.step_index >= initial_step
