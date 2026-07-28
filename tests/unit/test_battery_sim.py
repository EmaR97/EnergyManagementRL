from energymanagementrl.simulation.battery_sim import BatterySim


class TestBatterySim:
    def test_initial_charge(self):
        sim = BatterySim(
            max_charge_rate=5000,
            max_discharge_rate=5000,
            capacity=9000,
            battery_wear_rate=0,
            starting_charge=4500,
        )
        assert sim.get_stored() == 4500

    def test_charge_increases_soc(self):
        sim = BatterySim(
            max_charge_rate=5000,
            max_discharge_rate=5000,
            capacity=9000,
            battery_wear_rate=0,
            starting_charge=0,
        )
        remaining = sim.step(1000)
        assert sim.get_stored() > 0
        assert remaining >= 0

    def test_discharge_decreases_soc(self):
        sim = BatterySim(
            max_charge_rate=5000,
            max_discharge_rate=5000,
            capacity=9000,
            battery_wear_rate=0,
            starting_charge=4500,
        )
        sim.step(-1000)
        assert sim.get_stored() < 4500

    def test_charge_rate_limited(self):
        sim = BatterySim(
            max_charge_rate=5000,
            max_discharge_rate=5000,
            capacity=9000,
            battery_wear_rate=0,
            starting_charge=8000,
        )
        sim.step(10000)
        assert sim.get_stored() <= 9000

    def test_discharge_rate_limited(self):
        sim = BatterySim(
            max_charge_rate=5000,
            max_discharge_rate=5000,
            capacity=9000,
            battery_wear_rate=0,
            starting_charge=100,
        )
        sim.step(-10000)
        assert sim.get_stored() >= 0

    def test_no_energy_change(self):
        sim = BatterySim(
            max_charge_rate=5000,
            max_discharge_rate=5000,
            capacity=9000,
            battery_wear_rate=0,
            starting_charge=4500,
        )
        sim.step(0)
        assert sim.get_stored() == 4500

    def test_reset(self):
        sim = BatterySim(
            max_charge_rate=5000,
            max_discharge_rate=5000,
            capacity=9000,
            battery_wear_rate=0,
            starting_charge=4500,
        )
        sim.step(1000)
        sim.reset()
        assert sim.step_index == 0
        assert sim.current_charge_rate == 0
        assert sim.current_discharge_rate == 0

    def test_get_state(self):
        sim = BatterySim(
            max_charge_rate=5000,
            max_discharge_rate=5000,
            capacity=9000,
            battery_wear_rate=0,
            starting_charge=4500,
        )
        state = sim.get_state()
        assert "stored" in state
        assert "charge_rate" in state
        assert "discharge_rate" in state

    def test_random_starting_charge(self):
        sim = BatterySim(
            max_charge_rate=5000,
            max_discharge_rate=5000,
            capacity=9000,
            battery_wear_rate=0,
        )
        assert 0 <= sim.get_stored() <= 9000

    def test_efficiency_losses(self):
        sim = BatterySim(
            max_charge_rate=5000,
            max_discharge_rate=5000,
            capacity=9000,
            battery_wear_rate=0,
            efficiency=0.9,
            starting_charge=0,
        )
        sim.step(1000)
        assert sim.get_stored() < 1000
