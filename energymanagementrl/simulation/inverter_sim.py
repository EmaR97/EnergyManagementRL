from .base_sim import BaseSim
from .battery_sim import BatterySim
from .consumption_sim import ConsumptionSim
from .energy_sim import EnergySim
from .grid_sim import GridSim
from .production_sim import ProductionSim

MODE_A = 1
MODE_B = 0


class InverterSim(BaseSim):
    """
    InverterSim models the operation of an energy inverter system with energy production,
    consumption, battery storage, and grid interaction. The simulation offers two operation modes:
    Mode A (Max-Self-Consumption) and Mode B (Full-Feed-to-Grid). It uses precomputed sine and cosine
    values for each timestamp to simulate time-dependent behavior.

    Attributes:
        prod_sim (EnergySim): Simulation of energy production.
        cons_sim (EnergySim): Simulation of energy consumption.
        batt_sim (BatterySim): Battery simulation for energy storage.
        grid_sim (GridSim): Simulation of grid interaction.
    """

    def __init__(
            self,
            prod_sim: ProductionSim,
            cons_sim: ConsumptionSim,
            batt_sim: BatterySim,
            grid_sim: GridSim,
            seed=None,
    ):
        """
        Initializes the InverterSim with energy production, consumption, battery, grid simulations, and timestamps.

        Parameters:
            prod_sim (EnergySim): Instance for simulating energy production.
            cons_sim (EnergySim): Instance for simulating energy consumption.
            batt_sim (BatterySim): Instance for managing battery operations.
            grid_sim (GridSim): Instance for handling grid interactions.
        """
        super().__init__(seed)
        self.energy_balance = 0
        self.prod_sim = prod_sim
        self.cons_sim = cons_sim
        self.batt_sim = batt_sim
        self.grid_sim = grid_sim

    def check_max_steps(self, max_steps):
        self.prod_sim.check_max_steps(max_steps)
        self.cons_sim.check_max_steps(max_steps)
        self.grid_sim.check_max_steps(max_steps)

    def reset(self, seed=None, **kwargs) -> None:
        """
        Resets the simulation state to the starting conditions, resetting all components.
        """
        super().reset(seed)
        self.batt_sim.reset(seed, **kwargs)
        self.grid_sim.reset(seed, **kwargs)
        self.prod_sim.reset(seed, **kwargs)
        self.cons_sim.reset(seed, **kwargs)

    def step(self, action: int, **inputs) -> None:
        """
        Advances the simulation by one step and adjusts energy balance based on the chosen operation mode.

        Parameters:
            action (int): Operation mode (1 for Max-Self-Consumption, otherwise Full-Feed-to-Grid).

        Returns:
            int: Remaining energy balance after the step.
        """
        super().step(**inputs)
        prod_sim_step = self.prod_sim.step()
        cons_sim_step = prod_sim_step - self.cons_sim.step()  # Net energy (production - consumption)
        if action == MODE_A:  # Mode A (Max-Self-Consumption)
            batt_sim_step, grid_sim_step = self._manage_energy_mode_a(cons_sim_step)
        elif action == MODE_B:  # Mode B (Full-Feed-to-Grid)
            batt_sim_step, grid_sim_step = self._manage_energy_mode_b(cons_sim_step)
        else:
            raise ValueError(f'Invalid operation mode: {action}')
        # print(prod_sim_step, cons_sim_step, batt_sim_step, grid_sim_step)
        self.energy_balance = grid_sim_step

    def _manage_energy_mode_a(self, energy_balance: int):
        """
        Manages energy flow in Mode A (Max-Self-Consumption), prioritizing:
        1. Balancing production and consumption.
        2. Charging/discharging the battery.
        3. Feeding excess or drawing deficit from the grid.

        Parameters:
            energy_balance (int): Current net energy balance.

        Returns:
            int: Adjusted energy balance after managing battery and grid interaction.
        """
        batt_sim_step = self.batt_sim.step(energy_balance)
        return batt_sim_step, self.grid_sim.step(batt_sim_step)

    def _manage_energy_mode_b(self, energy_balance: int):
        """
        Manages energy flow in Mode B (Full-Feed-to-Grid), prioritizing:
        1. Balancing production and consumption.
        2. Feeding as much as possible to the grid.
        3. Storing any remaining surplus or deficit in the battery.

        Parameters:
            energy_balance (int): Current net energy balance.

        Returns:
            int: Adjusted energy balance after managing grid feed-in and battery usage.
        """
        # Consider grid as a load in Mode B
        grid_acceptance = self.grid_sim.get_grid_acceptance_ahead()
        energy_balance -= grid_acceptance  # Feed as much as possible to the grid
        energy_balance_after_batt = self.batt_sim.step(energy_balance)  # Battery handles steps excess or deficit
        energy_balance_after_batt += grid_acceptance
        return energy_balance_after_batt, self.grid_sim.step(energy_balance_after_batt)

    def random_start(self, max_steps):
        allowed_max_steps = min(
            self.grid_sim.get_allowed_max_steps(),
            self.cons_sim.get_allowed_max_steps(),
            self.prod_sim.get_allowed_max_steps()
        )
        allowed_max_steps -= max_steps + 2
        if allowed_max_steps > 0:
            starting_step = self.random_state.randint(0, allowed_max_steps)
            self.grid_sim.step_index = starting_step
            self.cons_sim.step_index = starting_step
            self.prod_sim.step_index = starting_step

    def get_state(self):
        state = {}
        state.update({'prod_sim.' + key: value for key, value in self.prod_sim.get_state().items()})
        state.update({'cons_sim.' + key: value for key, value in self.cons_sim.get_state().items()})
        state.update({'batt_sim.' + key: value for key, value in self.batt_sim.get_state().items()})
        state.update({'grid_sim.' + key: value for key, value in self.grid_sim.get_state().items()})
        state['prod_sim.energy'] -= self.energy_balance
        return state
