from .battery_sim import BatterySim
from .production_sim import ProductionSim,ProductionSimFromReal
from .grid_sim import GridSim
from .inverter_sim import InverterSim
from .consumption_sim import ConsumptionSim
from .utils import *


def get_simulation_params(config: dict) -> dict:
    return {
        "battery": config["battery"],
        "grid": config["grid"],
        "simulation": config["simulation"],
    }
