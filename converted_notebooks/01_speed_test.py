# %%
import pandas as pd

from src.simulation import EnergySim, BatterySim, GridSim, InverterSim, full_period
from src.rl import InverterEnv
import time
from functools import wraps

# Dictionary to store cumulative times and counts for each decorated function
timing_data = {}


def timed_step(class_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Start timing
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            func_name = f"{class_name}.{func.__name__}"

            # Update timing data
            if func_name not in timing_data:
                timing_data[func_name] = {'total_time': 0.0, 'count': 0}
            timing_data[func_name]['total_time'] += elapsed_time
            timing_data[func_name]['count'] += 1

            return result

        return wrapper

    return decorator


def print_all_means():
    """Print the mean execution time for all decorated functions."""
    for func_name, data in timing_data.items():
        mean_time = data['total_time'] / data['count']
        print(f"{func_name} - Mean execution time: {mean_time:.8f} seconds over {data['count']} calls.")


# Applying the timing decorator to the step functions
# EnergySim.step = timed_step("EnergySim")(EnergySim.step)
# GridSim.step = timed_step("GridSim")(GridSim.step)
# BatterySim.step = timed_step("BatterySim")(BatterySim.step)

# InverterEnv._update_state = timed_step("InverterEnv")(InverterEnv._update_state)
# InverterEnv.calc_reward = timed_step("InverterEnv")(InverterEnv.calc_reward)
# InverterEnv._get_timestep = timed_step("InverterEnv")(InverterEnv._get_timestep)
# InverterEnv._manage_energy_mode_a = timed_step("InverterEnv")(InverterEnv._manage_energy_mode_a)
# InverterEnv._manage_energy_mode_b = timed_step("InverterEnv")(InverterEnv._manage_energy_mode_b)
InverterEnv.step = timed_step("InverterEnv")(InverterEnv.step)

# %%
# Example to test the speed of each step function
complete_series_csv = '../data/complete_series.csv'
df = pd.read_csv(complete_series_csv, parse_dates=['timestamp'])
_prod_sim = EnergySim(max_step=7200, power_series=df.production_w.to_list())
_cons_sim = EnergySim(max_step=6000, power_series=df.consumption_w.to_list(), energy_type="consumption")
_batt_sim = BatterySim(
    max_charge_rate=5000,
    max_discharge_rate=5000,
    capacity=10000,
    battery_wear_rate=0,
    current_charge=10000
)
_grid_sim = GridSim(
    feed_in_max=3500,
    feed_in_min=0,
    voltage_max=250,
    voltage_min=230,
    max_taken_from=6000,
    energy_price_sell=0.1 / 1000,
    energy_price_buy=2 / 1000,
    voltage_series=df.grid_voltage.to_list()
)

_inverter_sim = InverterSim(
    prod_sim=_prod_sim,
    cons_sim=_cons_sim,
    batt_sim=_batt_sim,
    grid_sim=_grid_sim,
    timestamps=df.timestamp.to_list(),
)
env = InverterEnv(_inverter_sim, full_period)

# %%
timing_data = {}
# Run one step to measure time
for _ in range(10):
    state, info = env.reset()
    action = 0  # Example action
    for _ in range(40000):
        next_state, reward, done, truncated, _ = env.step(action)
print_all_means()
# %%
