# %%
# !pip -q install git+https://github.com/EmaR97/EnergyManagementRL.git@testing-7
# !pip -q install  fusion_solar_py pvlib retry-requests openmeteo_requests requests-cache
# dataset_folder = '/kaggle/working'

import sys

sys.path.append("..")
dataset_folder = '../data'

# %%
import os.path

dataset_simulation_inputs = 'emanuelerapisarda/simulation-inputs'  # Replace with the dataset you want
dataset_trained_models = 'emanuelerapisarda/reinforcement-learning-inverter-trained-models'  # Replace with the dataset you want
simulation_inputs_folder = os.path.join(dataset_folder, 'simulation_inputs')
trained_models_folder = os.path.join(dataset_folder, 'trained_models')
logs_folder = os.path.join(dataset_folder, 'logs')

best_model_save_path = os.path.join(logs_folder, 'best_model')
best_model_simple_save_path = os.path.join(logs_folder, 'best_model_simple')
best_model_path = os.path.join(best_model_save_path, 'best_model')
best_model_simple_path = os.path.join(best_model_simple_save_path, 'best_model')
results_folder = os.path.join(logs_folder, 'results')

latest_model_path = os.path.join(logs_folder, 'ppo_inverter')
input_series = os.path.join(simulation_inputs_folder, 'complete_series.14panels.csv')
results_file = os.path.join(trained_models_folder, 'result_metadata.csv')
new_model_version = os.path.join(trained_models_folder, 'models')

current_used_model = os.path.join(new_model_version, 'dqn_1.0_0.06_0.06_0.02_1000_l_2.policy_weights.pth')

# %%
# from kaggle_secrets import UserSecretsClient
#
# # Retrieve the Kaggle credentials securely
# user_secrets = UserSecretsClient()
# os.environ['KAGGLE_USERNAME'] = user_secrets.get_secret("KAGGLE_USERNAME")
# os.environ['KAGGLE_KEY'] = user_secrets.get_secret("KAGGLE_KEY")
from dotenv import load_dotenv

load_dotenv()
# %%
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate with Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset
api.dataset_download_files(dataset_simulation_inputs, path=simulation_inputs_folder, unzip=True)
api.dataset_download_files(dataset_trained_models, path=trained_models_folder, unzip=True)
api.dataset_metadata(dataset_trained_models, path=trained_models_folder)
api.dataset_metadata(dataset_simulation_inputs, path=simulation_inputs_folder)

# %%
import logging

import pandas as pd
from stable_baselines3 import DQN
import time
import copy

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from energymanagementrl.simulation import *
from energymanagementrl.rl import *
from stable_baselines3.common.env_checker import check_env
import torch

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# %%
# Load data
start_time = time.time()
df = pd.read_csv(input_series, parse_dates=['index'], index_col=['index'])
df.rename(columns={'GRID_VOLTAGE': 'grid_voltage'}, inplace=True)
df['production_power_kw_altered'] = np.where(df['SOC'] < 100, df['production_power_kw'],
                                             df['production_power_kw_weather_dependent'])
df = df[df.index > pd.Timestamp("2024-10-19")]
df = pd.concat([df, df[-288 * 2:]])
production_w_weather_dependent = df.production_power_kw_weather_dependent * 1000
production_w = df.production_power_kw_altered * 1000
optimal_production_w = df.production_power_kw_optimal * 1000
consumption_w = -df.load_power_kw * 1000
grid_voltage = df.grid_voltage
full_period = min(288 * 7 * 4 * 9, len(df) - 288 * 2)
print(f"Data loading took {time.time() - start_time:.2f} seconds")

# Initialize EnergySim
start_time = time.time()
p_sim = ProductionSimFromReal(
    power_series=production_w,
    optimal_power_series=optimal_production_w,
    weather_power_series=production_w_weather_dependent,
    forecast_steps=48,
)
print(f"ProductionSim initialization took {time.time() - start_time:.2f} seconds")

# Initialize ConsumptionSim
start_time = time.time()
c_sim = ConsumptionSim(
    power_series=consumption_w,
    daily_sample=6,
    forecast_steps=12
)
print(f"ConsumptionSim initialization took {time.time() - start_time:.2f} seconds")

# Initialize BatterySim
start_time = time.time()
b_sim = BatterySim(
    max_charge_rate=5000,
    max_discharge_rate=5000,
    capacity=9000,
    battery_wear_rate=0,
)
print(f"BatterySim initialization took {time.time() - start_time:.2f} seconds")

# Initialize GridSim
start_time = time.time()
g_sim = GridSim(
    feed_in_max=3500,
    feed_in_min=0,
    voltage_max=250,
    voltage_min=230,
    max_taken_from=6000,
    energy_price_sell=.1 / 1000,
    energy_price_buy=.4 / 1000,
    voltage_series=grid_voltage
)
print(f"GridSim initialization took {time.time() - start_time:.2f} seconds")

# Initialize InverterSim
start_time = time.time()
i_sim = InverterSim(
    prod_sim=p_sim,
    cons_sim=c_sim,
    batt_sim=b_sim,
    grid_sim=g_sim,
)
print(f"InverterSim initialization took {time.time() - start_time:.2f} seconds")

# Initialize Environment
start_time = time.time()
env = InverterEnv(i_sim, week)
print(f"InverterEnv initialization took {time.time() - start_time:.2f} seconds")
# p_sim.max_shift = 4
# p_sim.mix_probability = .5
check_env(env)
# %%
env.state_size
# %%
to_show = [
    # 'prod_sim.energy',
    # 'cons_sim.energy',
    'batt_sim.stored',
    # 'batt_sim.charge_rate',
    # 'batt_sim.discharge_rate',
    # 'grid_sim.feed_to_grid',
    # 'grid_sim.taken_from_grid',
    # 'reward_energy_sold',
    # 'penalty_energy_purchase',
    # 'penalty_battery_wear',
    # 'reward',
    # 'action'
]


def _test_plot(_env: InverterEnv, _model, _max_steps):
    return test_plot(_env, _model, _max_steps, to_show, 33)


env.set_max_steps(full_period)
for model_to_test in [
    ConservativeModel(),
    GreedyModel()
]:
    _test_plot(
        env,
        model_to_test,
        # env.max_steps,
        # week,
        full_period,
        # month,
        # day,
    )
env.set_max_steps(week)
# %%
train_energy_price_buy_kw = 1.
reward_near_full = .06
penalty_below_night_reserve = .06
penalty_below_min_reserve = .02
train_periods = 1000
min_reserve = .1
night_reserver = .85
near_full = .98
num_envs = 20

i_sim.reset()
train_env = InverterEnvBatteryMgmt(
    inverter_sim=copy.deepcopy(i_sim),
    max_steps=week * 2,
    reward_near_full=reward_near_full,
    penalty_below_night_reserve=penalty_below_night_reserve,
    penalty_below_min_reserve=penalty_below_min_reserve,
    min_reserve=min_reserve,
    night_reserver=night_reserver,
    near_full=near_full
)
train_env.inverter_sim.grid_sim.energy_price_buy = train_energy_price_buy_kw / 1000
train_env.shuffle = 2

eval_env = InverterEnvBatteryMgmt(
    inverter_sim=copy.deepcopy(i_sim),
    max_steps=full_period,
    reward_near_full=reward_near_full,
    penalty_below_night_reserve=penalty_below_night_reserve,
    penalty_below_min_reserve=penalty_below_min_reserve,
    min_reserve=min_reserve,
    night_reserver=night_reserver,
    near_full=near_full
)
eval_env.shuffle = 2
eval_env.inverter_sim.grid_sim.energy_price_buy = train_energy_price_buy_kw / 1000

eval_callback_base = EvalCallback(
    eval_env=eval_env,
    best_model_save_path=best_model_save_path,
    log_path=results_folder,
    eval_freq=week * 10
)

env.inverter_sim.grid_sim.energy_price_buy = .4 / 1000
env.set_max_steps(full_period)
eval_callback_simple = EvalCallback(
    eval_env=env,
    best_model_save_path=best_model_simple_save_path,
    log_path=results_folder,
    eval_freq=week * 10
)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=latest_model_path,
    name_prefix="latest_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)
# %%
def run_training(
        _train_env,
        total_timesteps,
        eval_callback,
        _latest_model_path,
        model_class=DQN,
        _num_envs=10,
        to_load_path=None,
):
    try:

        t_env = DummyVecEnv(
            [
                lambda: copy.deepcopy(train_env)
                for _ in range(num_envs)
            ],
        )
    except Exception as e:
        raise ValueError("Failed to initialize training environment.") from e

    # Load or create model
    if to_load_path:
        try:
            _model = model_class.load(to_load_path, env=t_env)
            print(f"Loaded model from {to_load_path}")
        except Exception as e:
            raise ValueError(f"Failed to load model from {to_load_path}.") from e
    else:
        _model = model_class(
            "MlpPolicy",
            t_env,
            verbose=0,
            learning_rate=0.0003,
            batch_size=2048,
            device="cuda",  # Ensures the model uses GPU if available
            # exploration_fraction=0.1,  # Increase this for a longer exploration period
            # exploration_final_eps=0.2,  # Set higher for more exploration after annealing
            # exploration_initial_eps=1.0,
        )

    try:
        _model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
        )

    except Exception:
        print("⚠️ Training crashed, saving latest model...")
        _model.save(f"{latest_model_path}/crashed_model")
        raise

    finally:
        # Always save final/latest version
        _model.save(f"{latest_model_path}/final_model")
        print(f"✅ Model saved to {latest_model_path}/final_model")

# %%
run_training(
    _train_env=train_env,
    total_timesteps=full_period * train_periods,
    _latest_model_path=latest_model_path,
    eval_callback=[eval_callback_base, eval_callback_simple, checkpoint_callback],
    _num_envs=num_envs,
    # to_load_path='../logs/ppo_inverter.conservative',
    # to_load_path=latest_model_path,
    # to_load_path=best_model_path,
)
# %%
# Load models
models = {
    # "latest": DQN.load(latest_model_path, env=env),
    "best": DQN.load(best_model_path, env=env),
    "best_simple": DQN.load(best_model_simple_path, env=env),
}

# Test and evaluate models
scores = {}
for model_name, model in models.items():
    scores[model_name] = _test_plot(env, model, full_period)

# Evaluate additional models
additional_models = [
    # Uncomment to test specific models
    # ConservativeModel(),
    # GreedyModel(),
    # DQN.load('../logs/ppo_inverter.1.2.b', env=env),
    # DQN.load('../logs/dqn_1.0_0.06_200_l', env=env),
    # DQN.load('../logs/dqn_1.0_0.06_0.06_0.02_400_l', env=env),
    # DQN.load('../logs/dqn_1.0_0.06_0.06_0.02_1000_l', env=env),
    load_model_with_weights(
        weight_path=current_used_model, env=env),
]
for model in additional_models:
    _test_plot(env, model, full_period)
# %%

# Load training results or initialize a new DataFrame
try:
    training_results_df = pd.read_csv(results_file)
except FileNotFoundError:
    training_results_df = pd.DataFrame()

# Prepare data for saving results
timestamp = pd.Timestamp.now().strftime('%Y_%m_%d_%H_%M_%S')
results_to_add = []

for model_type, model in models.items():
    score = scores[model_type]
    save_path = os.path.join(new_model_version, f"dqn.{timestamp}.{model_type}")
    torch.save(model.policy.state_dict(), save_path)

    results_to_add.append({
        "type": model_type,
        "score": score,
        "train_energy_price_buy_kw": train_energy_price_buy_kw,
        "reward_near_full": reward_near_full,
        "penalty_below_night_reserve": penalty_below_night_reserve,
        "penalty_below_min_reserve": penalty_below_min_reserve,
        "train_periods": train_periods,
        "min_reserve": min_reserve,
        "night_reserver": night_reserver,
        "near_full": near_full,
        "timestamp": timestamp,
    })

# Update and save results
training_results_df = pd.concat([training_results_df, pd.DataFrame(results_to_add)], ignore_index=True)
training_results_df.to_csv(results_file, index=False)
# %%
import json


def update_metadata(file_name: str, dataset_id: str, output_dir: str):
    try:
        with open(os.path.join(output_dir, file_name), "r") as f:
            metadata = json.load(f)
        # If it's a stringified JSON, parse it; otherwise leave it as-is.
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        metadata["id"] = dataset_id
        with open(os.path.join(output_dir, file_name), "w") as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"Metadata update error: {e}")


update_metadata('dataset-metadata.json', dataset_trained_models, trained_models_folder)
# %%
api.dataset_create_version(trained_models_folder, "Added new models and metadata", delete_old_versions=False,
                           dir_mode='zip')
