import os

import pandas as pd

from ..lib.builders import load_and_prepare_data, build_simulation_stack, get_full_period

from ...utility import get_logger

logger = get_logger(__name__)


def run(config: dict):
    logger.info("Starting model evaluation")

    logs_dir = config["data_paths"].get("logs", "../data/logs")
    trained_dir = config["data_paths"].get("trained_models", "../data/trained_models")

    df = load_and_prepare_data(config)
    _, _, _, _, i_sim = build_simulation_stack(config, df)
    full_period = get_full_period(df)

    from ...rl.env import InverterEnv
    from ...rl.models import GreedyModel, ConservativeModel
    env = InverterEnv(i_sim, full_period)

    results = {}
    for name, model_fn in [
        ("greedy", lambda: GreedyModel()),
        ("conservative", lambda: ConservativeModel()),
    ]:
        model = model_fn()
        score = _evaluate_model(env, model, full_period, name)
        results[name] = score

    final_path = os.path.join(logs_dir, "final_model")
    if os.path.exists(f"{final_path}.zip"):
        model = _load_rl_model(final_path, env)
        results["final"] = _evaluate_model(env, model, full_period, "final")

    best_path = os.path.join(logs_dir, "best_model", "best_model")
    if os.path.exists(f"{best_path}.zip"):
        model = _load_rl_model(best_path, env)
        results["best"] = _evaluate_model(env, model, full_period, "best")

    current_model_path = config.get("models", {}).get("current_used")
    if current_model_path:
        full_path = os.path.join(trained_dir, "models", current_model_path)
        if os.path.exists(full_path):
            from ...rl.utils import load_model_with_weights

            model = load_model_with_weights(env, full_path)
            results["current_used"] = _evaluate_model(env, model, full_period, "current_used")

    results_df = pd.DataFrame([{"model": k, "score": v} for k, v in results.items()])
    results_path = os.path.join(trained_dir, "evaluation_results.csv")
    os.makedirs(trained_dir, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Results:\n{results_df.to_string()}")


def _load_rl_model(path, env):
    from stable_baselines3 import DQN, PPO

    try:
        return DQN.load(path, env=env)
    except Exception:
        return PPO.load(path, env=env)


def _evaluate_model(env, model, max_steps, name):
    from ...rl.utils import test_plot

    env.set_max_steps(max_steps)
    score = test_plot(env, model, max_steps, to_show=["batt_sim.stored"], seed=33)
    logger.info(f"{name}: {score}")
    return score
