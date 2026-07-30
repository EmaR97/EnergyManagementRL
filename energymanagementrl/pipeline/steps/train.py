import copy
import os
import time

import pandas as pd
import torch

from ..lib.builders import load_and_prepare_data, build_simulation_stack, get_full_period

from ...simulation import week
from ...utility import get_logger

logger = get_logger(__name__)


def run(config: dict):
    logger.info("Starting model training")

    from datetime import datetime
    suffix = datetime.now().strftime("%Y%m%d_%H%M")

    train_cfg = config["training"]

    if train_cfg["device"] == "cuda":
        try:
            if not torch.cuda.is_available():
                logger.warning(
                    "Config requests device='cuda' but CUDA is not available. "
                    "Training will fall back to CPU. "
                    "Set training.device to 'cpu' in config.json to suppress this warning."
                )
        except ImportError:
            logger.warning("PyTorch not found — cannot verify CUDA availability.")

    logger.info(f"Using device: {train_cfg['device']}")

    logs_dir = config["data_paths"].get("logs", "../data/logs")
    os.makedirs(logs_dir, exist_ok=True)

    df = load_and_prepare_data(config)
    df = pd.concat([df, df[-288 * 2:]])
    p_sim, c_sim, b_sim, g_sim, i_sim = build_simulation_stack(config, df)
    full_period = get_full_period(df)

    from ...rl.env import InverterEnvBatteryMgmt

    logger.info("Configuring reward and reserve parameters")
    rewards = train_cfg["rewards"]
    reserves = train_cfg["battery_reserves"]

    logger.info("Creating training environment")
    train_env = InverterEnvBatteryMgmt(
        inverter_sim=copy.deepcopy(i_sim),
        max_steps=week * 2,
        reward_near_full=rewards["reward_near_full"],
        penalty_below_night_reserve=rewards["penalty_below_night_reserve"],
        penalty_below_min_reserve=rewards["penalty_below_min_reserve"],
        min_reserve=reserves["min_reserve"],
        night_reserver=reserves["night_reserver"],
        near_full=reserves["near_full"],
    )
    train_env.shuffle = train_cfg["shuffle"]
    train_env.inverter_sim.grid_sim.energy_price_buy = rewards["energy_price_buy_per_kwh"] / 1000

    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor

    class _LoggingEvalCallback(EvalCallback):
        def _on_step(self) -> bool:
            continue_training = True
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                if self.model.get_vec_normalize_env() is not None:
                    try:
                        from stable_baselines3.common.vec_env import sync_envs_normalization
                        sync_envs_normalization(self.training_env, self.eval_env)
                    except AttributeError as e:
                        raise AssertionError(
                            "Training and eval env are not wrapped the same way, "
                            "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                            "and warning above."
                        ) from e
                self._is_success_buffer = []
                from stable_baselines3.common.evaluation import evaluate_policy
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model, self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )
                if self.log_path is not None:
                    self.evaluations_timesteps.append(self.num_timesteps)
                    self.evaluations_results.append(episode_rewards)
                    self.evaluations_length.append(episode_lengths)
                    kwargs = {}
                    if len(self._is_success_buffer) > 0:
                        self.evaluations_successes.append(self._is_success_buffer)
                        kwargs = dict(successes=self.evaluations_successes)
                    import numpy as np
                    np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                             results=self.evaluations_results,
                             ep_lengths=self.evaluations_length, **kwargs)
                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                self.last_mean_reward = float(mean_reward)
                if self.verbose >= 1:
                    logger.info(
                        f"Eval num_timesteps={self.num_timesteps}, "
                        f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                    )
                    logger.info(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record("eval/mean_ep_length", mean_ep_length)
                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    if self.verbose >= 1:
                        logger.info(f"Success rate: {100 * success_rate:.2f}%")
                    self.logger.record("eval/success_rate", success_rate)
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(self.num_timesteps)
                if mean_reward > self.best_mean_reward:
                    if self.verbose >= 1:
                        logger.info("New best mean reward!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    self.best_mean_reward = float(mean_reward)
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()
                if self.callback is not None:
                    continue_training = continue_training and self._on_event()
            return continue_training

    num_envs = train_cfg["num_envs"]
    logger.info(f"Vectorizing training environment ({num_envs} parallel envs via SubprocVecEnv)")

    def _make_train_env(rank):
        def _init():
            env = copy.deepcopy(train_env)
            env.inverter_sim.reset(seed=rank * 1000 + 1, shuffle=train_cfg["shuffle"])
            return env

        return _init

    t_env = SubprocVecEnv([_make_train_env(i) for i in range(num_envs)])

    algorithm = train_cfg["algorithm"]
    load_path = train_cfg.get("load_path")

    if load_path:
        logger.info(f"Loading model from {load_path} for continued training")
        model_class = DQN if algorithm == "DQN" else PPO
        model = model_class.load(load_path, env=t_env)
    else:
        logger.info(f"Constructing {algorithm} model from scratch")
        if algorithm == "DQN":
            model = DQN(
                train_cfg["policy"],
                t_env,
                verbose=0,
                learning_rate=train_cfg["learning_rate"],
                batch_size=train_cfg["batch_size"],
                gradient_steps=train_cfg["gradient_steps"],
                device=train_cfg["device"],
            )
        elif algorithm == "PPO":
            model = PPO(
                train_cfg["policy"],
                t_env,
                verbose=0,
                learning_rate=train_cfg["learning_rate"],
                batch_size=train_cfg["batch_size"],
                n_steps=train_cfg["n_steps"],
                n_epochs=train_cfg["n_epochs"],
                gae_lambda=train_cfg["gae_lambda"],
                clip_range=train_cfg["clip_range"],
                device=train_cfg["device"],
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    eval_cfg = train_cfg["eval"]
    eval_episode_steps = 288 * eval_cfg["episode_days"]
    eval_num_envs = eval_cfg["num_envs"]

    logger.info(f"Creating evaluation environment ({eval_num_envs} parallel envs x {eval_episode_steps} steps)")

    def _make_eval_env(rank):
        def _init():
            env = InverterEnvBatteryMgmt(
                inverter_sim=copy.deepcopy(i_sim),
                max_steps=eval_episode_steps,
                reward_near_full=rewards["reward_near_full"],
                penalty_below_night_reserve=rewards["penalty_below_night_reserve"],
                penalty_below_min_reserve=rewards["penalty_below_min_reserve"],
                min_reserve=reserves["min_reserve"],
                night_reserver=reserves["night_reserver"],
                near_full=reserves["near_full"],
            )
            env.shuffle = train_cfg["shuffle"]
            env.inverter_sim.reset(seed=rank * 1000 + 100, shuffle=train_cfg["shuffle"])
            env.inverter_sim.grid_sim.energy_price_buy = rewards["energy_price_buy_per_kwh"] / 1000
            return Monitor(env)

        return _init

    eval_env_vec = SubprocVecEnv([_make_eval_env(i) for i in range(eval_num_envs)])

    logger.info("Setting up callbacks")
    eval_callback = _LoggingEvalCallback(
        eval_env=eval_env_vec,
        best_model_save_path=os.path.join(logs_dir, "best_model"),
        log_path=os.path.join(logs_dir, "results"),
        eval_freq=week * eval_cfg["eval_freq_multiplier"],
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["checkpoint"]["save_freq"],
        save_path=os.path.join(logs_dir, "checkpoints"),
        name_prefix=train_cfg["checkpoint"]["name_prefix"],
        save_replay_buffer=train_cfg["checkpoint"]["save_replay_buffer"],
        save_vecnormalize=train_cfg["checkpoint"]["save_vecnormalize"],
    )

    total_timesteps = full_period * train_cfg["train_periods"]
    logger.info(f"Training for {total_timesteps} timesteps")
    logger.info("Starting model.learn()")

    start = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])
    except Exception:
        logger.warning("Training crashed — saving crashed model before re-raising")
        model.save(os.path.join(logs_dir, "crashed_model"))
        raise
    finally:
        elapsed = time.time() - start
        logger.info(f"Training completed in {elapsed:.1f}s")
        model.save(os.path.join(logs_dir, "final_model"))
        logger.info(f"Model saved to {logs_dir}/final_model")
        model.save(os.path.join(logs_dir, f"final_model.{suffix}"))
        logger.info(f"Model saved to {logs_dir}/final_model.{suffix}")
