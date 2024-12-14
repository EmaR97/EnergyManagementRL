from matplotlib import pyplot as plt
import pandas as pd


def test_plot(env, model, steps, to_show=None, seed=None, ):
    state_history = []
    obs, _ = env.reset(seed, shuffle=0)
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        state = env.get_state_dict()
        state['action'] = action
        state_history.append(state)
        if done:
            break
    state_df = pd.DataFrame(
        state_history
    )

    print(
        f"{sum(state_df.reward):.1f}="
        f"{sum(state_df.reward_energy_sold):.1f}"
        f"-{sum(state_df.penalty_energy_purchase) :.1f}"
        f"-{sum(state_df.penalty_battery_wear):.1f}"
    )
    state_df['batt_sim.stored'] = state_df['batt_sim.stored'] / 10
    state_df.reward = state_df.reward * 10

    state_df = state_df[to_show]
    plt.figure(figsize=(12, 8))
    for column in state_df.columns:
        plt.plot(state_df[column], label=column)
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.title('State Evolution over Time Steps')
    plt.show()


def extract_values_gen(d):
    """
    Extracts all values from a nested dictionary using a generator for better performance.

    :param d: The nested dictionary
    :yield: All values in the dictionary
    """
    for v in d.values():
        if isinstance(v, dict):
            yield from extract_values_gen(v)  # Delegate to nested generator
        else:
            yield v


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary into a single dictionary with composite keys, optimized for performance.

    :param d: The dictionary to flatten
    :param parent_key: The base key (used in recursion)
    :param sep: Separator for composite keys
    :return: Flattened dictionary
    """
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            yield from flatten_dict(v, new_key, sep=sep)  # Delegate to nested generator
        else:
            yield new_key, v
