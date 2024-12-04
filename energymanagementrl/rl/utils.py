from matplotlib import pyplot as plt
import pandas as pd


def test_plot(env, model, steps, to_show=None):
    state_history = []
    obs, _ = env.reset()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        state_history.append(env.get_state_dict())
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
    state_df.batt_stored = state_df.batt_stored / 10
    state_df.reward = state_df.reward * 10
    if not to_show:
        to_show = env.state_names

    state_df = state_df[to_show]
    plt.figure(figsize=(12, 8))
    for column in state_df.columns:
        plt.plot(state_df[column], label=column)
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.title('State Evolution over Time Steps')
    plt.show()
