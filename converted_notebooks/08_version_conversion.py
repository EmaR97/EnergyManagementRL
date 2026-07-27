# %%
import torch
from stable_baselines3 import DQN  # Replace with your algorithm

name = 'dqn_1.0_0.06_0.06_0.02_1000_l_2'
model = DQN.load(
    '../logs/%s' % name
)
policy_weights = model.policy.state_dict()
torch.save(policy_weights, f"{name}.policy_weights.pth")

# %%
import torch
from stable_baselines3 import DQN

# Load weights and hyperparameters
policy_weights = torch.load("policy_weights.pth")

# Recreate and load the model
model = DQN("my_policy", env)
model.policy.load_state_dict(policy_weights)
model.save("dqn_1.0_0.05_200_b" + "compatibility")
