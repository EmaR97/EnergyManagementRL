# Energy Management Optimization with Reinforcement Learning

This project implements a reinforcement learning (RL) approach to optimize energy management in a facility using
renewable energy, battery storage, and grid interaction. By simulating an environment that includes renewable energy
sources, battery storage, and a grid connection, we aim to maximize the self-consumption of renewable energy, reduce
dependency on the grid, and minimize battery wear. This simulation serves as a controlled environment for training a
reinforcement learning agent, guiding it towards making optimal energy management decisions.

## System Overview

The system models a realistic energy management environment with the following goals:

- **Balance Energy Sources**: Optimize the use of renewable production, battery storage, and grid interactions.
- **Maximize Self-Consumption**: Prioritize using locally produced energy.
- **Minimize Costs**: Reduce expenses associated with grid purchases and battery wear.

This environment is suitable for developing intelligent agents that can learn effective strategies to handle real-time
energy flows, considering both short-term energy needs and long-term sustainability.

### Key Components

#### 1. `EnergySim`

Models energy production and consumption within the environment, providing:

- **Production Simulator (`prod_sim`)**: Models renewable energy generation, like solar power, accounting for
  fluctuations in production.
- **Consumption Simulator (`cons_sim`)**: Represents the energy demand of the facility.

These simulators provide real-time data on energy production and consumption as well as forecasts, allowing the RL agent
to anticipate future energy needs and availability.

#### 2. `BatterySim`

Represents the battery storage system, simulating charging and discharging processes:

- Tracks the battery’s state of charge, charge/discharge rates, capacity, and wear rate.
- Provides essential information for optimizing battery health and determining efficient energy storage and release
  strategies.

#### 3. `GridSim`

Manages grid interactions, including energy purchases and sales:

- Tracks energy fed to and drawn from the grid, respecting grid limits and energy prices.
- Simulates grid costs and limitations, aiding the agent in balancing self-consumption with grid dependency.

#### 4. `InverterSim`

Integrates the production, consumption, battery, and grid simulations, serving as the coordinator for energy flows:

- **Energy Balance Calculation**: Computes the difference between production and consumption.
- **Mode-based Management**: Supports two modes of operation:
    - **Mode A**: Prioritizes on-site consumption, reducing grid dependence.
    - **Mode B**: Focuses on maximizing grid feed-in for sale.

The inverter’s operations also leverage precomputed sine and cosine values to capture daily energy demand patterns.

#### 5. `InverterEnv`

Defines the Gym-compatible environment for reinforcement learning:

- **State Space**: Includes metrics on energy production, consumption, battery and grid states, and time-based features.
- **Action Space**: Enables the agent to choose operational modes (e.g., self-consumption vs. grid feeding).
- **Reward Structure**: Rewards energy sales to the grid while penalizing grid dependency and battery wear, guiding the
  agent to actions that lower costs and increase efficiency.

The environment steps through time, updating the state and computing rewards based on the agent's actions and simulation
feedback.

### System Workflow and Interactions

1. **Initialization**:
    - The system creates instances of `EnergySim`, `BatterySim`, and `GridSim`, representing energy
      production/consumption, battery storage, and grid interactions.
    - An `InverterSim` instance coordinates these components to maintain the energy balance.
    - The `InverterEnv` environment wraps `InverterSim`, providing an interface for the RL agent to interact with.

2. **Simulation Step**:
    - At each timestep, the agent selects an action, such as choosing a self-consumption or grid feed mode.
    - The environment calculates the energy balance, directing any surplus or deficit energy towards the battery or grid
      as per the selected mode:
        - **Mode A**: Prioritizes battery and on-site energy use to reduce grid reliance.
        - **Mode B**: Prioritizes grid feed-in, minimizing battery use and maximizing sales.
    - The state is updated to reflect these decisions, preparing the system for the next timestep.

3. **Reward Calculation**:
    - The reward for each step is calculated based on:
        - Revenue from energy sales to the grid.
        - Penalties for grid purchases.
        - Battery wear costs.
    - This reward structure helps the RL agent learn to reduce costs and maximize returns.

4. **Episode Termination**:
    - The environment runs until it reaches a defined limit (e.g., one week).
    - At the end of an episode, the environment resets, enabling the agent to start a new simulation and continue
      learning.

## Reinforcement Learning Approach

Using a Proximal Policy Optimization (PPO) algorithm, the RL agent is trained to navigate the energy management
environment effectively. Through trial and error, it learns strategies that strike a balance between self-consumption,
cost savings, and battery preservation. The trained model can then be evaluated for performance, offering insights into
its ability to adapt and make optimized energy management decisions over time.

## Challenges and Opportunities

This project addresses several challenges inherent to renewable energy systems:

- **Supply and Demand Variability**: Renewable sources, like solar power, are unpredictable, requiring the system to
  handle fluctuations in both production and demand.
- **Battery Wear and Cost-Benefit Analysis**: The agent must balance the benefits of using stored energy against the
  costs of battery degradation.
- **Real-time Decision Making**: The RL agent learns to make responsive, real-time decisions, enabling it to adapt to
  changing conditions.

