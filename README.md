# EnergyManagementRL

### Project Overview
**EnergyManagementRL** is a simulation-based project focused on optimizing energy usage in a system with renewable energy sources, specifically solar panels, a battery storage unit, and a connection to the electrical grid. Leveraging reinforcement learning (RL), this project aims to create a control strategy that maximizes self-consumption of solar energy, minimizes reliance on the grid, and optimizes battery usage.

### Key Objectives
- **Maximize Self-Consumption**: Optimize the usage of locally produced solar energy to reduce grid dependency.
- **Efficient Battery Management**: Develop a strategy for charging and discharging the battery that aligns with consumption forecasts and solar production.
- **Grid Interaction**: Minimize energy imports from the grid while also considering feed-in opportunities based on projected production and consumption.

### Features
- **Battery and Grid Simulation**: Models that simulate battery storage, charging/discharging dynamics, and grid interactions.
- **Forecast-Based Decision Making**: Uses short-term consumption forecasts and solar production forecasts to inform RL decisions.
- **Data Collection and Integration**: Collects and processes real-time and historical data for enhanced simulation accuracy.
- **Reinforcement Learning (RL)**: Implements RL techniques for control decisions in a simulation environment, using states like time of day, battery level, and forecasted data.

### Model Inputs
The model utilizes multiple input features to train the RL agent effectively:
- **Consumption Forecasts**: Last 24 hours and upcoming 24 hours at 5-minute intervals.
- **Battery State**: Current charge level and potential charge/discharge capacities.
- **Solar Production Forecasts**: Forecasts for the next 72 hours at 15-minute intervals, capturing the variability in production.
- **Grid Capacity**: Forecasted grid acceptance or feed-in potential for the next 72 hours, resampled as needed.

### Reinforcement Learning Approach
The RL agent is designed to control actions like:
1. Charging or discharging the battery based on forecasted energy production and consumption.
2. Feeding excess energy into the grid when battery storage is at capacity.
3. Drawing from the grid when solar production is low, and battery storage is insufficient.

By simulating different scenarios and refining the agent's policy, the project aims to develop a reliable, efficient control model adaptable to various energy production and consumption patterns.

### Installation
To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/YourUsername/EnergyManagementRL.git
cd EnergyManagementRL
pip install -r requirements.txt
```

### Usage
1. **Data Preparation**: Load your historical data for consumption, solar production, and grid availability into the `data/` directory.
2. **Train the Model**: Run the training script to start training the RL agent. This may require GPU resources for optimal performance.
   ```bash
   python train.py
   ```
3. **Run Simulations**: Use the `simulate.py` script to evaluate the model in various scenarios and analyze performance metrics.
   ```bash
   python simulate.py
   ```

### Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Make sure to include tests for any new features or updates.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
