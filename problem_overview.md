## **System Overview**

I have a photovoltaic (PV) system with storage and aim to control it to maximize efficiency and cost-effectiveness. The system’s production is automatically limited by the connected load, meaning production cannot exceed the current load. The load consists of three components: household consumption, battery charging, and energy fed to the external grid.

The system operates in two main modes, each prioritizing different load components, with household consumption always having the highest priority due to its cost implications:

1. **Maximize Self-Consumption Mode**: Prioritizes charging the battery over feeding energy to the grid. Any surplus production after charging the battery is directed to the grid.
2. **Fully Feed to Grid Mode**: Prioritizes the grid. If production alone is insufficient, the battery discharges to meet the grid's requirements.

---

### Cost Dynamics

When the system (production + battery) cannot meet household energy needs, electricity must be imported from the grid at a cost approximately four times higher than the price paid for energy fed into the grid. This makes it crucial to prioritize internal consumption over selling energy. At the same time, optimizing the balance between storing energy and feeding it into the grid can provide additional income.

---

### The Challenge

The grid has a limited acceptance capacity, preventing me from feeding all excess production into it whenever I want. Ideally, I would charge the battery first and then feed any remaining daily production into the grid. However, once the battery is full and household consumption is low, production drops to match the grid’s limited acceptance capacity.

To manage this, I manually switch to the second mode, allowing the battery to charge slowly while feeding some energy to the grid throughout the day. However, when production decreases (e.g., at night or during weather changes), the system discharges the battery to meet the grid's requirements, which I want to avoid. I need to preserve the battery's charge for nighttime usage. Any leftover charge at the end of the night, if not needed, can be fed into the grid in the second mode to make space for the next day's production.

---

### My Goal

I aim to optimize production and minimize the risk of running out of battery charge at night while ensuring cost-effective operation. This involves carefully managing mode switching during transitions such as day-to-night, night-to-day, or weather-induced reductions in production.

To automate this process and achieve optimal performance, I am experimenting with a reinforcement learning (RL) model to dynamically switch between the two modes based on the system’s state.
