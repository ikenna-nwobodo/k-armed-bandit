# K-Armed Bandit
This repository contains an implementation of various exploration strategies for the k-armed bandit problem in stationary and non-stationary settings. The simulation framework is built using NumPy and includes comprehensive visualizations and tuning utilities.

### Problem Description
The k-armed bandit problem is a classic reinforcement learning setup where an agent repeatedly chooses between k actions (arms), each providing stochastic rewards. The goal is to maximize the cumulative reward by balancing exploration and exploitation.

This assignment explores multiple agent strategies under different environmental dynamics:

1. Stationary: Fixed reward distributions.

2. Non-Stationary (Drift): Reward distributions gradually drift over time.

3. Non-Stationary (Mean-Reverting): Means follow an autoregressive process.

4. Non-Stationary (Abrupt Change): The true means are suddenly permuted at a specific timestep.

5. Abrupt with Hard Reset: Same as above, but agents are reset at the change point.


### Requirements
Ensure the following libraries are installed:

`pip install numpy matplotlib seaborn tqdm pandas scipy`

### How to Run
**Clone the repository:**

`git clone https://github.com/ikenna-nwobodo/k-armed-bandit.git`

`cd bandit_problem`

**Open the notebook:**

You can run the analysis in any Jupyter-compatible environment (Jupyter Lab, VS Code, Colab):

Run the cells in order.

### Agents
- **Greedy**:	Always exploits the current best action
- **Epsilon-Greedy**:	Explores with probability epsilon
- **Optimistic Greedy**:	Initializes estimates optimistically to encourage exploration initially
- **Gradient Bandit**:	Learns preferences over actions using gradient ascent

### Tuning Strategy
A simple grid search was used to tune epsilon and alpha:

epsilon ∈ [0.01, 0.05, 0.1, 0.2, 0.5]

alpha ∈ [0.01, 0.05, 0.1, 0.2, 0.5]

Each value was evaluated using average rewards and % optimal action over 1000 runs.

### Output
The notebook produces:

1. Violin plots of initial reward distributions

2. Line plots comparing average rewards and optimal action percentages

3. Tables summarizing performance across agents and settings
