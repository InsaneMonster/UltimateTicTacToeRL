Ultimate Tic-Tac-Toe Reinforcement Learning
*******************************************

A collection of experiments with different approaches using Q-Learning with function approximator at solving an Ultimate Tic-Tac-Toe environment.

The framework used is `USienaRL <https://github.com/InsaneMonster/USienaRL>`_.

Compatible with **USienaRL 0.4.3.**

*Authors*: Luca Pasqualini, Valentino Delle Rose

**Installation**

- Clone the git repository
- Generate a virtual environment using Python 3.6 (for example using Anaconda)
- Install tensorflow or tensorflow-gpu (advised version 1.10.0) using pip if a GPU with CUDA support is available
- Install usienarl using pip

**Note**

To watch the already trained models, always check if the path specified inside the checkpoint file (under metagraph directory) points towards the actual metagraph location.
Since it saves absolute paths it is likely to not be the case.