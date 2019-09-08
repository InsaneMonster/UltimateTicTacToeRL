#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# UltimateTicTacToeRL project is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import tensorflow
import logging
import os

# Import usienarl

from usienarl import Config, LayerType, run_experiment, command_line_parse
from usienarl.td_models import DuelingDeepQLearning
from usienarl.exploration_policies import EpsilonGreedyExplorationPolicy, BoltzmannExplorationPolicy

# Import required src

from src.dddql_ultimate_tictactoe_agent import DDDQLUltimateTicTacToeAgent
from src.ultimate_tictactoe_experiment import UltimateTicTacToeExperiment
from src.ultimate_tictactoe_environment_random import UltimateTicTacToeEnvironmentRandom, Player
from src.ultimate_tictactoe_pass_through_interface import UltimateTicTacToePassThroughInterface

# Define utility functions to run the experiment


def _define_dddqn_model(config: Config) -> DuelingDeepQLearning:
    # Define attributes
    learning_rate: float = 0.0000001
    discount_factor: float = 0.99
    buffer_capacity: int = 100000
    minimum_sample_probability: float = 0.01
    random_sample_trade_off: float = 0.6
    importance_sampling_value_increment: float = 0.4
    importance_sampling_value: float = 0.001
    error_clipping: bool = True
    # Return the model
    return DuelingDeepQLearning("model",
                                learning_rate, discount_factor,
                                buffer_capacity,
                                minimum_sample_probability, random_sample_trade_off,
                                importance_sampling_value, importance_sampling_value_increment,
                                config, error_clipping)


def _define_epsilon_greedy_exploration_policy() -> EpsilonGreedyExplorationPolicy:
    # Define attributes
    exploration_rate_max: float = 1.0
    exploration_rate_min: float = 0.1
    exploration_rate_decay: float = 0.000002
    # Return the explorer
    return EpsilonGreedyExplorationPolicy(exploration_rate_max, exploration_rate_min, exploration_rate_decay)


def _define_boltzmann_exploration_policy() -> BoltzmannExplorationPolicy:
    # Define attributes
    temperature_max: float = 1.0
    temperature_min: float = 0.1
    temperature_decay: float = 0.000002
    # Return the explorer
    return BoltzmannExplorationPolicy(temperature_max, temperature_min, temperature_decay)


def _define_epsilon_greedy_agent(model: DuelingDeepQLearning, exploration_policy: EpsilonGreedyExplorationPolicy) -> DDDQLUltimateTicTacToeAgent:
    # Define attributes
    weight_copy_step_interval: int = 500
    batch_size: int = 150
    # Return the agent
    return DDDQLUltimateTicTacToeAgent("dddqn_egreedy_agent", model, exploration_policy, weight_copy_step_interval, batch_size)


def _define_boltzmann_agent(model: DuelingDeepQLearning, exploration_policy: BoltzmannExplorationPolicy) -> DDDQLUltimateTicTacToeAgent:
    # Define attributes
    weight_copy_step_interval: int = 500
    batch_size: int = 150
    # Return the agent
    return DDDQLUltimateTicTacToeAgent("dddqn_boltzmann_agent", model, exploration_policy, weight_copy_step_interval, batch_size)


if __name__ == "__main__":
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Ultimate Tic Tac Toe random environment:
    #       - success threshold to consider both the training completed and the experiment successful is around 95% of match won by the agent (depending on reward assigned)
    environment_name: str = 'UltimateTicTacToeRandom'
    # Generate Ultimate Tic Tac Toe environment with random environment player and using the O player as the environment player with only low reward type
    environment: UltimateTicTacToeEnvironmentRandom = UltimateTicTacToeEnvironmentRandom(environment_name, Player.o,
                                                                                         1.0, -0.1, 0.0)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    # Define model
    inner_model: DuelingDeepQLearning = _define_dddqn_model(nn_config)
    # Define exploration_policies
    epsilon_greedy_exploration_policy: EpsilonGreedyExplorationPolicy = _define_epsilon_greedy_exploration_policy()
    boltzmann_exploration_policy: BoltzmannExplorationPolicy = _define_boltzmann_exploration_policy()
    # Define agents
    dddqn_epsilon_greedy_agent: DDDQLUltimateTicTacToeAgent = _define_epsilon_greedy_agent(inner_model, epsilon_greedy_exploration_policy)
    dddqn_boltzmann_agent: DDDQLUltimateTicTacToeAgent = _define_boltzmann_agent(inner_model, boltzmann_exploration_policy)
    # Define interface
    interface: UltimateTicTacToePassThroughInterface = UltimateTicTacToePassThroughInterface(environment)
    # Define experiments
    success_threshold: float = 0.95
    experiment_egreedy: UltimateTicTacToeExperiment = UltimateTicTacToeExperiment("eg_experiment", success_threshold,
                                                                                  environment,
                                                                                  dddqn_epsilon_greedy_agent, interface)
    experiment_boltzmann: UltimateTicTacToeExperiment = UltimateTicTacToeExperiment("b_experiment", success_threshold,
                                                                                    environment,
                                                                                    dddqn_boltzmann_agent, interface)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    training_episodes: int = 1000
    validation_episodes: int = 100
    max_training_episodes: int = 10000
    episode_length_max: int = 100
    # Run epsilon greedy experiment
    run_experiment(experiment_egreedy,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number)
    # Run boltzmann experiment
    run_experiment(experiment_boltzmann,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number)


