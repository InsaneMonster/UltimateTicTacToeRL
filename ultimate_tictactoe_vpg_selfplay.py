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
from usienarl.po_models import VanillaPolicyGradient

# Import required src

from src.vpg_ultimate_tictactoe_agent import VPGUltimateTicTacToeAgent
from src.ultimate_tictactoe_experiment import UltimateTicTacToeExperiment
from src.ultimate_tictactoe_environment_selfplay import UltimateTicTacToeEnvironmentSelfPlay, Player
from src.ultimate_tictactoe_pass_through_interface import UltimateTicTacToePassThroughInterface

# Define utility functions to run the experiment


def _define_vpg_model(config: Config) -> VanillaPolicyGradient:
    # Define attributes
    learning_rate_policy: float = 0.00003
    learning_rate_advantage: float = 0.0001
    discount_factor: float = 0.99
    value_steps_per_update: int = 80
    lambda_parameter: float = 0.95
    # Return the _model
    return VanillaPolicyGradient("model", discount_factor,
                                 learning_rate_policy, learning_rate_advantage,
                                 value_steps_per_update, config, lambda_parameter)


def _define_agent(model: VanillaPolicyGradient) -> VPGUltimateTicTacToeAgent:
    # Define attributes
    updates_per_training_volley: int = 20
    # Return the agent
    return VPGUltimateTicTacToeAgent("vpg_agent", model, updates_per_training_volley)


if __name__ == "__main__":
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense,
                               [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense,
                               [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense,
                               [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense,
                               [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    # Define model
    inner_model: VanillaPolicyGradient = _define_vpg_model(nn_config)
    # Define agent
    vpg_agent: VPGUltimateTicTacToeAgent = _define_agent(inner_model)
    # Ultimate Tic Tac Toe self play environment:
    #       - success threshold to consider both the training completed and the experiment successful is just the number of training episodes (a too high number) of match won by the agent (depending on reward assigned)
    environment_name: str = 'UltimateTicTacToeSelfPlay'
    # Generate Ultimate Tic Tac Toe environment with self play environment player and using the O player as the environment player with only low reward type
    environment: UltimateTicTacToeEnvironmentSelfPlay = UltimateTicTacToeEnvironmentSelfPlay(environment_name, Player.o,
                                                                                             1.0, -0.1, 0.0, vpg_agent)

    # Define interface
    interface: UltimateTicTacToePassThroughInterface = UltimateTicTacToePassThroughInterface(environment)
    # Define experiments
    success_threshold: float = 100.0
    experiment: UltimateTicTacToeExperiment = UltimateTicTacToeExperiment("experiment", success_threshold,
                                                                          environment,
                                                                          vpg_agent, interface)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    training_episodes: int = 1000
    validation_episodes: int = 1000
    max_training_episodes: int = 50000
    episode_length_max: int = 100
    # Run epsilon greedy experiment
    run_experiment(experiment,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number)


