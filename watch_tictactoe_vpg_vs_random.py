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

from usienarl import Config, LayerType, watch_experiment, command_line_parse
from usienarl.po_models import VanillaPolicyGradient

# Import required src

from src.vpg_ultimate_tictactoe_agent import VPGUltimateTicTacToeAgent
from src.ultimate_tictactoe_experiment import UltimateTicTacToeExperiment
from src.ultimate_tictactoe_environment_random import UltimateTicTacToeEnvironmentRandom, Player
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
    checkpoint_path, iteration_number, cuda_devices, render = command_line_parse(True)
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Ultimate Tic Tac Toe random environment:
    #       - success threshold to consider both the training completed and the experiment successful is around 82,5% of match won by the agent (depending on reward assigned)
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
    inner_model: VanillaPolicyGradient = _define_vpg_model(nn_config)
    # Define agent
    vpg_agent: VPGUltimateTicTacToeAgent = _define_agent(inner_model)
    # Define interface
    interface: UltimateTicTacToePassThroughInterface = UltimateTicTacToePassThroughInterface(environment)
    # Define experiments
    success_threshold: float = 0.825
    experiment: UltimateTicTacToeExperiment = UltimateTicTacToeExperiment("experiment", success_threshold,
                                                                          environment,
                                                                          vpg_agent, interface)
    # Define experiment data
    testing_episodes: int = 100
    test_cycles: int = 10
    episode_length_max: int = 100
    # Run epsilon greedy experiment for low reward
    watch_experiment(experiment,
                     testing_episodes, test_cycles,
                     episode_length_max,
                     render,
                     logger, checkpoint_path)


