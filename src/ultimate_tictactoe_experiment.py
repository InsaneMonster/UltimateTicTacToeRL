# Import packages

import logging

# Import usienarl

from usienarl import Experiment, Agent, Interface

# Import required src

from src.ultimate_tictactoe_environment import UltimateTicTacToeEnvironment


class UltimateTicTacToeExperiment(Experiment):
    """
    Ultimate Tic Tac Toe Experiment which is both validated and passed when the validation average total reward is above
    the given threshold.
    """

    def __init__(self,
                 name: str,
                 validation_threshold: float,
                 environment: UltimateTicTacToeEnvironment,
                 agent: Agent,
                 interface: Interface):
        # Define benchmark experiment attributes
        self._validation_threshold: float = validation_threshold
        # Generate the base experiment
        super(UltimateTicTacToeExperiment, self).__init__(name, environment, agent, interface)

    def _is_validated(self, logger: logging.Logger, last_average_validation_total_reward: float,
                      last_average_validation_scaled_reward: float, last_average_training_total_reward: float,
                      last_average_training_scaled_reward: float, last_validation_volley_rewards: [],
                      last_training_volley_rewards: []) -> bool:
        # Check if average validation reward (score) is over validation threshold
        if last_average_validation_total_reward >= self._validation_threshold:
            return True
        return False

    def _is_successful(self, logger: logging.Logger, average_test_total_reward: float,
                       average_test_scaled_reward: float, max_test_total_reward: float, max_test_scaled_reward: float,
                       last_average_validation_total_reward: float, last_average_validation_scaled_reward: float,
                       last_average_training_total_reward: float, last_average_training_scaled_reward: float,
                       test_cycles_rewards: [], last_validation_volley_rewards: [],
                       last_training_volley_rewards: []) -> bool:
        # Check if last validation reward (score) was above threshold
        if last_average_validation_total_reward >= self._validation_threshold:
            return True
        return False
