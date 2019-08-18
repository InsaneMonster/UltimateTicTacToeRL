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

    def _is_validated(self,
                      average_validation_total_reward: float, average_validation_average_reward: float,
                      average_training_total_reward: float, average_training_average_reward: float) -> bool:
        # Check if average validation reward (score) is over validation threshold
        if average_validation_total_reward >= self._validation_threshold:
            return True
        return False

    def _is_successful(self,
                       average_test_total_reward: float, average_test_average_reward: float,
                       max_test_total_reward: float, max_test_average_reward: float,
                       average_validation_total_reward: float, average_validation_average_reward: float,
                       average_training_total_reward: float, average_training_average_reward: float) -> bool:
        # Check if last validation reward (score) was above threshold
        if average_validation_total_reward >= self._validation_threshold:
            return True
        return False
