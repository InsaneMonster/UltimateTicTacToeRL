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

import logging

# Import src

from src.ultimate_tictactoe_environment import UltimateTicTacToeEnvironment, Player


class UltimateTicTacToeEnvironmentRandom(UltimateTicTacToeEnvironment):
    """
    Ultimate Tic-Tac-Toe environment in which the environment player plays with a random policy.
    """

    def __init__(self,
                 name: str,
                 environment_player: Player,
                 agent_player_win_reward: float,
                 environment_player_win_reward: float,
                 draw_reward: float):
        # Generate the base tic tac toe environment
        super(UltimateTicTacToeEnvironmentRandom, self).__init__(name, environment_player, agent_player_win_reward, environment_player_win_reward, draw_reward)

    def get_environment_player_first_action(self,
                                            logger: logging.Logger,
                                            session) -> int:
        # Just return a random action
        return self.get_random_action(logger, session)

    def get_environment_player_action(self,
                                      logger: logging.Logger,
                                      session) -> int:
        return self.get_random_action(logger, session)
