# Import packages

import logging

# Import src

from src.ultimate_tictactoe_environment import UltimateTicTacToeEnvironment, Player
from src.dddql_ultimate_tictactoe_agent import DDDQLUltimateTicTacToeAgent
from src.ultimate_tictactoe_pass_through_interface import UltimateTicTacToePassThroughInterface


class UltimateTicTacToeEnvironmentSelfPlay(UltimateTicTacToeEnvironment):
    """
    Ultimate Tic-Tac-Toe environment in which the environment player is the same agent.

    Then environment player sees the state as the inverse of the real state since it was trained to be the agent player.
    """

    def __init__(self,
                 name: str,
                 environment_player: Player,
                 agent_player_win_reward: float,
                 environment_player_win_reward: float,
                 draw_reward: float,
                 agent: DDDQLUltimateTicTacToeAgent):
        # Define environment attributes
        self._agent: DDDQLUltimateTicTacToeAgent = agent
        self._interface: UltimateTicTacToePassThroughInterface = UltimateTicTacToePassThroughInterface(self)
        # Generate the base tic tac toe environment
        super(UltimateTicTacToeEnvironmentSelfPlay, self).__init__(name, environment_player, agent_player_win_reward, environment_player_win_reward, draw_reward)

    def get_environment_player_first_action(self,
                                            logger: logging.Logger,
                                            session) -> int:
        # Just return a random action
        return self.get_random_action(logger, session)

    def get_environment_player_action(self,
                                      logger: logging.Logger,
                                      session) -> int:
        # Compute the observation for the agent using the pass-through interface on the flipped version of the state
        observation_current = self._interface.environment_state_to_observation(logger, session, self._encode_state_int(self._flipped_state))
        # Get the agent action
        # Give the illusion of playing for the agent player
        if self.current_player == Player.x:
            self.current_player = Player.o
        else:
            self.current_player = Player.x
        if self.last_player == Player.x:
            self.last_player = Player.o
        elif self.last_player == Player.o:
            self.last_player = Player.x
        agent_action: int = self._agent.act_adversarial(logger, session, self._interface, observation_current)
        # Revert back the environment player
        if self.current_player == Player.x:
            self.current_player = Player.o
        else:
            self.current_player = Player.x
        if self.last_player == Player.x:
            self.last_player = Player.o
        elif self.last_player == Player.o:
            self.last_player = Player.x
        # Get the environment flipped action using the pass-through interface
        environment_action_flipped: int = self._interface.agent_action_to_environment_action(logger, session, agent_action)
        # Flip back the environment action
        if self.environment_player == Player.x:
            environment_action: int = environment_action_flipped + 1
        else:
            environment_action: int = environment_action_flipped - 1
        # Return the action
        return environment_action
