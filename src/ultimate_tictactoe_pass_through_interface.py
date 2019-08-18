# Import packages

import logging

# Import usienarl

from usienarl import Interface, SpaceType

# Import required src

from src.ultimate_tictactoe_environment import UltimateTicTacToeEnvironment


class UltimateTicTacToePassThroughInterface(Interface):
    """
    Default pass-through interface for all Ultimate Tic Tac Toe environments.
    """

    def __init__(self,
                 environment: UltimateTicTacToeEnvironment):
        # Define specific ultimate tic tac toe environment variable
        self._ultimate_tictactoe_environment: UltimateTicTacToeEnvironment = environment
        # Generate the base interface
        super(UltimateTicTacToePassThroughInterface, self).__init__(environment)

    def agent_action_to_environment_action(self,
                                           logger: logging.Logger,
                                           session,
                                           agent_action):
        # Just return the agent action
        return agent_action

    def environment_action_to_agent_action(self,
                                           logger: logging.Logger,
                                           session,
                                           environment_action):
        # Just return the environment action
        return environment_action

    def environment_state_to_observation(self,
                                         logger: logging.Logger,
                                         session,
                                         environment_state):
        # Just return the environment state
        return environment_state

    def get_possible_actions(self,
                             logger: logging.Logger,
                             session):
        """
        Get the possible agent actions from the environment current state available actions.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: a list of agent actions which the agent can execute
        """
        environment_actions: [] = self._ultimate_tictactoe_environment.get_possible_actions(logger, session)
        agent_actions: [] = []
        for environment_action in environment_actions:
            agent_actions.append(self.environment_action_to_agent_action(logger, session, environment_action))
        return agent_actions

    @property
    def observation_space_type(self) -> SpaceType:
        # Just return the environment state space type
        return self._environment.state_space_type

    @property
    def observation_space_shape(self):
        # Just return the environment state space shape
        return self._environment.state_space_shape

    @property
    def agent_action_space_type(self) -> SpaceType:
        # Just return the environment action space type
        return self._environment.action_space_type

    @property
    def agent_action_space_shape(self):
        # Just return the environment action space shape
        return self._environment.action_space_shape
