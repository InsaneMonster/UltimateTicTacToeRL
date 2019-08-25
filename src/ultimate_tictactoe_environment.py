# Import packages

import logging
import numpy
import random
import time
import enum
import copy
import math

# Import usienarl

from usienarl import Environment, SpaceType


# Define player type class

class Player(enum.Enum):
    x = 1
    o = -1
    none = 0


class UltimateTicTacToeEnvironment(Environment):
    """
    Ultimate Tic-Tac-Toe abstract environment.

    The state is a vector representing the board with:
        - 0 is the empty cell
        - 1 is the X cell
        - -1 is the O cell

    The action is a number where:
        - 0 => X in 0-0
        - 1 => O in 0-0
        - 4 => X in 0-2
        - 5 => O in 0-2
        - 2n => X in 0-n
        - 2n+1 => O in 0-n
        - 2n+2 + 2n => X in 1-n
        - 2n+2 + 2n+1 => O in 1-n
        Note: the quotient of action / 18 is the board index, while the remainder is used to determine player and position.
    """

    def __init__(self,
                 name: str,
                 environment_player: Player,
                 agent_player_win_reward: float,
                 environment_player_win_reward: float,
                 draw_reward: float):
        # Define attributes
        self.winner: Player = Player.none
        self.last_player: Player = Player.none
        self.current_player: Player = Player.none
        self.environment_player: Player = environment_player
        if self.environment_player == Player.x:
            self.agent_player: Player = Player.o
        else:
            self.agent_player: Player = Player.x
        self.agent_player_win_reward: float = agent_player_win_reward
        self.environment_player_win_reward: float = environment_player_win_reward
        self.draw_reward: float = draw_reward
        # Define internal attributes
        self._move: int = 0
        self._episode_done: bool = False
        # Define internal empty attributes
        self._last_action: int = None
        self._board: [] = None
        self._flipped_board: [] = None
        self._intermediate_board: [] = None
        self._state: numpy.ndarray = None
        self._flipped_state: numpy.ndarray = None
        # Generate the base environment
        super(UltimateTicTacToeEnvironment, self).__init__(name)

    def setup(self,
              logger: logging.Logger) -> bool:
        # The environment setup is always successful
        return True

    def initialize(self,
                   logger: logging.Logger,
                   session):
        pass

    def close(self,
              logger: logging.Logger,
              session):
        pass

    def reset(self,
              logger: logging.Logger,
              session):
        # Reset attributes
        self.winner = Player.none
        self.last_player = Player.none
        # Reset internal attributes
        self._move: int = 0
        self._episode_done = False
        self._last_action = None
        # Reset state
        self._board = [
                       [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                       [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                       [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                       [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                       [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                       [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                       [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                       [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                       [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none]
                       ]
        self._flipped_board = [
                                [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                                [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                                [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                                [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                                [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                                [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                                [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                                [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none],
                                [Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none]
                              ]
        self._state = numpy.hstack(self._board)
        self._flipped_state = numpy.hstack(self._flipped_board)
        # Choose a random starting player
        self.current_player = Player.o
        if random.uniform(0, 1) <= 0.5:
            self.current_player = Player.x
        # If the current player is the environment player, let it decide how to play
        if self.current_player == self.environment_player:
            # Increase move count
            self._move += 1
            # Save the current representation of the board for rendering purpose
            self._intermediate_board = copy.deepcopy(self._board)
            # Get the environment player action
            environment_player_action: int = self.get_environment_player_first_action(logger, session)
            board, remainder = divmod(environment_player_action, 18)
            position, player = divmod(remainder, 2)
            self._last_action = environment_player_action
            # Set the player to its defined value and update the state
            if player == 0:
                player = Player.o
                flipped_player = Player.x
            else:
                player = Player.x
                flipped_player = Player.o
            self._board[board][position] = player
            self._flipped_board[board][position] = flipped_player
            self._state = numpy.hstack(self._board)
            self._flipped_state = numpy.hstack(self._flipped_board)
            # Update the last player and current player
            self.last_player = self.current_player
            if self.current_player == Player.x:
                self.current_player = Player.o
            else:
                self.current_player = Player.x
        # Return the first state encoded
        return self._encode_state_int(self._state)

    def step(self,
             logger: logging.Logger,
             action,
             session):
        # Increase move count
        self._move += 1
        # Change the state with the given action
        board, remainder = divmod(action, 18)
        position, player = divmod(remainder, 2)
        self._last_action = action
        # Set the player to its defined value and update the state
        if player == 0:
            player = Player.o
            flipped_player = Player.x
        else:
            player = Player.x
            flipped_player = Player.o
        self._board[board][position] = player
        self._flipped_board[board][position] = flipped_player
        self._state = numpy.hstack(self._board)
        self._flipped_state = numpy.hstack(self._flipped_board)
        # Update the last player and current player
        self.last_player = self.current_player
        if self.current_player == Player.x:
            self.current_player = Player.o
        else:
            self.current_player = Player.x
        # Reset the current intermediate state
        self._intermediate_board = None
        # Check for winner and episode completion flag
        self._episode_done, self.winner = self._check_if_final(self._board)
        # If the current player is the environment player, let it decide how to play
        if not self._episode_done and self.current_player == self.environment_player:
            # Increase move count
            self._move += 1
            # Save the current representation of the board for rendering purpose
            self._intermediate_board = copy.deepcopy(self._board)
            # Get the environment player action
            environment_player_action: int = self.get_environment_player_first_action(logger, session)
            board, remainder = divmod(environment_player_action, 18)
            position, player = divmod(remainder, 2)
            self._last_action = environment_player_action
            # Set the player to its defined value and update the state
            if player == 0:
                player = Player.o
                flipped_player = Player.x
            else:
                player = Player.x
                flipped_player = Player.o
            self._board[board][position] = player
            self._flipped_board[board][position] = flipped_player
            self._state = numpy.hstack(self._board)
            self._flipped_state = numpy.hstack(self._flipped_board)
            # Update the last player and current player
            self.last_player = self.current_player
            if self.current_player == Player.x:
                self.current_player = Player.o
            else:
                self.current_player = Player.x
            # Check for winner and episode completion flag
            self._episode_done, self.winner = self._check_if_final(self._board)
        # Assign rewards
        reward: float = 0.0
        if self._episode_done:
            if self.winner == Player.x:
                reward = self.agent_player_win_reward
            elif self.winner == Player.o:
                reward = self.environment_player_win_reward
            else:
                reward = self.draw_reward
        # Return the encoded state, the reward and the episode completion flag
        return self._encode_state_int(self._state), reward, self._episode_done

    def render(self,
               logger: logging.Logger,
               session):
        # Print the intermediate board, if any
        if self._intermediate_board is not None:
            self._print_board(self._intermediate_board)
            # Print separator
            print("____________________")
            time.sleep(0.1)
        # Print the current state
        self._print_board(self._board)
        # Print separator
        print("____________________")
        # Print end of episode footer and results
        if self._episode_done:
            print("MATCH END")
            print("Played moves: " + str(self._move))
            if self.winner != Player.none:
                print("Winner player is " + ("X" if self.winner == Player.x else "O"))
            else:
                print("There is no winner: it's a draw!")
            print("____________________")
        time.sleep(0.1)

    def get_random_action(self,
                          logger: logging.Logger,
                          session):
        # Choose a random action in the possible actions
        return random.choice(self.get_possible_actions(logger, session))

    @property
    def state_space_type(self):
        return SpaceType.continuous

    @property
    def state_space_shape(self):
        return 81,

    @property
    def action_space_type(self):
        return SpaceType.discrete

    @property
    def action_space_shape(self):
        return 81 * 2,

    def get_possible_actions(self,
                             logger: logging.Logger,
                             session) -> []:
        """
        Return a list of the indices of all the possible actions at the current state of the environment.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: a list of indices containing the possible actions
        """
        # Get the mask
        mask: numpy.ndarray = self.get_action_mask(logger, session)
        # Return the list of possible actions from the mask indices
        return numpy.where(mask >= 0.0)[0].tolist()

    def get_action_mask(self,
                        logger: logging.Logger,
                        session) -> numpy.ndarray:
        """
        Return all the possible action at the current state in the environment wrapped in a numpy array mask.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: an array of -infinity (for unavailable actions) and 0.0 (for available actions)
        """
        last_action_position = None
        if self._last_action is not None:
            _, remainder = divmod(self._last_action, 18)
            last_action_position, _ = divmod(remainder, 2)
            # Check the board pointed by the last action position, if it's completed leave the player free to play where possible
            # Note: the last action position with value None is used to determine whether or not the player is free to play
            # Also, check if the target board is available, if not, allow the player to move freely
            if self._get_board_winner(self._board[last_action_position]) != Player.none:
                last_action_position = None
            else:
                empty_found: bool = False
                for element in self._board[last_action_position]:
                    if element == Player.none:
                        empty_found = True
                        break
                if not empty_found:
                    last_action_position = None
        # Get all the possible action mask according to current board state and the last action
        mask: numpy.ndarray = -math.inf * numpy.ones(self.action_space_shape, dtype=float)
        for action in range(*self.action_space_shape):
            board_index, remainder = divmod(action, 18)
            # Check the last action position (the new action board) and skip the action if it's the wrong one
            # If last action position is None the player can choose where to play freely
            if last_action_position is not None and board_index != last_action_position:
                continue
            # Check if the board is completed and skip if it is
            if self._get_board_winner(self._board[board_index]) != Player.none:
                continue
            position, player = divmod(remainder, 2)
            # Set the player to its defined value
            if player == 0:
                player = Player.o
            else:
                player = Player.x
            if self._board[board_index][position] == Player.none and player == self.current_player:
                mask[action] = 0.0
        return mask

    @staticmethod
    def _get_board_winner(board: []) -> Player:
        """
        Get the winner of the given board, if any.

        :param board: the board list to check (a tic-tac-toe board)
        :return: the winner player, if any
        """
        winner: Player = Player.none
        if board[0] != Player.none and board[0] == board[1] == board[2]:
            winner = board[0]
        if board[0] != Player.none and board[0] == board[3] == board[6]:
            winner = board[0]
        if board[0] != Player.none and board[0] == board[4] == board[8]:
            winner = board[0]
        if board[2] != Player.none and board[2] == board[5] == board[8]:
            winner = board[2]
        if board[2] != Player.none and board[2] == board[4] == board[6]:
            winner = board[2]
        if board[1] != Player.none and board[1] == board[4] == board[7]:
            winner = board[1]
        if board[3] != Player.none and board[3] == board[4] == board[5]:
            winner = board[3]
        if board[6] != Player.none and board[6] == board[7] == board[8]:
            winner = board[6]
        # Return the winner player
        return winner

    @staticmethod
    def _check_if_final(full_board: []):
        """
        Check if the given state is final and also return the winner.

        :return: True if final, False otherwise and the winner
        """
        state: numpy.ndarray = numpy.array([Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none])
        available_boards: [] = []
        for index in range(len(full_board)):
            state[index] = UltimateTicTacToeEnvironment._get_board_winner(full_board[index])
            # Add the sub-board to the list of available sub-boards if there is not winner there yet
            if state[index] == Player.none:
                available_boards.append(full_board[index])
        # Check if state is final and compute winner
        winner: Player = Player.none
        episode_done: bool = False
        if state[0] != Player.none and state[0] == state[1] == state[2]:
            episode_done = True
            winner = state[0]
        if state[0] != Player.none and state[0] == state[3] == state[6]:
            episode_done = True
            winner = state[0]
        if state[0] != Player.none and state[0] == state[4] == state[8]:
            episode_done = True
            winner = state[0]
        if state[2] != Player.none and state[2] == state[5] == state[8]:
            episode_done = True
            winner = state[2]
        if state[2] != Player.none and state[2] == state[4] == state[6]:
            episode_done = True
            winner = state[2]
        if state[1] != Player.none and state[1] == state[4] == state[7]:
            episode_done = True
            winner = state[1]
        if state[3] != Player.none and state[3] == state[4] == state[5]:
            episode_done = True
            winner = state[3]
        if state[6] != Player.none and state[6] == state[7] == state[8]:
            episode_done = True
            winner = state[6]
        # Check for draw: no empty element in each sub-board not completed and no winner
        if winner == Player.none:
            empty_found: bool = False
            for board in available_boards:
                for element in board:
                    if element == Player.none:
                        empty_found = True
                        break
            if not empty_found:
                episode_done = True
        # Return completion flag and winner
        return episode_done, winner

    @staticmethod
    def _print_board(full_board: []):
        """
        Print the given board.

        :param full_board: the full board of the ultimate tic-tac-toe to print
        """
        # Get all the rows in each one of the boards
        first_row: [] = ["|"]
        second_row: [] = ["|"]
        third_row: [] = ["|"]
        fourth_row: [] = ["|"]
        fifth_row: [] = ["|"]
        sixth_row: [] = ["|"]
        seventh_row: [] = ["|"]
        eighth_row: [] = ["|"]
        ninth_row: [] = ["|"]
        for i, board in enumerate(full_board):
            for j in range(len(board)):
                # Convert the board to its graphical representation
                graphical_element: str = "-"
                if board[j] == Player.x:
                    graphical_element = "X"
                elif board[j] == Player.o:
                    graphical_element = "O"
                # Save in rows each sub-board
                if i < 3:
                    if j < 3:
                        first_row.append(graphical_element)
                        # Print vertical separator between sub-boards
                        if j == 2:
                            first_row.append("|")
                    elif 3 <= j < 6:
                        second_row.append(graphical_element)
                        # Print vertical separator between sub-boards
                        if j == 5:
                            second_row.append("|")
                    else:
                        third_row.append(graphical_element)
                        # Print vertical separator between sub-boards
                        if j == 8:
                            third_row.append("|")
                elif 3 <= i < 6:
                    if j < 3:
                        fourth_row.append(graphical_element)
                        # Print vertical separator between sub-boards
                        if j == 2:
                            fourth_row.append("|")
                    elif 3 <= j < 6:
                        fifth_row.append(graphical_element)
                        # Print vertical separator between sub-boards
                        if j == 5:
                            fifth_row.append("|")
                    else:
                        sixth_row.append(graphical_element)
                        # Print vertical separator between sub-boards
                        if j == 8:
                            sixth_row.append("|")
                else:
                    if j < 3:
                        seventh_row.append(graphical_element)
                        # Print vertical separator between sub-boards
                        if j == 2:
                            seventh_row.append("|")
                    elif 3 <= j < 6:
                        eighth_row.append(graphical_element)
                        # Print vertical separator between sub-boards
                        if j == 5:
                            eighth_row.append("|")
                    else:
                        ninth_row.append(graphical_element)
                        # Print vertical separator between sub-boards
                        if j == 8:
                            ninth_row.append("|")
        # Print each row with separators
        print(' '.join(first_row))
        print(' '.join(second_row))
        print(' '.join(third_row))
        print("-------------------------")
        print(' '.join(fourth_row))
        print(' '.join(fifth_row))
        print(' '.join(sixth_row))
        print("-------------------------")
        print(' '.join(seventh_row))
        print(' '.join(eighth_row))
        print(' '.join(ninth_row))

    @staticmethod
    def _encode_state_int(state: numpy.ndarray):
        """
        Encode the given state of the board (expressed in player occupied cells) with an integer sequence.

        :param state: the state to encode
        :return: the encoded state
        """
        encoded_state: numpy.ndarray = numpy.zeros(state.size, dtype=int)
        for i in range(state.size):
            encoded_state[i] = state[i].value
        return encoded_state

    def get_environment_player_first_action(self,
                                            logger: logging.Logger,
                                            session) -> int:
        """
        Get the first action from the environment player, if any.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: the action of the environment agent at the first state
        """
        raise NotImplementedError()

    def get_environment_player_action(self,
                                      logger: logging.Logger,
                                      session) -> int:
        """
        Get the action from the environment player, if any.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: the action of the environment agent at the current state
        """
        raise NotImplementedError()
