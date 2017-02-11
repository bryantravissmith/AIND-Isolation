"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import operator
from math import floor, sqrt, exp


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def get_distance(game, player):
    """Calculate the distance between two players

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        returns squared dista
    """
    own_loc = game.get_player_location(player)
    opp_loc = game.get_player_location(game.get_opponent(player))
    distance = (own_loc[0] - opp_loc[0])**2 + (own_loc[1] - opp_loc[1])**2
    return sqrt(distance)


def custom_heuristic_lower_distance(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player by finding the weighted sum


    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return 1 / get_distance(game, player)


def custom_heuristic_improve_as_game_ends(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player by finding the weighted sum


    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # number_spaces = game.width * game.height
    open_spaces = len(game.get_blank_spaces())
    # closed_space = number_spaces - open_spaces

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return -float(open_spaces * opp_moves / (own_moves + 1e-6))


def custom_heuristic_opp_has_less_moves_end_game(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player by finding the weighted sum


    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    # if game.is_loser(player):
    #    return float("-inf")

    # if game.is_winner(player):
    #    return float("inf")
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    close_spaces = game.height * game.width - len(game.get_blank_spaces())

    if player == game.inactive_player and opp_moves == 0:
        return float("inf")
    if player == game.active_player and own_moves == 0:
        return float("-inf")

    return -(opp_moves + 0.5) / (own_moves + 0.5) * close_spaces


def custom_heuristic_center(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    loc = game.get_player_location(player)
    r1 = loc[0] - floor(game.height / 2)
    r2 = loc[1] - floor(game.width / 2)
    r = sqrt(r1**2 + r2**2)
    return exp(-r)


def custom_heuristic_takeway_moves(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    opponent = game.get_opponent(player)
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opponent)

    if player == game.inactive_player and not opp_moves:
        return float("inf")
    if player == game.active_player and not own_moves:
        return float("-inf")

    own_locations = tuple(map(operator.add, [game.get_player_location(player)] * len(own_moves), own_moves))

    opp_locations = tuple(map(operator.add, [game.get_player_location(opponent)] * len(opp_moves), opp_moves))

    opp_left_over_locations = set(opp_locations) - set(own_locations)

    return (len(opp_locations) - len(opp_left_over_locations))


def custom_heuristic_change_during_game(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    number_spaces = game.width * game.height
    open_spaces = len(game.get_blank_spaces())
    closed_space = number_spaces - open_spaces

    if closed_space < 8:
        return custom_heuristic_center(game, player)
    else:
        return custom_heuristic_opp_has_less_moves_end_game(game, player)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Returns custome heuristic
    return custom_heuristic_opp_has_less_moves_end_game(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.depths = []

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # If not moves return the  no available legal moves result
        move = (-1, -1)
        if len(legal_moves) == 0:
            return move

        if len(game.get_blank_spaces()) == game.width * game.height:
            self.depths.append(0)
            return (game.height // 2, game.width // 2)

        depth = 1
        # The search method call (alpha beta or minimax) should happen in
        # here in order to avoid timeout. The try/except block will
        # automatically catch the exception raised by the search method
        # when the timer gets close to expiring
        try:
            if self.iterative:
                while True:
                    if self.method == "minimax":
                        s, move = self.minimax(game, depth)
                    elif self.method == "alphabeta":
                        s, move = self.alphabeta(game, depth)
                    else:
                        raise NotImplementedError

                    if s == float("inf"):
                        return move

                    depth += 1
            else:
                depth = self.search_depth
                if self.method == "minimax":
                    s, move = self.minimax(game, depth)
                elif self.method == "alphabeta":
                    s, move = self.alphabeta(game, depth)
                else:
                    raise NotImplementedError
                return move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            self.depths.append(depth)
            return move

        # Return the best move from the last completed search iteration
        self.depths.append(depth)
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Assumes the maximizing player is the player we are concerned with
        player = game.inactive_player
        if maximizing_player:
            player = game.active_player

        # Returns the score of the current node without any moves
        if depth == 0:
            return (self.score(game, player), (-1, -1))

        # Set Best Score
        best_score = float('-inf') if maximizing_player else float('+inf')
        best_move = (-1, -1)

        # For each legal move forecast the next move and find the score
        legal_moves = game.get_legal_moves(game.active_player)

        for move in legal_moves:
            game_forecast = game.forecast_move(move)
            s, _ = self.minimax(
                game_forecast,
                depth - 1,
                not maximizing_player
            )
            if maximizing_player:
                if s > best_score:
                    best_score = s
                    best_move = move
            else:
                if s < best_score:
                    best_score = s
                    best_move = move

        # Returns best score, best move.
        return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        player = game.inactive_player
        if maximizing_player:
            player = game.active_player

        # Returns the score of the current node without any moves
        if depth == 0:
            return (self.score(game, player), (-1, -1))

        # Set Best Score
        best_score = float('-inf') if maximizing_player else float('+inf')
        best_move = (-1, -1)

        # For each legal move forecast the next move and find the score
        legal_moves = game.get_legal_moves(game.active_player)

        for move in legal_moves:
            game_forecast = game.forecast_move(move)
            s, _ = self.alphabeta(
                game_forecast,
                depth - 1,
                alpha,
                beta,
                not maximizing_player
            )
            if maximizing_player:
                if s > best_score:
                    best_score = s
                    best_move = move
                    alpha = max(alpha, s)

                if s >= beta:
                    break
            else:
                if s < best_score:
                    best_score = s
                    best_move = move
                    beta = min(beta, s)
                if s <= alpha:
                    break

        return best_score, best_move
