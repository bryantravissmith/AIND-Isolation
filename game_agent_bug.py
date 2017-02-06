"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
# import random
from math import floor


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_heuristic_weighted_increase_moves(game, player, w1=1, w2=-1):
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

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(w1 * own_moves + w2 * opp_moves)


def custom_heuristic_2(game, player):
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
    return -1


def custom_heuristic_3(game, player):
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
    return -1


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
    return custom_heuristic_weighted_increase_moves(game, player)

    raise NotImplementedError


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
                 iterative=True, method='minimax', timeout=10., use_symmetries=False):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.use_symmetries = use_symmetries
        self.timeout_counter = 0
        self.counter = 0
        self.depths = 0

    def check_symmetry(self, game):
        """
        Determines if the board postition current has a vertical refleection symmetry.
        Only returns True for if there has been 5 or less moves


        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        Returns
        ----------
        (boolean, boolean, boolean, boolean)
            Returns true if there is a symmetry in the board.
            Order is (vertial reflect, horizontial reflect, left diagonal reflect, right diagonal reflect)
        """

        vertical = True
        horizontial = True
        left_diagonal = True
        right_diagonal = True

        h = game.height
        w = game.width
        hf = floor(h / 2)
        wf = floor(w / 2)

        for row_index in range(-hf, hf + 1):
            for column_index in range(-wf, wf + 1):
                i = row_index + hf
                j = column_index + wf

                cell = game.__board_state__[i][j]

                if vertical:
                    other_cell = game.__board_state__[-abs(row_index) + hf][j]
                    if cell != other_cell:
                        vertical = False

                if horizontial:
                    other_cell = game.__board_state__[i][-abs(column_index) + wf]
                    if cell != other_cell:
                        horizontial = False

                if right_diagonal:
                    m = -row_index + hf
                    n = -column_index + wf
                    other_cell = game.__board_state__[n][m]
                    if cell != other_cell:
                        right_diagonal = False

                if left_diagonal:
                    other_cell = game.__board_state__[j][i]
                    if cell != other_cell:
                        left_diagonal = False

                if not (vertical or horizontial or right_diagonal or left_diagonal):
                    return (False, False, False, False)

        return (vertical, horizontial, left_diagonal, right_diagonal)

    def get_filter_legal_moves(self, game, legal_moves):
        vertical, horizontial, left_diagonal, right_diagonal = self.check_symmetry(game)
        h = game.height
        w = game.width
        hf = floor(h / 2)
        wf = floor(w / 2)

        if vertical:
            new_moves = []
            for a, b in legal_moves:
                if a - hf < 0:
                    a = abs(a - hf) + hf
                new_moves.append((a, b))
            legal_moves = list(set(new_moves))

        if horizontial:
            new_moves = []
            for a, b in legal_moves:
                if b - wf < 0:
                    b = abs(b - wf) + wf
                new_moves.append((a, b))
            legal_moves = list(set(new_moves))

        if left_diagonal:
            new_moves = []
            for a, b in legal_moves:
                if a < b:
                    (a, b) = (b, a)
                new_moves.append((a, b))
            legal_moves = list(set(new_moves))

        if right_diagonal:
            new_moves = []
            for a, b in legal_moves:
                if -(a - hf) < (b - wf):
                    (a, b) = (-(b - wf) + wf, -(a - hf) + hf)
                new_moves.append((a, b))
            legal_moves = list(set(new_moves))

        return legal_moves

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
        self.counter += 1

        if self.use_symmetries:
            legal_moves = self.get_filter_legal_moves(game, legal_moves)

        # Check for symmetries in the board positions if flag set
        # if self.use_symmetries:
        #    legal_moves = self.get_filter_legal_moves(game, legal_moves)

        # If not moves return the  no available legal moves result
        if len(legal_moves) == 0:
            return (-1, -1)

        move = None
        depth = 0
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if self.iterative:
                while True:
                    moves = []
                    for possible_move in legal_moves:
                        game_forecast = game.forecast_move(possible_move)
                        if self.method == "minimax":
                            s, _ = self.minimax(game_forecast, depth)
                        elif self.method == "alphabeta":
                            s, _ = self.alphabeta(game_forecast, depth)
                        else:
                            raise NotImplementedError
                        moves.append((s, possible_move))

                    score, move = max(moves)
                    depth += 1
                    if score == float("-inf") or score == float("inf"):
                        return move
            else:
                depth = self.search_depth
                moves = []
                for possible_move in legal_moves:
                    game_forecast = game.forecast_move(possible_move)
                    if self.method == "minimax":
                        s, _ = self.minimax(game_forecast, depth)
                    elif self.method == "alphabeta":
                        s, _ = self.alphabeta(game_forecast, depth)
                    else:
                        raise NotImplementedError
                    moves.append((s, possible_move))
                _, move = max(moves)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            self.timeout_counter += 1
            if move is None:
                if len(moves) == 0:
                    move = legal_moves[0]
                else:
                    _, move = max(moves)

            return move

        # Return the best move from the last completed search iteration
        return move
        raise NotImplementedError

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

        # Create an array of (score, move)
        moves = []

        # For each legal move forecast the next move and find the score
        legal_moves = game.get_legal_moves(game.active_player)

        # remove equival moves if symmetry flag is set
        if self.use_symmetries:
            legal_moves = self.get_filter_legal_moves(game, legal_moves)

        if len(legal_moves) == 0:
            if maximizing_player:
                return (float("inf"), (-1, -1))
            else:
                return (float("-inf"), (-1, -1))

        for move in legal_moves:
            game_forecast = game.forecast_move(move)
            s, _ = self.minimax(
                game_forecast,
                depth - 1,
                not maximizing_player
            )

            moves.append((s, move))

        # Returns the max/min score and move for the current depths.
        if maximizing_player:
            return max(moves)
        else:
            return min(moves)

        raise NotImplementedError

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

        # Assumes the maximizing player is the player we are concerned with
        player = game.inactive_player
        if maximizing_player:
            player = game.active_player

        # Returns the score of the current node without any moves
        if depth == 0:
            return (self.score(game, player), (-1, -1))

        # Create an array of (score, move)
        moves = []

        # For each legal move forecast the next move and find the score
        legal_moves = game.get_legal_moves(game.active_player)

        # remove equival moves if symmetry flag is set
        # if self.use_symmetries:
        #    legal_moves = self.get_filter_legal_moves(game, legal_moves)

        if len(legal_moves) == 0:
            if maximizing_player:
                return (float("inf"), (-1, -1))
            else:
                return (float("-inf"), (-1, -1))

        for move in legal_moves:
            game_forecast = game.forecast_move(move)
            s, _ = self.alphabeta(
                game_forecast,
                depth - 1,
                alpha,
                beta,
                not maximizing_player
            )
            moves.append((s, move))
            if maximizing_player:
                alpha = max(alpha, s)
                if s >= beta:
                    break
            else:
                beta = min(beta, s)
                if s <= alpha:
                    break

        if maximizing_player:
            return max(moves)
        else:
            return min(moves)

        # TODO: finish this function!
        raise NotImplementedError
