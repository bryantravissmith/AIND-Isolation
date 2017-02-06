from isolation import Board
from game_agent import CustomPlayer
from game_agent import custom_score
from sample_players import HumanPlayer
from sample_players import improved_score
import random
TIME_LIMIT = 550

if __name__ == "__main__":

	count = 0
	wins = 0
	losses = 0
	firsts = 0

	for i in range(10):
		CUSTOM_ARGS = {"search_depth": 5, "method": 'minimax', 'iterative': True, 'use_symmetries': False}
		OTHER_ARGS = {"search_depth": 5, "method": 'minimax', 'iterative': False}
		player1 = CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS)
		player2 = CustomPlayer(score_fn=improved_score, **OTHER_ARGS)
		if random.random() < 0.5:
			game = Board(player1, player2)
			firsts += 1
		else:
			game = Board(player2, player1)
		winner, a, termination = game.play(time_limit=TIME_LIMIT)
		print(player1.timeout_counter / player1.counter, player2.timeout_counter / player2.counter)
		if player1 == winner:
			wins += 1
		else:
			losses += 1
		count += 1
		print(wins, losses, count, firsts)
