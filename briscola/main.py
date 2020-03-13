from epsgreedy_player import EpsGreedyPlayer
from game import GameEngine
from random_player import PseudoRandomPlayer
from smart_player import SmartPlayer
import tensorflow as tf
from path import Path

ROOT = Path(__file__).parent


def play_smart_vs_random(n=None, hparams_smart=None):
    if hparams_smart is None:
        hparams_smart = dict()
    p1 = SmartPlayer(**hparams_smart)
    p2 = PseudoRandomPlayer()
    play_matches([p1, p2], n)


def play_smart_vs_eps_greedy(eps=0.3, n=None, hparams_smart=None):
    if hparams_smart is None:
        hparams_smart = {}
    p1 = SmartPlayer(**hparams_smart)
    p2 = EpsGreedyPlayer(eps)
    play_matches([p1, p2], n)


def play_matches(players, n=None):
    if n is None:
        n = 1
    p1, p2 = players
    wins_p1 = 0
    for i in range(n):
        game = GameEngine([p1, p2], [0, 0])
        p1.set_observable_public_state(game.public_state)
        winner = game.setup_and_play()
        if not isinstance(winner, list):
            print('winner is ', winner.name)
            if p1.name == winner.name:
                wins_p1 += 1
        else:
            print('draw')
    print(f'wins_p1: {wins_p1}, win rate: {wins_p1 / n}\n',
          f'wins_p2: {n - wins_p1}, win rate: {1 - wins_p1 / n}\n')


if __name__ == '__main__':
    play_smart_vs_random()
