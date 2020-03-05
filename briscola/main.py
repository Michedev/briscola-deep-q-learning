from game import GameEngine
from random_player import PseudoRandomPlayer
from smart_player import SmartPlayer
import tensorflow as tf
from path import Path

ROOT = Path(__file__).parent
MODEL_PATH = ROOT.parent / 'brain.h5'

def play_smart_vs_random(n=None):
    if n is None:
        n = 1
    p1 = SmartPlayer()
    if MODEL_PATH.exists():
        p1.brain.load_weights(MODEL_PATH)
    p2 = PseudoRandomPlayer()
    wins_p1 = 0
    for i in range(n):
        game = GameEngine([p1, p2], [0,0])
        p1.set_observable_public_state(game.public_state)
        winner = game.setup_and_play()
        if not isinstance(winner, list):
            print('winner is ', winner.name)
            if p1.name == winner.name:
                wins_p1 += 1
        else:
            print('draw')
    print(f'wins_p1: {wins_p1}, win rate: {wins_p1 / n}')

if __name__ == '__main__':
    play_smart_vs_random()