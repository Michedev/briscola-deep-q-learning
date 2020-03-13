import sys

from fire import Fire
from path import Path
import os, logging

ROOT = Path(__file__).parent.abspath()
sys.path.append(ROOT / 'briscola')
sys.path.append(ROOT / 'briscola' / 'player')
sys.path.append(ROOT / 'briscola' / 'player' / 'smart_player')

from briscola.main import play_smart_vs_random, play_smart_vs_eps_greedy

logger = logging.getLogger('Briscola')
# logger.setLevel(logging.INFO)
# c_handler = logging.StreamHandler()
# f_handler = logging.FileHandler('BriscolaMatch.log')
# c_handler.setLevel(logging.INFO)
# f_handler.setLevel(logging.INFO)
# logger.addHandler(c_handler)
# logger.addHandler(f_handler)
#

os.system('rm -rf logs')


def main(n=100, adversary='random', epsilon_greedy=0.3,
         discount_factor=0.9, experience_size=300_000, update_q_fut=1000,
         sample_experience=64, update_freq=4, no_update_start=500):
    hparams = dict(discount_factor=discount_factor, experience_size=experience_size,
                   update_q_fut=update_q_fut, sample_experience=sample_experience,
                   update_freq=update_freq, no_update_start=no_update_start)
    if adversary == 'random':
        play_smart_vs_random(n, hparams)
    elif adversary == 'greedy':
        play_smart_vs_eps_greedy(epsilon_greedy, n, hparams)
    else:
        raise ValueError('adversary must be {"random", "greedy"} - your input is ' + adversary)


if __name__ == '__main__':
    Fire(main)
