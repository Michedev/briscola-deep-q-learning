import sys

from fire import Fire
from path import Path
import os, logging

ROOT = Path(__file__).parent.abspath()
sys.path.append(ROOT / 'briscola')
sys.path.append(ROOT / 'briscola' / 'player')
sys.path.append(ROOT / 'briscola' / 'player' / 'smart_player')

from briscola.main import play_smart_vs_random

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


def main(n=100):
    play_smart_vs_random(n)


if __name__ == '__main__':
    Fire(main)
