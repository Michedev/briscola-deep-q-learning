import sys
from path import Path
import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


ROOT = Path(__file__).parent.abspath()
sys.path.append(ROOT / 'briscola')
sys.path.append(ROOT / 'briscola' / 'player')

from briscola.main import  play_smart_vs_random

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
play_smart_vs_random(1)