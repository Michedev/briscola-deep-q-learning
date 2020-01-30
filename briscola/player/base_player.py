from abc import ABC, abstractmethod
import logging
from state import PublicState


class BasePlayer(ABC):

    def __init__(self, obs_public_state=None):
        self.hand = []
        self.name = None
        self.__obs_public_state = obs_public_state
        self.__logger = logging.getLogger('Briscola')

    def reset_player(self):
        self.hand = []

    @abstractmethod
    def choose_card(self) -> int:
        pass

    def discard_card(self):
        i = self.choose_card()
        try:
            c = self.hand.pop(i)
            self.__logger.info(f'{self.name} discard {c}')
            return c
        except IndexError as e:
            print('hand len: ', len(self.hand), 'i: ', i)
            raise e

    def is_empty_hand(self):
        return len(self.hand) == 0

    def get_public_state(self) -> 'PublicState':
        return self.__obs_public_state()

    def notify_turn_winner(self, points):
        pass

    def notify_game_winner(self, name: str):
        pass

    def set_observable_public_state(self, obs):
        self.__obs_public_state = obs