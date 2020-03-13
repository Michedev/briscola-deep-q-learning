from random import randint

from base_player import BasePlayer
from callbacks import Callbacks, WinRateLog, \
    IncreaseEpsilonOnLose
from feature_encoding import build_state_array, STATE_SIZE
from qagent import QAgent

GLOBAL_COUNTER = 0


class SmartPlayer(BasePlayer, QAgent):

    def __init__(self, discount_factor=0.9, experience_size=300_000, update_q_fut=1000,
                 sample_experience=64, update_freq=4, no_update_start=500):
        QAgent.__init__(self, [STATE_SIZE], discount_factor=discount_factor, experience_size=experience_size,
                        update_q_fut=update_q_fut, sample_experience=sample_experience,
                        update_freq=update_freq, no_update_start=no_update_start)
        BasePlayer.__init__(self)

        self.name = "intelligent_player"
        self.counter = 0
        self._init_game_vars()
        self.matches_callbacks = Callbacks(WinRateLog(self.writer, every=10),
                                           WinRateLog(self.writer, every=100),
                                           IncreaseEpsilonOnLose(self, every=50, increase_perc=0.1,
                                                                 start_epsilon_increase=0.3))

    def _init_game_vars(self):
        self._reward = 0
        self._out_of_hand = False
        self.last_winner = True
        self.first_turn = True
        self.id_enemy_discard = -1
        self.my_card_discarded = []
        self.enemy_discarded = []
        self.id_self_discard = -1

    def choose_card(self) -> int:
        i = self.decide(self.build_complete_state())
        if i >= len(self.hand):
            i = randint(0, len(self.hand) - 1)
            self._out_of_hand = True
        self.experience_buffer.put_a_t(i)
        self.id_self_discard = self.hand[i].id
        self.first_turn = False
        return i

    def notify_game_winner(self, name: str):
        self.reset()
        if name == self.name:
            self.get_reward(self.build_complete_state(), 1.0)
            for win_logger in self.matches_callbacks.callbacks:
                win_logger.notify_win()
        else:
            self.get_reward(self.build_complete_state(), -1.0)
        self.counter += 1
        self.matches_callbacks(self.counter)
        self.writer.add_scalar('epsilon', self.epsilon, self.counter)
        self._init_game_vars()

    def on_enemy_discard(self, card):
        self.id_enemy_discard = card.id
        if self.step > 0:
            self.experience_buffer.put_next_enemy_card_id(card.id)

    def notify_turn_winner(self, points: int):
        reward = points / 22.0 / 2.0  # [-0.5, 0.5]
        if self._out_of_hand:
            self._reward = -0.7
            self._out_of_hand = False
        if points > 0:
            self._on_my_win()
        elif points < 0:
            self._on_enemy_win()
        elif self.last_winner:
            self._on_my_win()
        else:
            self._on_enemy_win()
        state_discarded = self.build_complete_state()
        self.get_reward(state_discarded, reward=reward)

    def build_complete_state(self):
        state = build_state_array(self.get_public_state(), self.hand, self.name)
        state_discarded = [state, self.my_card_discarded + self.enemy_discarded, [c.id for c in self.hand]]
        return state_discarded

    def _on_my_win(self):
        self.my_card_discarded.append(self.id_enemy_discard)
        self.my_card_discarded.append(self.id_self_discard)
        self.last_winner = True

    def _on_enemy_win(self):
        self.enemy_discarded.append(self.id_enemy_discard)
        self.enemy_discarded.append(self.id_self_discard)
        self.last_winner = False
