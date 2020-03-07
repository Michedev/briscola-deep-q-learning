from random import randint, random

from torch.utils.tensorboard import SummaryWriter

from base_player import BasePlayer
from brain import Brain
from callbacks import Callbacks, TrainStep, TargetNetworkUpdate, QValuesLog, WeightsLog, WinRateLog, \
    IncreaseEpsilonOnLose
from feature_encoding import build_state_array, build_discarded_remaining_array
from experience_buffer import ExperienceBuffer

GLOBAL_COUNTER = 0
from path import Path

FOLDER = Path(__file__).parent
BRAINFILE = FOLDER / 'brain.pth'

import torch
from radam import RAdam
from torch.optim import Adam


class QAgent:

    def __init__(self, input_size, discount_factor=0.9, experience_size=300_000, update_q_fut=1000,
                 sample_experience=64, update_freq=4, no_update_start=500):
        self.no_update_start = no_update_start
        self.update_freq = update_freq
        self.sample_experience = sample_experience
        self.update_q_fut = update_q_fut
        self.epsilon = 1.0
        self.discount_factor = discount_factor
        self.input_size = list(input_size)
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = Brain(input_size)  # throw first, second, third card
        self.target_network = Brain(input_size)
        if BRAINFILE.exists():
            weights = torch.load(BRAINFILE)
            self.brain.load_state_dict(weights)
            self.target_network.load_state_dict(weights)
            del weights
        self.brain.to(self._device)
        self.target_network.to(self._device)
        self.opt = RAdam(self.brain.parameters(), eps=0.0003)
        self.step = 1
        self.episode = 0
        self.step_episode = 0
        self.mse = torch.nn.MSELoss('mean')
        self.writer = SummaryWriter('briscola_logs')
        self.epsilon = 1.0
        self._curiosity_values = None
        self.same_counter = 0
        self.destination_position = None
        self.experience_buffer = ExperienceBuffer(input_size, experience_size)
        self.decide_callbacks = Callbacks(
            TrainStep(self.opt, self.experience_buffer, self.target_network, self.brain,
                      self.update_freq, self.no_update_start, self.discount_factor, self._device,
                      self.sample_experience, self.writer),
            TargetNetworkUpdate(self.brain, self.target_network, self.update_q_fut),
            WeightsLog(self.brain, self.writer, every=1000, gradient=True),
        )
        self.q_values_log = QValuesLog(self.writer)

    def decide(self, state):
        self.decide_callbacks(self.step)
        table, discarded, hand_ids = state
        table = torch.from_numpy(table).to(self._device).float().unsqueeze(0)
        extra = build_discarded_remaining_array(discarded, hand_ids).to(self._device).unsqueeze(0)
        self.experience_buffer.put_s_t(table)
        self.experience_buffer.put_d_t(extra)
        self.brain.eval()
        if self.epsilon > random():
            i = randint(0, len(hand_ids))
        else:
            q_values = self.brain(table, extra).squeeze(0)
            i = torch.argmax(q_values).item()
            self.q_values_log(self.step, q_values, i)
        return i

    def get_reward(self, next_state, reward):
        table, next_discarded, next_hand_ids = next_state
        table = torch.from_numpy(table).float()
        extra = build_discarded_remaining_array(next_discarded, next_hand_ids)
        self._store_reward(extra, table, reward)
        self.writer.add_scalar('true reward', reward, self.step)
        self.step += 1
        self.step_episode += 1
        self.experience_buffer.increase_i()

    def _store_reward(self, extra, next_state, reward):
        self.experience_buffer.put_r_t(reward)
        self.experience_buffer.put_s_t1(next_state)
        self.experience_buffer.put_d_t1(extra)
        if self.experience_buffer.i_experience >= 1:
            self.experience_buffer.put_r_t1(reward)
            if reward != 1 and self.experience_buffer.get_r_t(decrease=1) != 1:
                self.experience_buffer.put_s_t1(next_state, decrease=1)
                self.experience_buffer.put_d_t1(extra, decrease=1)
        if self.experience_buffer.i_experience >= 2:
            self.experience_buffer.put_r_t2(reward)
            if reward != 1 and \
                    self.experience_buffer.get_r_t1(decrease=1) != 1.0 and \
                    self.experience_buffer.get_r_t(decrease=2) != 1.0:
                self.experience_buffer.put_s_t1(next_state, decrease=2)
                self.experience_buffer.put_d_t1(extra, decrease=2)

    def reset(self):
        torch.save(self.brain.state_dict(), BRAINFILE)
        self.epsilon = max(self.epsilon - 0.001, 0.1)
        self.writer.add_scalar('steps per episode', self.step_episode, self.episode)
        self.episode += 1
        self.step_episode = 0


class SmartPlayer(BasePlayer, QAgent):

    def __init__(self):
        QAgent.__init__(self, [34])
        BasePlayer.__init__(self)

        self.name = "intelligent_player"
        self.counter = 0
        self._init_game_vars()
        self.matches_callbacks = Callbacks(WinRateLog(self.writer, every=10),
                                           WinRateLog(self.writer, every=100),
                                           IncreaseEpsilonOnLose(self, every=20, increase_perc=0.3))

    def _init_game_vars(self):
        self._reward = 0
        self._out_of_hand = False
        self.last_winner = True
        self.first_turn = True
        self.id_enemy_discard = -1
        self.my_card_discarded = []
        self.enemy_discarded = []
        self.id_self_discard = -1
        self.last_state = []

    def choose_card(self) -> int:
        state = build_state_array(self.get_public_state(), self.hand, self.name)
        self.last_state = [state, self.my_card_discarded + self.enemy_discarded, [c.id for c in self.hand]]
        if not self.first_turn:
            self.get_reward(self.last_state, self._out_of_hand)
        i = self.decide(self.last_state)
        if i >= len(self.hand):
            i = randint(0, len(self.hand) - 1)
            self._out_of_hand = True
        self.id_self_discard = self.hand[i].id
        self.first_turn = False
        return i

    def notify_game_winner(self, name: str):
        self.reset()
        if name == self.name:
            self.get_reward(self.last_state, 1.0)
            for win_logger in self.matches_callbacks.callbacks:
                win_logger.notify_win()
        else:
            self.get_reward(self.last_state, -1.0)
        self.counter += 1
        self.matches_callbacks(self.counter)
        self.writer.add_scalar('epsilon', self.epsilon, self.counter)
        self._init_game_vars()

    def on_enemy_discard(self, card):
        self.id_enemy_discard = card.id
        if self.step > 0:
            self.experience_buffer.put_next_enemy_card_id(card.id)

    def notify_turn_winner(self, points: int):
        self._reward = points / 22.0 / 2.0  # [-0.5, 0.5]
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

    def _on_my_win(self):
        self.my_card_discarded.append(self.id_enemy_discard)
        self.my_card_discarded.append(self.id_self_discard)
        self.last_winner = True

    def _on_enemy_win(self):
        self.enemy_discarded.append(self.id_enemy_discard)
        self.enemy_discarded.append(self.id_self_discard)
        self.last_winner = False
