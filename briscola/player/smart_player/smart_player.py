from random import randint, random

from torch.utils.tensorboard import SummaryWriter

from base_player import BasePlayer
from brain import Brain
from callbacks import Callbacks, TrainStep, TargetNetworkUpdate, ProbLog, WeightsLog, WinRateLog, \
    IncreaseEpsilonOnLose
from feature_encoding import build_state_array
from experience_buffer import ExperienceBuffer

GLOBAL_COUNTER = 0
from path import Path

FOLDER = Path(__file__).parent
BRAINFILE = FOLDER / 'brain.pth'

import torch
from radam import RAdam
from pyro.distributions import Categorical


class QAgent:

    def __init__(self, input_size, discount_factor=0.9, update_q_fut=1000,
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
        if BRAINFILE.exists():
            weights = torch.load(BRAINFILE)
            self.brain.load_state_dict(weights)
            del weights
        self.brain.to(self._device)
        self.opt = RAdam(self.brain.parameters(), lr=10e-2)
        self.step = 1
        self.episode = 0
        self.step_episode = 0
        self.mse = torch.nn.MSELoss('mean')
        self.writer = SummaryWriter('briscola_logs_1')
        self.epsilon = 1.0
        self._curiosity_values = None
        self.same_counter = 0
        self.destination_position = None
        self.experience_buffer = ExperienceBuffer(input_size, 10000)
        self.decide_callbacks = Callbacks(
        )
        self.policy_log = ProbLog(self.writer, every=30)

    def decide(self, state):
        self.decide_callbacks(self.step)
        state = torch.from_numpy(state).to(self._device).float().unsqueeze(0)
        self.experience_buffer.put_s_t(state)
        self.brain.eval()
        p_a, value = self.brain(state)
        p_a.squeeze_(0)
        c = Categorical(probs=p_a)
        i = c.sample((1,)).squeeze()
        self.policy_log(self.step, p_a, i)
        return i

    def get_reward(self, next_state, reward):
        next_state = torch.from_numpy(next_state).float()
        self._store_reward(next_state, reward)
        self.writer.add_scalar('true reward', reward, self.step)
        self.step += 1
        self.step_episode += 1
        self.experience_buffer.increase_i()

    def _store_reward(self, next_state, reward):
        self.experience_buffer.put_r_t(reward)
        self.experience_buffer.put_s_t1(next_state)

    def reset(self):
        torch.save(self.brain.state_dict(), BRAINFILE)
        self.epsilon = max(self.epsilon - 0.001, 0.1)
        self.writer.add_scalar('steps per episode', self.step_episode, self.episode)
        self.episode += 1
        self.step_episode = 0
        self.experience_buffer.reset()


class SmartPlayer(BasePlayer, QAgent):

    def __init__(self):
        QAgent.__init__(self, input_size=[34])
        BasePlayer.__init__(self)

        self.name = "intelligent_player"
        self.counter = 0
        self._init_game_vars()
        self.matches_callbacks = Callbacks(WinRateLog(self.writer, every=10),
                                           WinRateLog(self.writer, every=100),
                                           IncreaseEpsilonOnLose(self, every=20, increase_perc=0.3),
                                           TrainStep(self.opt, self.experience_buffer, None, self.brain,
                                                     10, 0, self.discount_factor, self._device,
                                                     self.sample_experience, self.writer),
                                           )

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
        state = build_state_array(self.get_public_state(), self.hand, self.name)
        if not self.first_turn:
            self.get_reward(state, self._reward)
        i = self.decide(state)
        if i >= len(self.hand):
            i = randint(0, len(self.hand) - 1)
            self._out_of_hand = True
        self.id_self_discard = self.hand[i].id
        self.experience_buffer.put_a_t(i)
        self.first_turn = False
        return i

    def notify_game_winner(self, name: str):
        self.experience_buffer.set_done()
        self.matches_callbacks(self.counter)
        self.reset()
        if name == self.name:
            for win_logger in self.matches_callbacks.callbacks:
                if hasattr(win_logger, 'notify_win'):
                    win_logger.notify_win()
        self.counter += 1
        self.writer.add_scalar('epsilon', self.epsilon, self.counter)
        self._init_game_vars()

    def on_enemy_discard(self, card):
        self.id_enemy_discard = card.id
        if not self.first_turn:
            self.experience_buffer.put_next_enemy_card_id(card.id)

    def notify_turn_winner(self, points: int):
        self._reward = points / 22.0 # [-0.5, 0.5]
        if self._out_of_hand:
            self._reward = -0.7
            self._out_of_hand = False
        if points > 0:
            self._on_my_turn_win()
        elif points < 0:
            self._on_enemy_turn_win()
        elif self.last_winner:
            self._on_my_turn_win()
        else:
            self._on_enemy_turn_win()

    def _on_my_turn_win(self):
        self.my_card_discarded.append(self.id_enemy_discard)
        self.my_card_discarded.append(self.id_self_discard)
        self.last_winner = True

    def _on_enemy_turn_win(self):
        self.enemy_discarded.append(self.id_enemy_discard)
        self.enemy_discarded.append(self.id_self_discard)
        self.last_winner = False
