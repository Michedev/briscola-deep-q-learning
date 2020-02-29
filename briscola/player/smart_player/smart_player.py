from random import randint, random

import gc
from torch.utils.tensorboard import SummaryWriter

from base_player import BasePlayer
from brain import Brain
from feature_encoding import build_state_array, build_discarded_remaining_array
from experience_buffer import ExperienceBuffer, SARS

values_points = {1: 11, 2: 0, 3: 10, 4: 0, 5: 0, 6: 0, 7: 0, 8: 2, 9: 3, 10: 4}
GLOBAL_COUNTER = 0
from path import Path

FOLDER = Path(__file__).parent
BRAINFILE = FOLDER / 'brain.pth'

import math
import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class QAgent:

    def __init__(self, input_size, discount_factor=0.8, experience_size=300_000, update_q_fut=1000,
                 sample_experience=128, update_freq=60, no_update_start=500):
        '''

        :param input_size:
        :param discount_factor:
        :param experience_size:
        :param update_q_fut:
        :param sample_experience: sample size drawn from the buffer
        :param update_freq: number of steps for a model update
        :param no_update_start: number of initial steps which the model doesn't update
        '''
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
            self.brain.load_state_dict(BRAINFILE)
        self.target_network = Brain(input_size)
        self.opt = RAdam(self.brain.parameters())
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

    def decide(self, state):
        self.__triggers_decide()
        state, discarded, hand_ids = state
        state = torch.from_numpy(state).to(self._device)
        extra = build_discarded_remaining_array(discarded, hand_ids).unsqueeze(0)
        self.experience_buffer.put_d_t(extra)
        if self.epsilon > random():
            i = randint(0, len(hand_ids))
        else:
            q_values = self.brain(state, discarded).item()
            i = torch.argmax(q_values)
        return i

    def get_reward(self, next_state, reward):
        next_state, next_discarded, next_hand_ids = next_state
        next_state = torch.from_numpy(next_state)
        extra = build_discarded_remaining_array(next_discarded, next_hand_ids)
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
        self.writer.add_scalar('true reward', reward, self.step)
        self.step += 1
        self.step_episode += 1
        self.experience_buffer.increase_i()

    def _train_step(self, sars, discount_factor):
        self.opt.zero_grad()
        self._put_into_device(sars)
        exp_rew_t = self.brain(sars.s_t, sars.d_t)
        exp_rew_t = exp_rew_t[:, sars.a_t.long()]
        is_finished_episode = ((torch.ne(sars.r_t, 1.0) & torch.ne(sars.r_t1, 1.0)) & torch.ne(sars.r_t2, 1.0))
        is_finished_episode = is_finished_episode.float().unsqueeze(-1)
        exp_rew_t3 = is_finished_episode * self.target_network(sars.s_t1, sars.d_t1)
        exp_rew_t3 = torch.max(exp_rew_t3, dim=1)
        if isinstance(exp_rew_t3, tuple):
            exp_rew_t3 = exp_rew_t3[0]
        y = sars.r_t + discount_factor * sars.r_t1 + \
            discount_factor ** 2 * sars.r_t2 + \
            discount_factor ** 3 * exp_rew_t3
        qloss = self.mse(y, exp_rew_t)
        del sars
        qloss = torch.mean(qloss)
        qloss.backward()
        gc.collect()
        self.opt.step()
        return qloss

    def _save_nn(self):
        for lname, lvalues in self.brain.state_dict().items():
            self.writer.add_histogram(lname.replace('.', '/'), lvalues, self.step)

    def reset(self):
        self.epsilon = max(1.0 - 0.004 * self.episode, 0.1)
        self.step_episode = 0

    def on_finish(self, winner):
        self.writer.add_scalar('steps per episode', self.step_episode, self.episode)
        self.episode += 1

    def _put_into_device(self, sars):
        result = SARS(*[v.to(self._device) for v in sars])
        return result

    def __triggers_decide(self):
        if self.step % self.update_freq == 0 and self.step > self.no_update_start:
            sample = self.experience_buffer.sample(128)
            self._train_step(sample, self.discount_factor)
        if self.step % 1000 == 0 and self.step > 0:
            gc.collect()
            self.target_network.load_state_dict(self.brain.state_dict())


class SmartPlayer(BasePlayer, QAgent):

    def __init__(self):
        QAgent.__init__(self, [34])
        BasePlayer.__init__(self)

        self.name = "intelligent_player"
        self.counter_wins = 0
        self.counter = 0
        self._init_game_vars()

    def _init_game_vars(self):
        self._reward = 0
        self.last_winner = True
        self.id_enemy_discard = -1
        self.my_card_discarded = []
        self.enemy_discarded = []
        self.id_self_discard = -1
        self.last_state = []

    def choose_card(self) -> int:
        state = build_state_array(self.get_public_state(), self.hand, self.name)
        self.last_state = [state, self.my_card_discarded + self.enemy_discarded, [c.id for c in self.hand]]
        if self.step_episode > 0:
            self.get_reward(self.last_state, self._reward)
        i = self.decide(self.last_state)
        if i >= len(self.hand):
            i = randint(0, len(self.hand) - 1)
            self._reward = -0.7
        self.id_self_discard = self.hand[i].id
        return i

    def notify_game_winner(self, name: str):
        self.reset()
        if name == self.name:
            self.get_reward(self.last_state, 1.0)
            self.counter_wins += 1
        else:
            self.get_reward(self.last_state, -1.0)
        self.counter += 1
        self._init_game_vars()


    def on_enemy_discard(self, card):
        self.id_enemy_discard = card.id

    def notify_turn_winner(self, points: int):
        self._reward = points / 22.0 / 2.0  # [-0.5, 0.5]
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
