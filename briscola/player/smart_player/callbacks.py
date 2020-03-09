from math import ceil

from experience_buffer import SARS, ExperienceBuffer
from abc import ABC, abstractmethod
import torch
import gc

from torch.optim.adamw import AdamW
from torch.optim import SGD, Adam
from radam import RAdam


class Callback(ABC):

    def __init__(self, every, no_update_start=None):
        self.no_update_start = no_update_start or 0
        self.every = every

    def __call__(self, iteration, *extra):
        if iteration > self.no_update_start and iteration % self.every == 0:
            self.call(iteration, extra)

    @abstractmethod
    def call(self, iteration, extra: list):
        pass


class WinRateLog(Callback):

    def __init__(self, logger, every=None):
        self.every = every or 10
        self.logger = logger
        self.wins = 0
        super().__init__(self.every)

    def notify_win(self):
        self.wins += 1

    @property
    def win_rate(self):
        return self.wins / self.every

    def call(self, iteration, extra: list):
        self.logger.add_scalar(f'match/win_rate_{self.every}',
                               self.win_rate,
                               iteration // self.every)
        self.logger.add_scalar(f'match/lose_rate_{self.every}',
                               1 - self.win_rate,
                               iteration // self.every)
        self.wins = 0


class IncreaseEpsilonOnLose(Callback):

    def __init__(self, player: {'epsilon'}, every=10, increase_perc=0.4, start_epsilon_increase=0.5):
        super().__init__(every, no_update_start=100)
        self.start_epsilon_increase = start_epsilon_increase
        self.player = player
        self.wins = 0
        self.increase_perc = increase_perc

    def notify_win(self):
        self.wins += 1

    @property
    def win_rate(self):
        return self.wins / self.every

    def call(self, iteration, extra: list):
        if self.player.epsilon < self.start_epsilon_increase:
            self.player.epsilon += (1 - self.win_rate) * self.increase_perc
        self.wins = 0


class BatchIndexIterator(object):

    def __init__(self, length, batch_size):
        self.batch_size = batch_size
        self.length = length
        self.num_batches = ceil(length / batch_size)
        self.slices = []
        for i in range(self.num_batches):
            batch_slice = slice(i * batch_size, (i+1) * batch_size)
            if i == self.num_batches - 1:
                batch_slice = slice(i * batch_size, length)
            self.slices.append(batch_slice)

    def __iter__(self):
        return iter(self.slices)


class TrainStep(Callback):

    def __init__(self, opt, buffer, target_network, brain, every,
                 no_update_start, discount_factor,
                 device, batch_size, logger=None):
        self.logger = logger
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.buffer: ExperienceBuffer = buffer
        self.no_update_start = no_update_start
        self.opt = Adam(brain.parameters())
        self.every = every
        self.target_network = target_network
        self.batch_counter = 0
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.brain = brain
        self.device = device
        self.log_weights = WeightsLog(self.brain, self.logger, gradient=True)
        super().__init__(every, no_update_start)

    def put_into_device(self, sars):
        result = SARS(*[v.to(self.device) for v in sars])
        return result

    def call(self, iteration, extra: list):
        self.brain.train()
        episodes = self.buffer.get_all()
        episodes = self.put_into_device(episodes)
        self.brain.train(True)
        self.brain.requires_grad_(True)
        for i_batch in BatchIndexIterator(self.buffer.experience_size, self.batch_size):
            p_a, state_value = self.brain(episodes.s_t[i_batch], predict_enemy=False)
            p_a = p_a.gather(1, episodes.a_t[i_batch].long().unsqueeze(-1)).squeeze()
            state_value = state_value.squeeze(1)
            _, state_value_next = self.brain(episodes.s_t1[i_batch])
            state_value_next = state_value_next.squeeze(1)
            advantage = (1 - episodes.done[i_batch]) * self.discount_factor * (state_value_next - state_value) + episodes.r_t[i_batch]
            policy_loss = torch.log(p_a) * advantage.detach()
            policy_loss = - policy_loss.mean(dim=0)
            state_loss = (advantage ** 2).mean(dim=0)
            tot_loss = state_loss + policy_loss
            tot_loss.backward()
            if self.batch_counter % 10 == 0:
                self.log_weights.call(iteration, [])
            self.opt.step()
            if self.logger:
                self.logger.add_scalar('loss/policy_loss', policy_loss, self.batch_counter)
                self.logger.add_scalar('loss/state_value_loss', state_loss, self.batch_counter)
                self.logger.add_scalar('loss/tot_loss', tot_loss, self.batch_counter)
                self.logger.add_scalars('loss/losses', {'policy_loss': policy_loss, 'state loss': state_loss,
                                                        'tot_loss': tot_loss},
                                        self.batch_counter)
            self.batch_counter += 1
        gc.collect()
        torch.save(self.opt.state_dict(), 'opt_state.pt')


class TargetNetworkUpdate(Callback):

    def __init__(self, brain, target, every=None):
        self.target = target
        self.brain = brain
        self.every = every or 1000
        super().__init__(self.every)

    def call(self, iteration, extra: list):
        self.target.load_state_dict(self.brain.state_dict())


class ProbLog(Callback):

    def __init__(self, logger, every=10):
        super().__init__(every)
        self.logger = logger

    def call(self, iteration, extra: list):
        p_a, action, state_value = extra
        self.logger.add_scalars('policy',
                                {'first': p_a[0], 'second': p_a[1], 'third': p_a[2]},
                                iteration)
        self.logger.add_scalar('policy/first_card', p_a[0], iteration)
        self.logger.add_scalar('policy/second_card', p_a[1], iteration)
        self.logger.add_scalar('policy/third_card', p_a[2], iteration)
        self.logger.add_scalar('policy/action', action, iteration)
        self.logger.add_scalar('policy/state_value', state_value, iteration)



class Callbacks:

    def __init__(self, *callbacks):
        self.callbacks = callbacks

    def __call__(self, iteration):
        for callback in self.callbacks:
            callback(iteration)

    def __getitem__(self, item):
        return self.callbacks[item]


class WeightsLog(Callback):

    def __init__(self, brain, logger, every=None, gradient=False):
        self.logger = logger
        self.brain = brain
        self.every = every or 100
        self.gradient = gradient
        super().__init__(self.every)

    def call(self, iteration, extra: list):
        for modname, params in self.brain.state_dict().items():
            path_module = modname.replace('.', '/')
            self.logger.add_histogram(path_module, params, iteration)
            if params.grad is not None:
                self.logger.add_histogram(path_module + '/grad', params.grad, iteration)
