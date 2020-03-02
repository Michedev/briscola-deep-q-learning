from experience_buffer import SARS
from abc import ABC, abstractmethod
import torch
import gc


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

    def call(self, iteration, extra: list):
        self.logger.add_scalar(f'match/win_rate_{self.every}',
                               self.wins / self.every,
                               iteration // self.every)
        self.logger.add_scalar(f'match/lose_rate_{self.every}',
                               1 - self.wins / self.every,
                               iteration // self.every)
        self.wins = 0


class TrainStep(Callback):

    def __init__(self, opt, buffer, target_network, brain, every,
                 no_update_start, discount_factor,
                 device, batch_size, logger=None):
        self.logger = logger
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.buffer = buffer
        self.no_update_start = no_update_start
        self.every = every
        self.target_network = target_network
        self.opt = opt
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.brain = brain
        self.device = device
        super().__init__(every, no_update_start)

    def put_into_device(self, sars):
        result = SARS(*[v.to(self.device) for v in sars])
        return result

    def call(self, iteration, extra: list):
        self.brain.train()
        sars = self.buffer.sample(self.batch_size)
        self.opt.zero_grad()
        sars = self.put_into_device(sars)
        exp_rew_t = self.brain(sars.s_t, sars.d_t)
        exp_rew_t = exp_rew_t[:, sars.a_t.long()]
        is_finished_episode = ((torch.ne(sars.r_t, 1.0) & torch.ne(sars.r_t1, 1.0)) & torch.ne(sars.r_t2, 1.0))
        is_finished_episode = is_finished_episode.float().unsqueeze(-1)
        exp_rew_t3 = is_finished_episode * self.target_network(sars.s_t1, sars.d_t1)
        exp_rew_t3 = torch.max(exp_rew_t3, dim=1)
        if isinstance(exp_rew_t3, tuple):
            exp_rew_t3 = exp_rew_t3[0]
        y = sars.r_t + self.discount_factor * sars.r_t1 + \
            self.discount_factor ** 2 * sars.r_t2 + \
            self.discount_factor ** 3 * exp_rew_t3
        qloss = self.mse(y, exp_rew_t)
        del sars
        qloss = torch.mean(qloss)
        if self.logger:
            self.logger.add_scalar('q loss', qloss, iteration)
        qloss.backward()
        gc.collect()
        self.opt.step()
        return qloss


class TargetNetworkUpdate(Callback):

    def __init__(self, brain, target, every=None):
        self.target = target
        self.brain = brain
        self.every = every or 1000
        super().__init__(self.every)

    def call(self, iteration, extra: list):
        self.target.load_state_dict(self.brain.state_dict())


class QValuesLog(Callback):

    def __init__(self, logger, every=10):
        super().__init__(every)
        self.logger = logger

    def call(self, iteration, extra: list):
        q_values, action = extra
        self.logger.add_scalars('q_values',
                                {'first': q_values[0], 'second': q_values[1], 'third': q_values[2]},
                                iteration)
        self.logger.add_scalar('q_values/first', q_values[0], iteration)
        self.logger.add_scalar('q_values/second', q_values[1], iteration)
        self.logger.add_scalar('q_values/third', q_values[2], iteration)
        self.logger.add_scalar('q_values/action', action, iteration)


class Callbacks:

    def __init__(self, *callbacks):
        self.callbacks = callbacks

    def __call__(self, iteration):
        for callback in self.callbacks:
            callback(iteration)


class WeightsLog(Callback):

    def __init__(self, brain, logger, every=None):
        self.logger = logger
        self.brain = brain
        self.every = every or 100
        super().__init__(self.every)

    def call(self, iteration, extra: list):
        for modname, parmas in self.brain.state_dict().items():
            self.logger.add_histogram(modname.replace('.', '/'), parmas, iteration)
