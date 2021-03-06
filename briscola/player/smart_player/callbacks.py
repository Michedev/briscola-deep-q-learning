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
        self.loss_log = LossLogger(self.logger, every=1, no_update_start=0)
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
        self.brain.train(True)
        self.brain.requires_grad_(True)
        exp_rew_t, predict_enemy, predict_s_t1 = self.brain(sars.s_t, sars.d_t, predict_enemy=True, predict_next_state=True)
        qloss = self.calc_q_loss(exp_rew_t, sars)
        error_predict = 0.01 * self.calc_error_predict(predict_enemy, sars)
        error_predict_s_t1 = self.calc_error_predict_ns(predict_s_t1, sars)
        tot_loss = qloss + error_predict + error_predict_s_t1
        if self.logger:
            self.loss_log(iteration, qloss, error_predict, error_predict_s_t1, tot_loss)
        tot_loss.backward()
        self.opt.step()
        gc.collect()
        return tot_loss

    def calc_error_predict(self, predict_enemy, sars):
        enemy_cards = sars.enemy_card.long() - 1
        filter_predict = enemy_cards >= 0
        error_predict = predict_enemy[filter_predict].gather(1, enemy_cards[filter_predict].unsqueeze(-1))
        error_predict = - error_predict.mean(dim=0)
        return error_predict

    def calc_q_loss(self, exp_rew_t, sars):
        exp_rew_t = exp_rew_t.gather(1, sars.a_t.long().unsqueeze(-1))
        is_finished_episode = (1 - sars.done)
        is_finished_episode = is_finished_episode.float().unsqueeze(-1)
        exp_rew_t3 = is_finished_episode * self.target_network(sars.s_t1, sars.d_t1)
        exp_rew_t3 = torch.max(exp_rew_t3, dim=1)
        if isinstance(exp_rew_t3, tuple):
            exp_rew_t3 = exp_rew_t3[0]
        y = sars.r_t + self.discount_factor * sars.r_t1 + \
            self.discount_factor ** 2 * sars.r_t2 + \
            self.discount_factor ** 3 * exp_rew_t3
        exp_rew_t = exp_rew_t.squeeze(1)
        qloss = self.mse(y, exp_rew_t)
        return qloss

    def calc_error_predict_ns(self, predict_next_state, sars):
        return self.mse(sars.s_t1, predict_next_state)


class LossLogger(Callback):

    def __init__(self, logger, every=10, no_update_start=0):
        super().__init__(every, no_update_start)
        self.logger = logger

    def call(self, iteration, extra: list):
        qloss, error_predict, error_predict_next_state, tot_loss = extra
        self.logger.add_scalar('loss/q loss', qloss, iteration)
        self.logger.add_scalar('loss/predict_loss', error_predict, iteration)
        self.logger.add_scalar('loss/predict_next_state', error_predict_next_state, iteration)
        self.logger.add_scalars('loss/losses', {'q loss': qloss, 'predict loss': error_predict,
                                                'predict_next_state': error_predict_next_state, 'tot_loss': tot_loss},
                                iteration)
        self.logger.add_scalar('loss/tot_loss', tot_loss, iteration)


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
