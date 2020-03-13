from random import random, randint

from path import Path
from torch.utils.tensorboard import SummaryWriter

from brain import Brain
from callbacks import *
from experience_buffer import ExperienceBuffer
from feature_encoding import build_discarded_remaining_array
from radam import RAdam

FOLDER = Path(__file__).parent
BRAINFILE = FOLDER / 'brain.pth'


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
        self.writer = SummaryWriter('briscola_tb_logs')
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
            i = randint(0, len(hand_ids)-1)
        else:
            q_values = self.brain(table, extra, predict_enemy=False, predict_next_state=False).squeeze(0)
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