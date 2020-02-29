from collections import namedtuple
from typing import Union, Tuple, List

import torch

from card import Deck

SARS = namedtuple('SARS', ['s_t', 'd_t', 'a_t', 'r_t', 's_t1', 'd_t1', 'r_t1', 'r_t2'])


class ExperienceBuffer:

    def __init__(self, state_size: Union[Tuple, List], experience_size):
        self.state_size = state_size
        self.experience_max_size = experience_size
        self.i_experience = 0
        self.experience_size = 0
        self.experience_buffer = [
            torch.zeros(experience_size, *self.state_size, dtype=torch.float32, device='cpu'),  # s_t
            torch.zeros(experience_size, 41, dtype=torch.float32, device='cpu'),  # discarded/remaining cards at time t
            torch.zeros(experience_size, dtype=torch.int8, device='cpu'),  # action
            torch.zeros(experience_size, dtype=torch.float32, device='cpu'),  # reward_t
            torch.zeros(experience_size, *self.state_size, dtype=torch.float32, device='cpu'),  # s_t1
            torch.zeros(experience_size, 41, dtype=torch.float32, device='cpu'),  # discarded/remaining cards at time t1
            torch.zeros(experience_size, dtype=torch.float32, device='cpu'),  # reward_t1
            torch.zeros(experience_size, dtype=torch.float32, device='cpu')  # reward_t2
        ]

    def put_s_t(self, value):
        self.experience_buffer[0][self.i_experience] = value

    def put_d_t(self, value):
        self.experience_buffer[1][self.i_experience] = value

    def put_a_t(self, value):
        self.experience_buffer[2][self.i_experience] = value

    def put_r_t(self, value):
        self.experience_buffer[3][self.i_experience] = value

    def get_r_t(self, decrease=0):
        return self.experience_buffer[3][self.i_experience - decrease]

    def put_s_t1(self, value, decrease=0):
        self.experience_buffer[4][self.i_experience - decrease] = value

    def put_d_t1(self, value, decrease=0):
        self.experience_buffer[5][self.i_experience - decrease] = value

    def put_r_t1(self, value):
        self.experience_buffer[6][self.i_experience - 1] = value

    def get_r_t1(self, decrease=0):
        return self.experience_buffer[6][self.i_experience - decrease]

    def put_r_t2(self, value):
        self.experience_buffer[7][self.i_experience - 2] = value

    def get_r_t2(self, decrease=0):
        return self.experience_buffer[7][self.i_experience - decrease]

    def increase_i(self):
        self.i_experience += 1
        self.experience_size = max(self.experience_size, self.i_experience)
        if self.i_experience == self.experience_max_size:
            self.i_experience = 0

    def sample(self, batch_size=128):
        i_batch = torch.randint(0, self.experience_size, [batch_size])
        batch = [buffer[i_batch] for buffer in self.experience_buffer]
        return SARS(*batch)
