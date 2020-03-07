from collections import namedtuple
from typing import Union, Tuple, List
from torch.utils.data import DataLoader
import torch
from path import Path
from card import Deck

SARS = namedtuple('SARS', ['s_t', 'a_t', 'r_t', 's_t1', 'enemy_card'])


class ExperienceBuffer:
    BUFFERPATH = Path(__file__).parent / 'experience.pth'

    def __init__(self, state_size: Union[Tuple, List], experience_size):
        self.state_size = state_size
        self.experience_max_size = experience_size
        self.i_experience = 0
        self.experience_size = 0
        self.experience_buffer = [
            torch.zeros(experience_size, *self.state_size, dtype=torch.float32, device='cpu'),  # s_t
            torch.zeros(experience_size, dtype=torch.int8, device='cpu'),  # action
            torch.zeros(experience_size, dtype=torch.float32, device='cpu'),  # reward_t
            torch.zeros(experience_size, *self.state_size, dtype=torch.float32, device='cpu'),  # s_t1
            torch.zeros(experience_size, dtype=torch.int8, device='cpu')  # enemy card
        ]

    def put_s_t(self, value):
        self.experience_buffer[0][self.i_experience] = value

    def put_a_t(self, value):
        self.experience_buffer[1][self.i_experience] = value

    def put_r_t(self, value):
        self.experience_buffer[2][self.i_experience] = value

    def get_r_t(self, decrease=0):
        return self.experience_buffer[2][self.i_experience - decrease]

    def put_s_t1(self, value, decrease=0):
        self.experience_buffer[3][self.i_experience - decrease] = value

    def put_next_enemy_card_id(self, value):
        self.experience_buffer[4][self.i_experience - 1] = value

    def get_next_enemy_card_id(self, value):
        self.experience_buffer[4][self.i_experience - 1] = value

    def increase_i(self):
        self.i_experience += 1
        self.experience_size = max(self.experience_size, self.i_experience)
        if self.i_experience == self.experience_max_size:
            self.i_experience = 0
            try:
                torch.save(self.experience_buffer, self.BUFFERPATH)
            except Exception as e:
                print("Error when save experience buffer")
                print(e)

    def calc_q_values(self, episode_length, discount_rate):
        rewards = self.experience_buffer[2][-episode_length:]
        qvalues = torch.zeros(episode_length)
        for i in range(episode_length):
            qvalues[i] = rewards[i]
            if i < episode_length - 1:
                qvalues[i] += (rewards[i+1:] * discount_rate ** (torch.arange(1, episode_length+1-i))).sum()
        self.experience_buffer[2][-episode_length:] = qvalues

    def get_all(self):
        return SARS(*[buffer[:self.experience_size] for buffer in self.experience_buffer])

    def reset(self):
        for buffer in self.experience_buffer:
            buffer[:] = 0
        self.i_experience = 0
        self.experience_size = 0

    def batch_it(self):
        d = DataLoader()
