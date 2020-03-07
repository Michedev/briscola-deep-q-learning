import torch
from torch.nn import *


def EnemyNextCard(input_size):
    return Sequential(
        Linear(input_size, input_size),
        ReLU(),
        Linear(input_size, 40),
        LogSoftmax(dim=1)
    )


class Brain(Module):

    def __init__(self, state_size: list):
        super(Brain, self).__init__()
        self.middle_size = 35
        self.common_features = Sequential(
            BatchNorm1d(state_size[-1]),
            Linear(state_size[-1], self.middle_size),
            ReLU(),
            Linear(self.middle_size, self.middle_size),
            ReLU()
        )
        self.recurrent = GRU(self.middle_size, self.middle_size, 2)
        self.policy_nn = Sequential(
            Linear(self.middle_size, 30),
            ReLU(),
            Linear(30, 3),
            Softmax(dim=1)
        )
        self.state_nn = Sequential(Linear(self.middle_size, 30),
                                   ReLU(),
                                   Linear(30, 1))

        self.enemy_predict = EnemyNextCard(self.middle_size)

    def forward(self, state: torch.Tensor, predict_enemy=False):
        output = self.common_features(state)
        output = output.unsqueeze(1)
        output, hidden = self.recurrent(output)
        output = output.squeeze(1)
        p_a = self.policy_nn(output)
        state_value = self.state_nn(output)
        if predict_enemy:
            p_next_enemy_card = self.enemy_predict(output)
            return p_a, state_value, p_next_enemy_card
        return p_a, state_value


__all__ = ['Brain']
