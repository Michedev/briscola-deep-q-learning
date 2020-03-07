import torch
from torch.nn import *


def EnemyNextCard(input_size):
    return Sequential(
        Linear(input_size, input_size),
        BatchNorm1d(input_size),
        ReLU(),
        Linear(input_size, 40),
        LogSoftmax()
    )


class Brain(Module):

    def __init__(self, state_size: list):
        super(Brain, self).__init__()
        self.middle_size = 256
        self.common_features = Sequential(
            Linear(state_size[-1], self.middle_size),
            BatchNorm1d(self.middle_size),
            ReLU(),
            Linear(self.middle_size, self.middle_size),
            BatchNorm1d(self.middle_size),
            ReLU(),
            GRU(self.middle_size, self.middle_size, 1)
        )
        self.policy_nn = Sequential(
            Linear(self.middle_size, 30),
            BatchNorm1d(30),
            ReLU(),
            Linear(30, 3),
            Softmax(dim=1)
        )
        self.state_nn = Sequential(Linear(self.middle_size, 30),
                                   BatchNorm1d(30),
                                   ReLU(),
                                   Linear(30, 1))

        self.enemy_predict = EnemyNextCard(self.middle_size)

    def forward(self, state: torch.Tensor, predict_enemy=False):
        output = self.common_features(state)
        p_a = self.policy_nn(output)
        state_value = self.state_nn(output)
        if predict_enemy:
            p_next_enemy_card = self.enemy_predict(output)
            return p_a, state_value, p_next_enemy_card
        return p_a, state_value


__all__ = ['Brain']
