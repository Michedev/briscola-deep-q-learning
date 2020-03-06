import torch
from torch.nn import *
from feature_encoding import encode_card
from card import Deck

values_points = {1: 11, 2: 0, 3: 10, 4: 0, 5: 0, 6: 0, 7: 0, 8: 2, 9: 3, 10: 4}
cards = Deck.all_cards()
card_features = [encode_card(None)] + [encode_card(c) for c in cards]
card_features = torch.cuda.FloatTensor(card_features) if torch.cuda.is_available() else torch.FloatTensor(card_features)


class DiscardedModule(Module):

    def __init__(self, num_patterns=64):
        super(DiscardedModule, self).__init__()
        self.num_patterns = num_patterns
        self.tc1 = self.temporal_conv(1)
        self.tc2 = self.temporal_conv(2)
        self.tc4 = self.temporal_conv(4)

    def forward(self, discarded: torch.Tensor):
        """
        :param discarded: Tensor of size [B x N x 6] (B and N are variables)
        :type discarded: torch.Tensor
        """
        discarded = card_features[discarded.long()]
        discarded = discarded.permute(0, 2, 1)
        output = self.tc1(discarded).mean(dim=-1)
        output += self.tc2(discarded).mean(dim=-1)
        output += self.tc4(discarded).mean(dim=-1)
        return output

    def temporal_conv(self, dilation, ):
        return Sequential(
            Conv1d(6, self.num_patterns, kernel_size=2, dilation=dilation),
            # BatchNorm1d(64),
            ReLU()
        )


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
        self.l1 = Sequential(
            Linear(state_size[-1], self.middle_size),
            BatchNorm1d(self.middle_size),
            ReLU(),
            Linear(self.middle_size, self.middle_size),
            BatchNorm1d(self.middle_size),
            ReLU(),
        )
        self.advantange_nn = Sequential(
            Linear(self.middle_size, 30),
            BatchNorm1d(30),
            ReLU(),
            Linear(30, 3)
        )
        self.state_nn = Sequential(Linear(self.middle_size, 30),
                                   BatchNorm1d(30),
                                   ReLU(),
                                   Linear(30, 1))

        self.discarded_nn = DiscardedModule(self.middle_size)
        self.remaining_nn = DiscardedModule(self.middle_size)
        self.enemy_predict = EnemyNextCard(self.middle_size)

    def forward(self, state: torch.Tensor, others, predict_enemy=False):
        i_seps = others[:, 0]
        mask_remaining = i_seps < torch.arange(41).to(state.device).unsqueeze(1)
        mask_remaining.t_()
        mask_discarded = ~mask_remaining
        discarded = others * mask_discarded.float()
        remaining = others * mask_remaining.float()
        output = self.l1(state)
        output += self.discarded_nn(discarded)
        output += self.remaining_nn(remaining)
        if predict_enemy:
            p_next_enemy_card = self.enemy_predict(output)
        advantage = self.advantange_nn(output)
        state_value = self.state_nn(output)
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        output = state_value + advantage
        if predict_enemy:
            return output, p_next_enemy_card
        return output


__all__ = ['Brain']
