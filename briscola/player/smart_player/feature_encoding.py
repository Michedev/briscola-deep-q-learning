from typing import List, Final

import numpy as np
import torch

from card import Card
from seed import Seed
from game_rules import values_points, select_winner
from state import PublicState

STATE_SIZE: Final = 40


def build_state_array(public_state, hand: List[Card], pname: str) -> np.ndarray:
    x = np.zeros((STATE_SIZE,)) -1
    for i, c in enumerate(hand):
        range_i = slice(i * 6, (i + 1) * 6)
        x[range_i] = encode_card(c)
        if len(public_state.table) > 0:
            x[-len(hand) + i] = select_winner(public_state.table + [c], public_state.briscola)
            x[-len(hand) * 2 + i] = (c.points + public_state.table[0].points) / 22.0
    offset = 18
    if len(public_state.table) > 0:
        x[offset:offset + 6] = encode_card(public_state.table[0])
    offset = 24
    x[offset:offset + 2] = public_state.points
    x[offset:offset + 2] /= 60.0
    offset = 26
    x[offset:offset + 6] = encode_card(public_state.briscola)
    offset = 32
    x[offset] = public_state.order[0] == pname
    offset = 33
    x[offset] = public_state.turn / 23.0
    return x


def build_x_discarded(state: 'PublicState', hand: List[Card]):
    x = np.zeros(shape=(1, 6, 4 * len(state.discarded), 1), dtype='float32')
    if len(state.discarded) == 0:
        x = np.zeros(shape=(1, 6, 4, 1), dtype='float32')
        return x
    for j, c in enumerate(state.discarded):
        for i in range(3):
            if i < len(hand):
                x[:, :, j + i] = encode_card(hand[i]).reshape((1, 6, 1))
            else:
                x[:, :, j + i] = encode_card(None).reshape((1, 6, 1))
        x[:, :, j + 3] = encode_card(c).reshape((1, 6, 1))
    return x


def encode_card(c: Card) -> np.ndarray:
    if c is None:
        return np.zeros((6,)) -1
    ohe_seed = Seed.ohe_repr(c.seed)
    points_card = values_points[c.value] / 11.0
    value_card = c.value / 10.0
    features = np.array([value_card, points_card])
    return np.concatenate([features, ohe_seed])


def build_discarded_remaining_array(discarded, hand_ids):
    remaining = set(range(40)).difference(set(discarded)).difference(set(hand_ids))
    remaining = list(remaining)
    i_split = len(discarded)
    extra = torch.zeros(41)
    extra[0] = i_split
    extra[1:len(discarded) + 1] = torch.FloatTensor(discarded)
    extra[len(discarded) + 1:len(discarded) + 1 + len(remaining)] = torch.FloatTensor(remaining)
    return extra