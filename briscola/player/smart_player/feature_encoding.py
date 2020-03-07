from typing import List, Union

import numpy as np
import torch

from card import Card
from seed import Seed
from game_rules import values_points


def build_state_array(public_state, hand: List[Card], pname: str) -> np.ndarray:
    x = np.zeros((34,))
    x[:18] = -1
    for i, c in enumerate(hand):
        range_i = slice(i * 6, (i + 1) * 6)
        x[range_i] = encode_card(c)
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


def encode_card(c: Union[Card, None]) -> np.ndarray:
    if c is None:
        return np.zeros((6,)) -1
    ohe_seed = Seed.ohe_repr(c.seed)
    points_card = values_points[c.value] / 11.0
    value_card = c.value / 10.0
    features = np.array([value_card, points_card])
    return np.concatenate([features, ohe_seed])
