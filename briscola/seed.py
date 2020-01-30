from typing import Union
import numpy as np


class Seed:
    Hearts = 1
    Clubs = 2
    Spades = 3
    Diamonds = 4
    __ohe = np.eye(4)

    @classmethod
    def get_seed(cls, i: Union[str, int]):
        if isinstance(i, str):
            return cls.__dict__[i.capitalize()]
        if i == 0:
            return cls.Hearts
        elif i == 1:
            return cls.Clubs
        elif i == 2:
            return cls.Spades
        elif i == 3:
            return cls.Diamonds
        else:
            raise ValueError(f"input {i} should be between [0, 3]")

    @classmethod
    def ohe_repr(cls, seed):
        return cls.__ohe[seed - 1, :]
