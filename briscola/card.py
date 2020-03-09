from dataclasses import dataclass, field
from game_rules import values_points
from briscola.seed import Seed
from random import shuffle


@dataclass()
class Card:
    id: int
    value: int
    seed: int

    def __post_init__(self):
        self.points = values_points[self.value]


class Deck:
    __slots__ = ['cards']

    def __init__(self):
        self.cards = self.all_cards()
        shuffle(self.cards)

    @classmethod
    def all_cards(cls):
        return [Card(i+1, i % 10 + 1, Seed.get_seed(i // 10)) for i in range(40)]

    def draw(self):
        return self.cards.pop(0)

    def is_empty(self) -> bool:
        return len(self.cards) == 0
