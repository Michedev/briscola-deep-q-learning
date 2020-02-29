from dataclasses import dataclass

from briscola.seed import Seed
from random import shuffle


@dataclass(frozen=True)
class Card:
    id: int
    value: int
    seed: int


class Deck:
    __slots__ = ['cards']

    def __init__(self):
        self.gen_deck()
        shuffle(self.cards)

    @classmethod
    def all_cards(cls):
        return [Card(i+1, i % 10 + 1, Seed.get_seed(i // 10)) for i in range(40)]

    def gen_deck(self):
        self.cards = self.all_cards()

    def draw(self):
        return self.cards.pop(0)

    def is_empty(self) -> bool:
        return len(self.cards) == 0
