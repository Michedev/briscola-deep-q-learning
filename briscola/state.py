from dataclasses import dataclass
from typing import *

from briscola.card import Card


@dataclass(frozen=True)
class PublicState:
    points: Tuple[int, int]
    table: List[Card]
    discarded: List[Card]
    turn: int
    briscola: Card
    order: List[str]
