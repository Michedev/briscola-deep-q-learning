from base_player import BasePlayer
from random import randint, random
from game_rules import select_winner


class EpsGreedyPlayer(BasePlayer):

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.name = 'EpsGreedyPlayer'

    def choose_card(self) -> int:
        if self.epsilon > random():
            return randint(0, len(self.hand)-1)
        return self.greedy_action()

    def greedy_action(self):
        im_first = self.get_public_state().order[0] == self.name
        if im_first:
            return self.card_min_points()
        else:
            return self.card_max_gain()

    def card_max_gain(self):
        i_max = -1
        max_gain = -100
        state = self.get_public_state()
        table = state.table[:]
        table.append(None)
        for i, c in enumerate(self.hand):
            table[-1] = c
            winner = select_winner(table, state.briscola)
            coef_pts = 1 if winner else -1
            gain = coef_pts * sum(map(lambda c: c.points, table))
            if gain > max_gain:
                i_max = i
                max_gain = gain
        return i_max

    def card_min_points(self):
        i_min = -1
        min_pts = 1000
        for i, c in enumerate(self.hand):
            if c.points < min_pts:
                i_min = i
                min_pts = c.points
        return i_min
