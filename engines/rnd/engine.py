"""
Random engine
"""

from ..core import CoreEngine
import random


class Engine(CoreEngine):

    def __init__(self):
        CoreEngine.__init__(self, 'random')
        return

    def step(self):
        # function to be implemented by children
        move = random.choice(list(self.board.legal_moves))
        self.board.push(move)
        return move, self.board, {}

