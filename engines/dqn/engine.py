from ...core import CoreEngine
import random

class Engine(CoreEngine):

    def __init__(self):
        CoreEngine.__init__(self, 'dqn.v1')
        return

    def Step(self):
        # function to be implemented by children
        move = random.choice(list(self.board.legal_moves))
        self.board.push(move)
        return move, self.board


# request handler
def handleRequest(context):
    engine = Engine()
    return engine.HandlePostRequest(context)