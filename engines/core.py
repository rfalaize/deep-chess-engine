"""
Core Engine is the parent class of all chess engines.

Children engines inherit from it and call the GenerateMove function to return a move.
GenerateMove takes a 'compute' function as input, which is implemented by the children.
GenerateMove wraps the call to 'compute' and adds standard metrics to the results (elapsed time... etc).
"""

import chess
import logging
from datetime import datetime


class CoreEngine:

    def __init__(self, engine_name='core'):
        self.board = chess.Board()
        self.engine_name = engine_name
        return

    def decode_fen(self, fen):
        return fen.replace('_', ' ')

    def generate_move(self, fen):
        result = {}
        start_time = datetime.now()

        try:
            decoded_fen = self.decode_fen(fen)
            self.board = chess.Board(decoded_fen)

            # search next move using the specified 'Step' function
            move, board, stats = self.step()

            # return response
            result['status'] = 'success'
            result['move'] = str(move)
            result['board'] = board.fen()
            result['input'] = decoded_fen
            result['isCheckMate'] = board.is_checkmate()

        except Exception as e:
            result['status'] = 'error'
            result['message'] = str(e)
            stats = {}
            logging.error("Error during Step function:", e)

        end_time = datetime.now()
        core_stats = {'elapsed_time': (end_time - start_time).total_seconds()}
        if 'nodes_evaluated' in stats and stats['nodes_evaluated'] > 0:
            core_stats['ms_per_move'] = 1000 * core_stats['elapsed_time'] / stats['nodes_evaluated']
        result['stats'] = {**stats, **core_stats}   # join 2 dictionaries

        return result

    def step(self):
        # function to be implemented by children
        # return move, self.board <-- return format
        raise NotImplementedError
