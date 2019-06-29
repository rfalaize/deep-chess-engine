from ...core import CoreEngine

class Engine(CoreEngine):

    def __init__(self):
        CoreEngine.__init__(self, 'minimax.v1')

        self.MAX_SCORE = 999999

        # *************************************************
        # piece square table evaluation
        # source: https://www.chessprogramming.org/Simplified_Evaluation_Function
        # *************************************************
        self.SCORES = {'W': {}, 'B':{}}

        self.SCORES['W']['PAWN'] = \
            [0, 0, 0, 0, 0, 0, 0, 0,
             5, 10, 10, -20, -20, 10, 10, 5,
             5, -5, -10, 0, 0, -10, -5, 5,
             0, 0, 0, 20, 20, 0, 0, 0,
             5, 5, 10, 25, 25, 10, 5, 5,
             10, 10, 20, 30, 30, 20, 10, 10,
             50, 50, 50, 50, 50, 50, 50, 50,
             0, 0, 0, 0, 0, 0, 0, 0]

        self.SCORES['W']['KNIGHT'] = \
            [-50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50]

        self.SCORES['W']['BISHOP'] = \
            [-20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20]

        self.SCORES['W']['ROOK'] = \
            [0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, 10, 10, 10, 10, 5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            0, 0, 0, 5, 5, 0, 0, 0]

        self.SCORES['W']['QUEEN'] = \
            [-20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20]

        self.SCORES['W']['KING'] = \
            [20, 30, 10, 0, 0, 10, 30, 20,
             20, 20, 0, 0, 0, 0, 20, 20,
             -10, -20, -20, -20, -20, -20, -20, -10,
             -20, -30, -30, -40, -40, -30, -30, -20,
             -30, -40, -40, -50, -50, -40, -40, -30,
             -30, -40, -40, -50, -50, -40, -40, -30,
             -30, -40, -40, -50, -50, -40, -40, -30,
             -30, -40, -40, -50, -50, -40, -40, -30]

        # mirror score tables for black
        self.SCORES['B']['PAWN'] = self.MirrorScore(self.SCORES['W']['PAWN'])
        self.SCORES['B']['KNIGHT'] = self.MirrorScore(self.SCORES['W']['KNIGHT'])
        self.SCORES['B']['BISHOP'] = self.MirrorScore(self.SCORES['W']['BISHOP'])
        self.SCORES['B']['ROOK'] = self.MirrorScore(self.SCORES['W']['ROOK'])
        self.SCORES['B']['QUEEN'] = self.MirrorScore(self.SCORES['W']['QUEEN'])
        self.SCORES['B']['KING'] = self.MirrorScore(self.SCORES['W']['KING'])

        return

    def Step(self):
        # function to be implemented by children
        stats = { 'nodes_evaluated': 0, 'max_depth': 2 }
        score, move, stats = self.Minimax(self.board, depth=0, max_depth=stats['max_depth'], isMaximizer=self.board.turn, stats=stats)
        stats['predicted_score'] = score
        print("result: move=", move, "; score=", score, "; stats=", stats)
        self.board.push(move)
        return move, self.board, stats

    def Evaluate(self, board):
        # evaluation function
        if board.is_checkmate():
            if board.turn:
                return self.MAX_SCORE
            else:
                return -self.MAX_SCORE
        elif board.is_game_over():
            # if game ended without checkmate, then it's a draw
            return 0

        if board.turn:
            player = 'W'
        else:
            player = 'B'

        score = 0
        for square in range(64):
            piece = board.piece_at(square)
            if piece is None:
                continue
            if piece.piece_type == 1:
                piece_score = 100 + self.SCORES[player]['PAWN'][square]
            elif piece.piece_type == 2:
                piece_score = 320 + self.SCORES[player]['KNIGHT'][square]
            elif piece.piece_type == 3:
                piece_score = 330 + self.SCORES[player]['BISHOP'][square]
            elif piece.piece_type == 4:
                piece_score = 500 + self.SCORES[player]['ROOK'][square]
            elif piece.piece_type == 5:
                piece_score = 900 + self.SCORES[player]['QUEEN'][square]
            elif piece.piece_type == 6:
                piece_score = 20000 + self.SCORES[player]['KING'][square]

            if piece.color:
                # white
                score += piece_score
            else:
                # black
                score -= piece_score

        return score

    def Minimax(self, board, depth=0, max_depth=1, isMaximizer=True, stats = {}):
        # when reaching a leaf node, return its evaluation
        if depth >= max_depth or board.is_game_over():
            stats['nodes_evaluated'] += 1
            return self.Evaluate(board), None, stats

        # evaluate next moves up to a certain depth
        best_move = None

        if isMaximizer:
            # player is maximizer
            best_score = -999999
            for move in board.legal_moves:
                board.push(move)
                score, _, stats = self.Minimax(board, depth+1, max_depth, not isMaximizer, stats)
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move
        else:
            # player is minimizer
            best_score = +999999
            for move in board.legal_moves:
                board.push(move)
                score, _, stats = self.Minimax(board, depth+1, max_depth, not isMaximizer, stats)
                board.pop()
                if score < best_score:
                    best_score = score
                    best_move = move

        # print("depth=", depth, "/", max_depth, "; best_move=", best_move, "; best_score=", best_score)
        return best_score, best_move, stats

    def MirrorScore(self, scores):
        return scores[56:64] \
               + scores[48:56] \
               + scores[40:48] \
               + scores[32:40] \
               + scores[24:32] \
               + scores[16:24] \
               + scores[8:16] \
               + scores[0:8]

    def Copy(self):
        return Engine()

# request handler
def handleRequest(context):
    engine = Engine()
    return engine.HandlePostRequest(context)