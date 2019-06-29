from ...core import CoreEngine
import operator

class Engine(CoreEngine):

    def __init__(self):
        CoreEngine.__init__(self, 'minimax.v3')

        # Constants
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

        # piece values
        self.piece_scores = {1: 100, 2: 320, 3: 330, 4:500, 5:900, 6:2000}

        # variable to monitor training
        self.nodes_count = 0
        self.nodes_evaluated = 0
        return

    def Step(self):
        # function to be implemented by children
        self.nodes_count = 0
        score, move = self.Minimax(self.board, depth=0, max_depth=6,
                                          alpha=-9999, beta=9999,
                                          isMaximizer=self.board.turn)
        stats = {}
        stats['nodes_count'] = self.nodes_count
        stats['nodes_evaluated'] = self.nodes_evaluated
        stats['predicted_score'] = score

        self.board.push(move)
        return move, self.board, stats

    def Evaluate(self, board):
        # evaluation function
        if board.is_checkmate():
            if board.turn:
                return -9999
            else:
                return 9999
        elif board.is_game_over():
            # if game ended without checkmate, then it's a draw
            return 0

        if board.turn:
            player = 'W'
        else:
            player = 'B'
        player_score = self.SCORES[player]
        pawn_score = player_score['PAWN']
        knight_score = player_score['KNIGHT']
        bishop_score = player_score['BISHOP']
        rook_score = player_score['ROOK']
        queen_score = player_score['QUEEN']
        king_score = player_score['KING']

        score = 0
        for square in range(64):
            piece = board.piece_at(square)
            if piece == None:
                continue
            if piece.piece_type == 1:
                piece_score = self.piece_scores[1] + pawn_score[square]
            elif piece.piece_type == 2:
                piece_score = self.piece_scores[2] + knight_score[square]
            elif piece.piece_type == 3:
                piece_score = self.piece_scores[3] + bishop_score[square]
            elif piece.piece_type == 4:
                piece_score = self.piece_scores[4] + rook_score[square]
            elif piece.piece_type == 5:
                piece_score = self.piece_scores[5] + queen_score[square]
            elif piece.piece_type == 6:
                piece_score = self.piece_scores[6] + king_score[square]

            if piece.color:
                # white
                score += piece_score
            else:
                # black
                score -= piece_score

        return score

    def Minimax(self, board, depth=0, max_depth=1, alpha=-1000000, beta=1000000, isMaximizer=True):
        self.nodes_count +=1

        # when reaching a leaf node, return its evaluation
        if (depth>=max_depth) or (board.is_game_over()):
            self.nodes_evaluated += 1
            return self.Evaluate(board), None

        # evaluate next moves up to a certain depth
        best_move = None

        # sort moves so that the ones that capture pieces are analyzed first
        # ******************************************************************
        legal_moves_values = {}
        for move in board.legal_moves:
            piece = board.piece_at(move.to_square)
            if piece != None:
                # use value of the captured opponent piece
                legal_moves_values[move] = self.piece_scores[piece.piece_type]
            else:
                # use value of the piece to move
                # legal_moves_values[move] = self.piece_scores[board.piece_at(move.from_square).piece_type] / 1000
                legal_moves_values[move] = 0
        # capture pieces with highest values first
        legal_moves_sorted = [x[0] for x in sorted(legal_moves_values.items(), key=operator.itemgetter(1), reverse=True)]

        if isMaximizer:
            # player is maximizer
            best_score = (-1)*self.MAX_SCORE

            for move in legal_moves_sorted:
                board.push(move)
                score, _ = self.Minimax(board, depth+1, max_depth, alpha, beta, False)
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move

                    alpha = max(alpha, best_score)
                    if (alpha >= beta):
                        # no need to continue as the minimizer will pick the lower value (beta) somewhere else in the tree
                        # so this branch will be discarded.
                        # print('maximizer pruned node at depth {}: alpha ({}) >= beta ({})'.format(depth, alpha, beta))
                        break;
        else:
            # player is minimizer
            best_score = self.MAX_SCORE
            for move in legal_moves_sorted:
                board.push(move)
                score, _ = self.Minimax(board, depth+1, max_depth, alpha, beta, True)
                board.pop()
                if score < best_score:
                    best_score = score
                    best_move = move

                    beta = min(beta, best_score)
                    if (alpha >= beta):
                        # no need to continue as maximizer will pick the higher value (alpha) somewhere else in the tree
                        # print('minimizer pruned node at depth {}: alpha ({}) >= beta ({})'.format(depth, alpha, beta))
                        break;

        return best_score, best_move

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