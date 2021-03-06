from ...core import CoreEngine


class Engine(CoreEngine):

    def __init__(self):
        CoreEngine.__init__(self, 'minimax.v2')

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
        self.SCORES['B']['PAWN'] = self.mirror_score(self.SCORES['W']['PAWN'])
        self.SCORES['B']['KNIGHT'] = self.mirror_score(self.SCORES['W']['KNIGHT'])
        self.SCORES['B']['BISHOP'] = self.mirror_score(self.SCORES['W']['BISHOP'])
        self.SCORES['B']['ROOK'] = self.mirror_score(self.SCORES['W']['ROOK'])
        self.SCORES['B']['QUEEN'] = self.mirror_score(self.SCORES['W']['QUEEN'])
        self.SCORES['B']['KING'] = self.mirror_score(self.SCORES['W']['KING'])

        # variable to monitor training
        self.nodes_count = 0
        self.nodes_evaluated = 0
        return

    def step(self):
        # function to be implemented by children
        self.nodes_count = 0
        score, move = self.minimax(self.board, depth=0, max_depth=4,
                                          alpha=-9999, beta=9999,
                                          isMaximizer=self.board.turn)
        stats = {}
        stats['nodes_count'] = self.nodes_count
        stats['nodes_evaluated'] = self.nodes_evaluated
        stats['predicted_score'] = score

        self.board.push(move)
        return move, self.board, stats

    def evaluate(self, board):
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
            if piece is None:
                continue
            if piece.piece_type == 1:
                piece_score = 100 + pawn_score[square]
            elif piece.piece_type == 2:
                piece_score = 320 + knight_score[square]
            elif piece.piece_type == 3:
                piece_score = 330 + bishop_score[square]
            elif piece.piece_type == 4:
                piece_score = 500 + rook_score[square]
            elif piece.piece_type == 5:
                piece_score = 900 + queen_score[square]
            elif piece.piece_type == 6:
                piece_score = 2000 + king_score[square]
            else:
                piece_score = 0

            if piece.color:
                # white
                score += piece_score
            else:
                # black
                score -= piece_score

        return score

    def minimax(self, board, depth=0, max_depth=1, alpha=-1000000, beta=1000000, isMaximizer=True):
        self.nodes_count += 1

        # when reaching a leaf node, return its evaluation
        if (depth >= max_depth) or (board.is_game_over()):
            self.nodes_evaluated += 1
            return self.evaluate(board), None

        # evaluate next moves up to a certain depth
        best_move = None

        if isMaximizer:
            # player is maximizer
            best_score = (-1)*self.MAX_SCORE
            for move in board.legal_moves:
                board.push(move)
                score, _ = self.minimax(board, depth+1, max_depth, alpha, beta, False)
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move

                    alpha = max(alpha, best_score)
                    if alpha >= beta:
                        # no need to continue as the minimizer will pick the lower value (beta)
                        # somewhere else in the tree, so this branch will be discarded.
                        # print('maximizer pruned node at depth {}: alpha ({}) >= beta ({})'.format(depth, alpha, beta))
                        break
        else:
            # player is minimizer
            best_score = self.MAX_SCORE
            for move in board.legal_moves:
                board.push(move)
                score, _ = self.minimax(board, depth+1, max_depth, alpha, beta, True)
                board.pop()
                if score < best_score:
                    best_score = score
                    best_move = move

                    beta = min(beta, best_score)
                    if alpha >= beta:
                        # no need to continue as maximizer will pick the higher value (alpha) somewhere else in the tree
                        # print('minimizer pruned node at depth {}: alpha ({}) >= beta ({})'.format(depth, alpha, beta))
                        break

        return best_score, best_move

    def mirror_score(self, scores):
        return scores[56:64] \
               + scores[48:56] \
               + scores[40:48] \
               + scores[32:40] \
               + scores[24:32] \
               + scores[16:24] \
               + scores[8:16] \
               + scores[0:8]
