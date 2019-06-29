import chess
import numpy as np
import math
from nnet import BoardEncoder

class MCTS:

    def __init__(self, board, nnet):
        self.board = board
        self.nnet = nnet

        self.Ns = {}    # stores #times board s was visited
        self.Nsa = {}   # stores #times edge (s,a) was visited
        self.Wsa = {}   # stores total action value for edge (s,a)
        self.Qsa = {}   # stores average action value for edge (s,a)
        self.Psa = {}   # stores prior probability returned by neural net
        self.Vs = {}    # stores valid moves for state s

        self.boardEncoder = BoardEncoder()

        # hyper parameters
        self.numMCTSsims = 2

    def getActionProb(self, temp=1):
        # this function performs numMctsSims simulations
        # returns:
        #   probs: a policy vector where the probability of the ith action
        #           is proportional to Nsa[(s,a)]**(1/temp)
        s = self.board.fen()

        for i in range(self.numMCTSsims):
            self.search(self.board.copy())

        counts = []
        moves = []
        for move in self.board.legal_moves:
            a = (move.from_square, move.to_square)
            if (s, a) in self.Nsa:
                counts.append(self.Nsa[(s, a)])
            else:
                counts.append(0)
            moves.append(move)

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs, moves

        # counts = [x ** (1./temp) for x in temp]
        probs = [x / float(sum(counts)) for x in counts]
        return probs, moves

    def search(self, board):
        '''
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that has
        the maximum upper confidence bound.

        Once a leaf node is found, 2 options:
        1) leaf node is a terminal node; in that case, calculate the score
        2) leaf node is not terminal node; in that case the neural network is called
            to return an initial policy P(s,a) and a value v for the state.

        In both cases, value is propagated up the search path.
        The values of Ns, Nsa, Wsa and Qsa are then updated.

        Note: the values are the negative of the values of the current state.
        This is done since v in [-1; 1], and if v is the value of a state for
        the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current board
        '''

        s = board.fen()

        # terminal node
        # **********************************************************************
        if board.is_game_over():
            if board.is_checkmate():
                if board.turn == chess.WHITE:
                    r = -1
                else:
                    r = 1
            else:
                r = 0
            return (-1) * r

        # leaf node
        # **********************************************************************
        if s not in self.Ns:
            # leaf node
            encodedBoard = self.boardEncoder.EncodeBoard(board)
            probas, v = self.nnet.forward(encodedBoard)
            probas = probas.data.numpy()
            v = v.data.numpy()[0]

            # get legal moves with probabilities
            move_probas = self.boardEncoder.DecodeLegalMovesProbas(board, probas)

            for a in move_probas:
                self.Nsa[(s, a)] = 0
                self.Qsa[(s, a)] = 0
                self.Wsa[(s, a)] = 0
                self.Psa[(s, a)] = move_probas[a]

            self.Ns[s] = 0
            self.Vs[s] = list(move_probas.keys())   # cache valid moves to speed-up
            return -v

        # pick the action with the highest upper confidence bound
        # **********************************************************************
        valid_moves = self.Vs[s]
        best_UCB = -float('inf')
        best_act = -1
        for a in valid_moves:
            # exploration factor
            Usa = 1.0 * self.Psa[(s, a)] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            # upper confidence bound
            UCB = self.Qsa[(s, a)] + Usa

            if UCB > best_UCB:
                best_UCB = UCB
                best_act = a

        a = best_act

        # continue search down the tree until a leaf or terminal node is found
        # **********************************************************************
        board.push(chess.Move(a[0], a[1]))

        v = self.search(board)

        board.pop()

        # propagate value up the tree
        # **********************************************************************
        self.Nsa[(s, a)] += 1
        self.Wsa[(s, a)] += v
        self.Qsa[(s, a)] += self.Wsa[(s, a)] / self.Nsa[(s, a)]
        self.Ns[s] += 1

        '''
        if (s, a) in self.Qsa:
            self.Nsa[(s, a)] += 1
            self.Qsa[(s, a)] = (self.Nsa[(s, a)]*self.Qsa[(s, a)] + v)/(self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Nsa[(s, a)] = 1
            self.Wsa[(s, a)] = v
            self.Qsa[(s, a)] = v
        '''

        return -v
