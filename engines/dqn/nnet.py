# DQN Model

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
from utils.meter import AverageMeter
from utils.progress.bar import Bar
import time
import os

# neural network guiding tree search
# ************************************************************************
class ChessNet(nn.Module):

    def __init__(self):
        super(ChessNet, self).__init__()

        # inputs: 6 pieces * 2 colors * 8*8 binary feature maps + 1 (turn)
        self.input_size = 13 * 64
        # outputs: 64 'from' squares, 64 'to' squares
        self.action_size = 64 * 64

        # model
        # *********************************************
        self.fc1 = nn.Linear(self.input_size, 1024)
        # self.fc1_bn = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 1024)
        # self.fc2_bn = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 1024)
        # self.fc3_bn = nn.BatchNorm1d(1024)

        # return move probabilities and estimated score
        self.out1 = nn.Linear(1024, self.action_size)
        self.out2 = nn.Linear(1024, 1)

    def forward(self, s):
        s = Variable(torch.from_numpy(s)).float()
        s = f.relu(self.fc1(s))
        s = f.relu(self.fc2(s))
        s = f.relu(self.fc3(s))
        pi = f.relu(self.out1(s))
        v = f.relu(self.out2(s))

        return f.softmax(pi, dim=0), torch.tanh(v)

    def train(self, examples, batch_size, cuda=True):
        # examples: list of examples of the form (board, pi, v)

        optimizer = optim.Adam(lr=0.001)

        for epoch in range(10):
            print("Epoch", epoch, "...")
            #self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples) / batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples)/batch_size):
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis).astype(np.float64))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                # predict
                if cuda:
                    boards, target_pis, target_vs = \
                        boards.contiguous().cuda(), \
                        target_pis.contiguous().cuda(), \
                        target_vs.contiguous().cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.forward(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                # compute gradient
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                    batch=batch_idx,
                    size=int(len(examples) / batch_size),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                bar.next()

            bar.finish()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return -torch.sum((targets - outputs.view(-1))**2) / targets.size()[0]

    def save_checkpoint(self, folder='~/temp/checkpoint', filename='checkpoint.pth.tar'):
        file_path = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist; making directory {}".format(folder))
            os.mkdir(folder)
        torch.save({
            'state_dict' : self.state_dict(),
        }, file_path)

    def load_checkpoint(self, folder='~/temp/checkpoint', filename='checkpoint.pth.tar'):
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            raise("No model in path {}".format(file_path))
        map_location = None if self.cuda else 'cpu'
        checkpoint = torch.load(file_path, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])


# encoder to convert board into tensor
# ************************************************************************
class BoardEncoder:

    def EncodeBoard(self, board):

        # encode the board into binary feature maps
        feature_maps = []

        # piece positions
        # *****************************************************
        d = {}
        for color in (chess.WHITE, chess.BLACK):
            d[color] = {}
            d[color][1] = np.zeros(64)  # pawns
            d[color][2] = np.zeros(64)  # knights
            d[color][3] = np.zeros(64)  # bishops
            d[color][4] = np.zeros(64)  # rooks
            d[color][5] = np.zeros(64)  # queens
            d[color][6] = np.zeros(64)  # king
            for key in d[color]:
                feature_maps.append(d[color][key])

        for square in range(64):
            piece = board.piece_at(square)
            if piece is None:
                continue
            d[piece.color][piece.piece_type][square] = 1

        # additional features
        # *****************************************************

        # current turn
        if board.turn == chess.WHITE:
            feature_maps.append(np.ones(64))
        else:
            feature_maps.append(np.zeros(64))

        result = np.stack(feature_maps).ravel()
        return result

    def EncodeLegalMoves(self, board):
        # generate a binary mask of legal moves
        mask = np.zeros([64, 64])
        for move in board.legal_moves:
            mask[move.from_square, move.to_square] = 1
        mask = mask.ravel()
        return mask

    def DecodeLegalMovesProbas(self, board, probas):
        # input:
        #   board
        #   probabilities as 4096*1 array

        # decode and unpack from-to squares
        legal_moves_probas = {}
        probas = probas.reshape(64, 64)
        sum_probas = 0
        for move in board.legal_moves:
            from_square = move.from_square
            to_square = move.to_square
            proba = probas[from_square, to_square]
            if proba == 0:
                # increase a bit exploration
                proba = 0.001
            legal_moves_probas[(from_square, to_square)] = proba
            sum_probas += proba

        # rescale probas to that they sum to 1
        if sum_probas == 0:
            # WARN: this can happen if network lacks predicting power, or overfits
            print('All legal moves had 0 proba')
            equi_proba = 1 / len(legal_moves_probas)
            for key in legal_moves_probas:
                legal_moves_probas[key] = equi_proba
        else:
            for key in legal_moves_probas:
                legal_moves_probas[key] /= sum_probas

        return legal_moves_probas


if __name__ == '__main__':
    board = chess.Board()
    encoder = BoardEncoder()
    # encode board
    encodedBoard = encoder.EncodeBoard(board)
    assert (832,) == encodedBoard.shape
    # encode legal moves
    mask = encoder.EncodeLegalMoves(board)
    assert (4096,) == mask.shape
    assert 20 == np.sum(mask)
    # decode moves
    probas = np.zeros([64, 64])
    probas[12, 28] = 0.3    # e2e4
    probas[6, 21] = 0.5     # g1f3
    probas[10, 34] = 0.2    # c2c5 (illegal)
    decodedMoves = encoder.DecodeLegalMovesProbas(board, probas.ravel())
    assert decodedMoves[(6, 21)] >= 0.611 and decodedMoves[(6, 21)] <= 0.612
    exit(0)
