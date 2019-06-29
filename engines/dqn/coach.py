
import chess
from nnet import ChessNet, BoardEncoder
from mcts import MCTS
from arena import PlayGames
import numpy as np
from collections import deque
from utils.meter import AverageMeter
from utils.progress.bar import Bar
import os
import sys
import time
from random import shuffle
from pickle import Pickler, Unpickler

# Executes self-play + learning.
class Coach:

    def __init__(self):
        self.board = chess.Board()
        self.curPlayer = chess.WHITE
        self.nnet = ChessNet()
        self.pnet = ChessNet()
        self.mcts = None
        self.boardEncoder = BoardEncoder()
        self.trainExamplesHistory = []

        # hyper parameters
        self.numIters = 50
        self.numEpisodes = 3
        self.maxLenHistory = 200000
        self.numItersForTrainExamplesHistory = 2

        # checkpoint
        if os.name == 'posix':
            self.path = '~/temp/deep-chess/checkpoints/'
        else:
            self.path = 'D:/temp/deep-chess/checkpoints/'

    def executeEpisode(self):
        # Execute one episode of self-play, starting with player 1 until end of game.

        episodeSteps = []
        board = self.board
        board.reset()
        self.curPlayer = chess.WHITE
        episodeStep = 0

        while True:
            episodeStep += 1
            # get action probabilities
            encodedBoard = self.boardEncoder.EncodeBoard(self.board)
            pi, moves = self.mcts.getActionProb()
            episodeSteps.append([encodedBoard.copy(), self.curPlayer, pi])

            move_id = np.random.choice(len(pi), p=pi)
            self.board.push(moves[move_id])
            self.curPlayer = not self.board.turn

            if board.is_game_over() or len(self.board.move_stack)>=150:
                if board.is_checkmate():
                    if board.turn:
                        r = -1  # black won
                    else:
                        r = 1   # white won
                else:
                    r = 0
                result = []
                for brd, player, probas in episodeSteps:
                    score = r * (-1) ** (player != self.curPlayer)
                    result.append((brd, probas, score))
                return result

    def learn(self):
        # Perform numMiniBatches iterations with numEpisodes of self-play in each iteration.
        # After each iteration, retain the network with
        # Then, pit the model against its previous version, and keep the new one only if
        # it wins > 60% of the games.

        for i in range(1, self.numIters):
            print('------ITER ' + str(i) + '------')
            iterationTrainExamples = deque([], maxlen=self.maxLenHistory)

            meter = AverageMeter()
            bar = Bar('Self play', max=self.numEpisodes)
            end = time.time()

            for eps in range(self.numEpisodes):
                self.mcts = MCTS(self.board, self.nnet)
                # execute episode and append to memory
                iterationTrainExamples += self.executeEpisode()
                # progress bar
                meter.update(time.time() - end)
                end = time.time()
                bar.suffix = '({eps}/{maxeps})' \
                    'Eps Time: {et:.3f}s ' \
                    '| Total: {total:} '.format(
                        eps=eps + 1,
                        maxeps=self.numEpisodes,
                        et=meter.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td)
                bar.next()
            bar.finish()

            # add iteration samples to history
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                      " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)

            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # suffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # train new model and keep copy of the old one

            self.nnet.save_checkpoint(folder=self.path, filename='temp.nnet.tar')
            self.pnet.load_checkpoint(folder=self.path, filename='temp.nnet.tar')

            self.nnet.train(trainExamples)

            nmcts = MCTS(self.board, self.nnet)
            pmcts = MCTS(self.board, self.pnet)

            print('Pit model against previous version...')
            score = PlayGames(nmcts, pmcts, 100)
            print('new model won', 100*score, '% of games')
            if score >= 0.6:
                print('accept new model')
                self.nnet.save_checkpoint(folder=self.path, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.path, filename='best.nnet.tar')
            else:
                print('reject new model')
                self.nnet.load_checkpoint(folder=self.path, filename='temp.nnet.tar')


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        filename = os.path.join(self.path, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True