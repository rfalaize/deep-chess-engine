from ...core import CoreEngine
import numpy as np
import pandas as pd
import datetime
import random
import operator

# ********************************************************************
# Node of the game tree
# ********************************************************************
class Node:

    def __init__(self, board):
        # current board position of the board
        self.board = board
        # statistics to keep for training
        self.visits = 0
        self.score = 0
        self.descendants = []

    def get_descendants(self):
        # return descendants of this node, i.e nodes reachable after each each legal move
        if len(self.descendants) == 0:
            # initialize child nodes
            if not self.is_leaf():
                for move in self.board.legal_moves:
                    descendant_board = self.board.copy()
                    descendant_board.push(move)
                    descendant_node = Node(descendant_board)
                    self.descendants.append(descendant_node)
        return self.descendants

    def is_leaf(self):
        return self.board.is_game_over()

    def copy(self):
        copy_board = self.board.copy()
        copy_node = Node(copy_board)
        copy_node.visits = self.visits
        copy_node.score = self.score
        copy_node.descendants = self.get_descendants()
        return copy_node


# ********************************************************************
# Monte carlo tree search engine
# ********************************************************************
class Engine(CoreEngine):

    def __init__(self):
        CoreEngine.__init__(self, 'mcts.v1')

        # hyper parameters
        self.max_search_time = 10       # in seconds
        self.max_tree_searches = 0      # maximum tree searches
        self.max_simulation_depth = 10  # limit the simulation depth

        # helpers during monte carlo tree search
        self.root_node_player = True    # player to play
        self.tree_search_branch = []    # visited nodes during one round of tree search

        # tree and rollout policies
        self.policies = None

        # metrics to keep traversing the tree
        self.stats = {}

        return

    def Step(self):
        # function to be implemented by children
        root_node = Node(self.board)
        # fix seed to reproduce results
        move, stats = self.MCTS(root_node,
                                max_tree_searches=0,
                                max_search_time=30,
                                max_simulation_depth=1,
                                random_seed=None)
        self.board.push(move)
        return move, self.board, stats

    def MCTS(self, root_node, max_tree_searches=0, max_search_time=30, max_simulation_depth=1, random_seed=None):
        st = datetime.datetime.now()
        self.policies = Policies(root_node.board.turn, random_seed)
        self.root_node_player = root_node.board.turn
        self.max_tree_searches = max_tree_searches
        self.max_search_time = max_search_time
        self.max_simulation_depth = max_simulation_depth

        print("White to play?", self.root_node_player)
        tree_search_count = 1
        root_node.visits = 1

        while True:
            # ####################################################################
            # one round of tree search
            # ####################################################################

            if (self.max_tree_searches > 0) and (tree_search_count > self.max_tree_searches):
                break

            current_node = root_node                    # start each search from the root node
            self.tree_search_branch = [current_node]    # initialize list of visited nodes

            if tree_search_count == 1 or tree_search_count % 500 == 0:
                checkpoint_stats = {}
                for descendant in current_node.get_descendants():
                    checkpoint_stats[str(descendant.board.peek())] = descendant.visits
                checkpoint_stats = sorted(checkpoint_stats.items(), key=operator.itemgetter(1), reverse=True)
                print("tree search", tree_search_count, "... scores=", checkpoint_stats)

            tree_search_count += 1

            # 1) Selection
            # ********************************************************************
            while True:
                if current_node.is_leaf():
                    # leaf found
                    break
                # get next node using the tree policy
                next_node = self.policies.SelectionPolicy(current_node, epsilon=0.1, score_type='observed')
                self.tree_search_branch += [next_node]
                current_node = next_node
                if next_node.visits == 0:
                    # 2) Expansion
                    # ************************************************************
                    break

            # if str(current_node.board.peek()) == "e6f5":
            #    print("STOP")

            # 3) Simulation
            # ********************************************************************
            # start simulation from the unexplored node (i.e. the current_node), and create a simulation branch
            # from there. This branch is not stored as we are only interested about the end result, i.e the leaf.
            simulated_branch_depth = 0
            simulated_node = current_node.copy()
            while not simulated_node.is_leaf():
                # select next node in a random manner (light play out)
                # following the probability distribution provided by the roll-out policy
                simulated_node = self.policies.SelectionPolicy(simulated_node, epsilon=0.0, score_type='policy')
                simulated_branch_depth += 1
                if simulated_branch_depth >= self.max_simulation_depth:
                    # stop simulation after max depth
                    break

            # 4) Back propagation
            # ********************************************************************
            # evaluate the simulated node for the root player
            simulated_score = self.policies.WeightedScore(simulated_node)

            # update all nodes visited during tree search
            for visited_node in self.tree_search_branch:
                visited_node.visits += 1
                visited_node.score += simulated_score

            # check if there is time left
            # ********************************************************************
            if self.max_search_time > 0 and (tree_search_count % 10) == 0:
                et = datetime.datetime.now()
                elapsed_time = (et - st).total_seconds()
                if elapsed_time > self.max_search_time:
                    print("elapsed time > max search time => stop tree search")
                    break

        # ####################################################################
        # 5) Move selection
        # ####################################################################
        max_visits = -1
        # max_score = -np.inf
        next_node = None

        print("******************** End search. Next move selection *******************")
        checkpoint_results = {'node': [], 'visits': [], 'score': [], 'choice': []}
        for descendant in root_node.get_descendants():
            checkpoint_results['node'].append(descendant.board.peek())
            checkpoint_results['visits'].append(descendant.visits)
            checkpoint_results['score'].append(descendant.score/max(descendant.visits, 1))
            checkpoint_results['choice'].append('')
            if descendant.visits > max_visits:
                max_visits = descendant.visits
                next_node = descendant
        checkpoint_results = pd.DataFrame(checkpoint_results).sort_values('visits', ascending=False)
        checkpoint_results = checkpoint_results.set_index('node')
        checkpoint_results['choice'].iloc[0] = '<==='
        print(checkpoint_results)
        print("******************** End *******************")

        et = datetime.datetime.now()
        self.stats['search_time'] = (et - st).total_seconds()
        self.stats['tree_search_count'] = tree_search_count
        self.stats['node_visits'] = next_node.visits
        self.stats['node_avg_score'] = next_node.score / max(descendant.visits, 1)
        next_move = next_node.board.peek()
        print("next move", next_move)
        return next_move, self.stats

    def Copy(self):
        return Engine()

# ********************************************************************
# Policy
# ********************************************************************
class Policies:

    def __init__(self, root_player_turn, random_seed=None):
        # store root player
        self.root_player_turn = root_player_turn

        # hyper-parameters
        self.C = 1.0  # can be increase to favor exploration
        self.random_seed = random_seed

        # piece values
        self.piece_scores = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 1800}

        # *************************************************
        # piece square table evaluation
        # source: https://www.chessprogramming.org/Simplified_Evaluation_Function
        # *************************************************
        self.SCORES = {True: {}, False: {}}

        # pawns
        self.SCORES[True][1] = \
            [ 0,  0,   0,   0,   0,   0,  0,  0,
              5, 10,  10, -20, -20,  10, 10,  5,
              5, -5, -10,   0,   0, -10, -5,  5,
              0,  0,   0,  20,  20,   0,  0,  0,
              5,  5,  10,  25,  25,  10,  5,  5,
             10, 10,  20,  30,  30,  20, 10, 10,
             50, 50,  50,  50,  50,  50, 50, 50,
              0,  0,   0,   0,   0,   0,  0,  0]

        self.SCORES[False][1] = \
            [ 0,  0,   0,   0,   0,   0,  0,  0,
             50, 50,  50,  50,  50,  50, 50, 50,
             10, 10,  20,  30,  30,  20, 10, 10,
              5,  5,  10,  25,  25,  10,  5,  5,
              0,  0,   0,  20,  20,   0,  0,  0,
              5, -5, -10,   0,   0, -10, -5,  5,
              5, 10,  10, -20, -20,  10, 10,  5,
              0,  0,   0,   0,   0,   0,  0,  0]

        # knights
        self.SCORES[True][2] = \
            [-50, -40, -30, -30, -30, -30, -40, -50,
             -40, -20,   0,   0,   0,   0, -20, -40,
             -30,   0,  10,  15,  15,  10,   0, -30,
             -30,   0,  15,  20,  20,  15,   0, -30,
             -30,   0,  15,  20,  20,  15,   0, -30,
             -30,   0,  10,  15,  15,  10,   0, -30,
             -40, -20,   0,   0,   0,   0, -20, -40,
             -50, -40, -30, -30, -30, -30, -40, -50]

        self.SCORES[False][2] = self.SCORES[True][2]

        # bishops
        self.SCORES[True][3] = \
            [-20, -10, -10, -10, -10, -10, -10, -20,
             -10,  10,   0,   0,   0,   0,  10, -10,
             -10,  10,  10,  10,  10,  10,  10, -10,
             -10,   0,  10,  10,  10,  10,   0, -10,
             -10,   5,   5,  10,  10,   5,   5, -10,
             -10,   0,   5,  10,  10,   5,   0, -10,
             -10,   0,   0,   0,   0,   0,   0, -10,
             -20, -10, -10, -10, -10, -10, -10, -20]

        self.SCORES[False][3] = \
            [-20, -10, -10, -10, -10, -10, -10, -20,
             -10,   0,   0,   0,   0,   0,   0, -10,
             -10,   0,   5,  10,  10,   5,   0, -10,
             -10,   5,   5,  10,  10,   5,   5, -10,
             -10,   0,  10,  10,  10,  10,   0, -10,
             -10,  10,  10,  10,  10,  10,  10, -10,
             -10,  10,   0,   0,   0,   0,  10, -10,
             -20, -10, -10, -10, -10, -10, -10, -20]

        # rooks
        self.SCORES[True][4] = \
            [ 0,  5,  0, 20, 20,  0,  5,  0,
              0,  0,  0,  0,  0,  0,  0,  0,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
             30, 50, 50, 50, 50, 50, 50, 30,
              0,  0,  0,  0,  0,  0,  0,  0]

        self.SCORES[False][4] = \
            [ 0,  0,  0,  0,  0,  0,  0,  0,
             30, 50, 50, 50, 50, 50, 50, 30,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  5,  0, 20, 20,  0,  5,  0]

        # queens
        self.SCORES[True][5] = \
            [-20, -10, -10, -5, -5, -10, -10, -20,
             -10,   0,   5,  0,  0,   0,   0, -10,
             -10,   5,   5,  5,  5,   5,   0, -10,
              -5,   0,   5,  5,  5,   5,   0,  -5,
               0,   0,   5,  5,  5,   5,   0,  -5,
             -10,   0,   5,  5,  5,   5,   0, -10,
             -10,   0,   0,  0,  0,   0,   0, -10,
             -20, -10, -10, -5, -5, -10, -10, -20]

        self.SCORES[False][5] = self.SCORES[True][5]

        # king
        self.SCORES[True][6] = \
            [ 20,  30,  10,   0,   0,  10,  30,  20,
              20,  20,   0,   0,   0,   0,  20,  20,
             -10, -20, -20, -20, -20, -20, -20, -10,
             -20, -30, -30, -40, -40, -30, -30, -20,
             -30, -40, -40, -50, -50, -40, -40, -30,
             -30, -40, -40, -50, -50, -40, -40, -30,
             -30, -40, -40, -50, -50, -40, -40, -30,
             -30, -40, -40, -50, -50, -40, -40, -30]

        self.SCORES[False][6] = \
            [-30, -40, -40, -50, -50, -40, -40, -30,
             -30, -40, -40, -50, -50, -40, -40, -30,
             -30, -40, -40, -50, -50, -40, -40, -30,
             -30, -40, -40, -50, -50, -40, -40, -30,
             -20, -30, -30, -40, -40, -30, -30, -20,
             -10, -20, -20, -20, -20, -20, -20, -10,
              20,  20,   0,   0,   0,   0,  20,  20,
              20,  30,  10,   0,   0,  10,  30,  20]

    # Scoring functions (>0 is white leads, <0 if black leads)
    # ********************************************************************
    def ObservedScore(self, node):
        if node.visits == 0:
            return 0
        return node.score / node.visits

    def PolicyScore(self, node):
        # score of a position estimated using handcrafted knowledge
        # score has to be higher if root player is leading
        score = 0
        for square in range(64):
            piece = node.board.piece_at(square)
            if piece is None:
                continue
            piece_score = self.piece_scores[piece.piece_type] + self.SCORES[piece.color][piece.piece_type][square]
            if piece.color:
                score += piece_score    # white
            else:
                score -= piece_score    # black

        if not self.root_player_turn:
            # score for black is opposite
            score *= (-1)

        return score

    def WeightedScore(self, node, eta=0.5):
        # eta is a parameter to control the incorporation of game knowledge
        observed_score = self.ObservedScore(node)
        policy_score = self.PolicyScore(node)
        return eta * observed_score + (1-eta) * policy_score

    # Selection policy
    # ********************************************************************
    def SelectionPolicy(self, node, epsilon=0.0, score_type='weighted'):
        descendants = node.get_descendants()

        np.random.seed(self.random_seed)
        if epsilon > 0 and np.random.random() < epsilon:
            # choose a random node with probability epsilon
            np.random.seed(self.random_seed)
            return np.random.choice(descendants)

        # probabilities are the estimated scores of each child node, transformed by a soft-max function.
        scores = []
        scaling = 25
        for descendant in descendants:
            # score that root player leads
            if score_type == 'weighted':
                score = self.WeightedScore(descendant)
            elif score_type == 'observed':
                score = self.ObservedScore(descendant)
            elif score_type == 'policy':
                score = self.PolicyScore(descendant)
            else:
                raise ValueError('score_type=', score_type, ' not supported.')

            # scale so that it gives nice probabilities with soft-max
            score /= scaling
            if not (node.board.turn == self.root_player_turn):
                # if it is the opponent's turn, he is more likely to pick that move that minimizes
                # the root player's score. To reflect this, we flip the score so that
                # low score => high probability to be picked
                score = (-1) * score
            scores.append(score)

        # soft-max activation
        exp_scores = np.exp(np.array(scores))
        probabilities = exp_scores / exp_scores.sum()
        # [x for x in zip([str(x.board.peek()) for x in descendants], [x*scaling for x in scores], probabilities)]
        np.random.seed(self.random_seed)
        next_node = np.random.choice(descendants, p=probabilities)
        return next_node

    # Tree policy
    # ********************************************************************
    def TreePolicyUCBGreedy(self, node, epsilon=0.5):
        # tree policy using upper confidence bound
        # AND epsilon greedy to increase exploration further
        descendants = node.get_descendants()

        if random.random() < epsilon:
            # choose a random node with probability epsilon
            return random.choice(descendants)

        best_ucb = -np.inf
        next_node = None

        for descendant in descendants:

            if descendant.visits == 0:
                # if the descendant has never been visited, we have to explore it
                next_node = descendant
                break

            # calculate upper confidence bound of the descendant
            ucb_exploitation = descendant.score / descendant.visits
            ucb_exploitation = (500 + ucb_exploitation) / 100
            ucb_exploration = self.C * np.sqrt(2 * np.log(node.visits) / descendant.visits)
            # print("UCB", "{0:.3f}".format(ucb_exploitation), "{0:.3f}".format(ucb_exploration))
            ucb = ucb_exploitation + ucb_exploration
            if ucb > best_ucb:
                best_ucb = ucb
                next_node = descendant

        return next_node

    def TreePolicyEpsilonGreedy(self, node, epsilon=0.5):
        # epsilon greedy tree policy
        descendants = node.get_descendants()

        if random.random() <= epsilon:
            # choose a random node with probability epsilon
            return random.choice(descendants)

        # otherwise return the descendant with the best score
        best_score = -np.inf
        next_node = None

        for descendant in node.get_descendants():
            if descendant.visits == 0:
                # node has never been visited
                # let's select it to force a minimum exploration?
                return descendant

            descendant_score = descendant.score / descendant.visits
            if descendant_score > best_score:
                best_score = descendant_score
                next_node = descendant

        return next_node


# request handler
def handleRequest(context):
    engine = Engine()
    return engine.HandlePostRequest(context)