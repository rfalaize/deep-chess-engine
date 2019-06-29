'''
Arena class to pit models against each other
'''

import chess
import multiprocessing as mp
import datetime


def PlayGame(player1, player2, verbose=False):
    # Executes one episode of the game. Returns +1 if player1 won, -1 if player2 won, 0 otherwise
    players = {True: player1, False: player2}
    cur_player = True
    board = chess.Board()
    i = 0
    while not board.is_game_over():
        i += 1
        if verbose:
            print("-------- turn", str(i), "player", str(cur_player), "-------- turn")
            # print(self.board)
        # get action
        players[cur_player].board = board
        move, board, _ = players[cur_player].Step()
        # move on to next player
        cur_player = not cur_player

    results = {}
    result = 0
    if board.is_checkmate():
        if board.turn:
            result = -1
        else:
            result = 1
    if verbose:
        print("game over: turn=", str(i), "result=", str(result), "moves=", board.move_stack)
        print(board)
    results['score'] = result
    results['moves'] = board.move_stack
    return results


def PlayGameFromQueue(process_name, tasks, outputs):
    while True:
        print("process", process_name, "picked up task")

        task = tasks.get()
        if task == -1:
            # no task left in queue
            outputs.put(-1)
            break
        (player1, player2) = task
        output = PlayGame(player1, player2)
        outputs.put(output)
    return


def PlayGames(player1, player2, num_games):
    # spawn 1 process per core, i.e use 100% cpu capacity
    st = datetime.datetime.now()
    num_processes = min(mp.cpu_count(), num_games)
    print("****************** START ******************")
    print(player1.engineName, " VS ", player2.engineName)
    print("play", num_games, "games on", num_processes, "parallel processes...")

    # define IPC manager
    manager = mp.Manager()

    # define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    outputs = manager.Queue()

    # initiate the worker processes
    processes = []
    for i in range(num_processes):
        # set process name
        process_name = 'P%i' % i
        # create the process, and connect it to the worker function
        new_process = mp.Process(target=PlayGameFromQueue, args=(process_name, tasks, outputs))
        # add new process to the list of processes
        processes.append(new_process)
        # start the process
        new_process.start()

    # fill task queue
    for game in range(num_games):
        tasks.put((player1.Copy(), player2.Copy()))

    # quit the worker processes by sending them -1 signal
    for i in range(num_processes):
        tasks.put(-1)

    # read calculation results
    num_finished_processes = 0
    results = []
    while True:
        # read result
        new_result = outputs.get()

        # have a look at the results
        if new_result == -1:
            # process has finished
            num_finished_processes += 1

            if num_finished_processes == num_processes:
                break
        else:
            # Output result
            print('Result:' + str(new_result))
            results.append(new_result)

    et = datetime.datetime.now()
    total_time = (et - st).total_seconds()
    print("finished in", total_time, "s")

    print("results:")
    wins = 0
    for result in results:
        if result == 1:
            wins += 1

    wins /= len(results)
    print(player1.engineName, "won {0:.0%}".format(wins), "of the games.")
    print("****************** END ******************")
    return wins
