'''
from engines.arena import PlayGame, PlayGames
from engines.rnd.engine import Engine as EngineRandom
from engines.minimax.v2.engine import Engine as EngineMinimaxV2

def _PlayGame():
    player1 = EngineRandom()
    player2 = EngineMinimaxV2()
    PlayGame(player1, player2, verbose=True)

def test_PlayGames():
    player1 = EngineRandom()
    player2 = EngineMinimaxV2()
    PlayGames(player1, player2, num_games=8)

if __name__ == '__main__':
    print("********************* PIT *************************")
    test_PlayGames()
    print("********************* END *************************")
'''