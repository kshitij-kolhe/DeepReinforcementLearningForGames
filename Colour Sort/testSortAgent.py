from agent import Agent
from sorter import Sorter
import torch
from plot import Plot


def test():

    scores = []
    scoreMean = []
    scoreTotal = 0
    record = 0
    agent = Agent()
    game = Sorter()
    graph = Plot()
    agent.brain.load_state_dict(torch.load("colourAIModel.pth"))
    agent.brain.eval()

    while agent.numberOfGames < 20:
            
        action = [0,0,0,0]
        state = agent.getGameState(game)
        stateTensor = torch.tensor(state, dtype=torch.float)
        stateTensor = stateTensor.to(agent.gpu)
        predictedAction = agent.brain(stateTensor)
        action[torch.argmax(predictedAction).item()] = 1
        gameOver, score, reward = game.performAction(action)

        if gameOver:
            game.resetGame()
            agent.numberOfGames += 1

            print('Game', agent.numberOfGames, 'Score', score, 'Record:', record)

            scores.append(score)
            scoreTotal += score
            scoreMean.append(scoreTotal / agent.numberOfGames)

            graph.plot('Testing Colour Sort Game Agent', scores, scoreMean)



if __name__ == '__main__':
    test()