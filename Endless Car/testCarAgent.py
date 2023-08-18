from caragent import Agent
from car import CARGAME
import torch
from plot import Plot


def test():

    scores = []
    scoreMean = []
    scoreTotal = 0
    record = 0
    agent = Agent()
    game = CARGAME()
    graph = Plot()
    agent.model.load_state_dict(torch.load("carAIModel.pth"))
    agent.model.eval()

    while agent.numberOfGames < 20:
            
        action = [0,0,0]
        state = agent.getGameState(game)
        stateTensor = torch.tensor(state, dtype=torch.float)
        stateTensor = stateTensor.to(agent.device)
        predictedAction = agent.model(stateTensor)
        action[torch.argmax(predictedAction).item()] = 1
        gameOver, score, reward = game.performAction(action)

        if gameOver:
            game.resetGame()
            agent.numberOfGames += 1

            print('Game', agent.numberOfGames, 'Score', score, 'Record:', record)

            scores.append(score)
            scoreTotal += score
            scoreMean.append(scoreTotal / agent.numberOfGames)

            graph.plot('Testing Endless Car Game Agent', scores, scoreMean)



if __name__ == '__main__':
    test()