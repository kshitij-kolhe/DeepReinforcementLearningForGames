import torch
import random
import numpy as np
from collections import deque
from sorter import Sorter
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from plot import Plot




class AI(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, 100)
        self.linear2 = nn.Linear(100, 200)
        self.linear4 = nn.Linear(200, outputSize)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.log_softmax(self.linear4(x), dim=-1)
        return x

    def save(self, modelName='colourAIModel.pth'):
        torch.save(self.state_dict(), modelName)


class DRLTrainer:
    def __init__(self, gamma, model, lr):
        self.learningRate = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
        self.criterion = nn.MSELoss()

    def putToGPU(self, currentState, nextState, direction, reward):
        gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor(currentState, dtype=torch.float).to(gpu), torch.tensor(nextState, dtype=torch.float).to(gpu), torch.tensor(direction, dtype=torch.long).to(gpu), torch.tensor(reward, dtype=torch.float).to(gpu)

    def train(self, currentState, nextState, direction, reward, gameOver):
        currentState, nextState, direction, reward = self.putToGPU(currentState, nextState, direction, reward)

        if len(currentState.shape) == 1:
            currentState, nextState, direction, reward, gameOver = self.unsqueezeData(currentState, nextState, direction, reward, gameOver)

        pred = self.model(currentState)

        target = pred.clone()
        for idx in range(len(gameOver)):
            newQ = reward[idx]
            if not gameOver[idx]:
                newState0 = nextState[idx].clone()
                newState0 = torch.unsqueeze(newState0, 0)
                newQ = reward[idx] + self.gamma * torch.max(self.model(newState0))
            
            i = torch.argmax(direction[idx]).item()
            target[idx][i] = newQ
        
        self.backPropogation(target, pred)

    def unsqueezeData(self, currentState, nextState, direction, reward, gameOver):
        return torch.unsqueeze(currentState, 0), torch.unsqueeze(nextState, 0), torch.unsqueeze(direction, 0), torch.unsqueeze(reward, 0), (gameOver, ) 

    def backPropogation(self, target, pred):
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return


class Agent:

    def __init__(self):
        self.gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.brain = AI(12, 4).to(self.gpu)
        self.numberOfGames = 0
        self.epsilon = 0
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9 
        self.drlTrainer = DRLTrainer( gamma=self.gamma, model=self.brain, lr=0.0001)

    def storeGameEnvironment(self, currentState, nextState, action, reward, gameOver):
        self.memory.append((currentState, nextState, action, reward, gameOver))

    def trainAction(self, currentState, nextState, action, reward, gameOver):
        currentState, nextState, action, reward = np.array(currentState), np.array(nextState), np.array(action), np.array(reward)
        self.drlTrainer.train(currentState, nextState, action, reward, gameOver)

    def getGameState(self, game: Sorter):
        fieldOfView = np.zeros((12), dtype= int)
        sorter = game.sorter.copy()

        if sorter.y - 20 < 0 or (sorter.x, sorter.y - 20) in game.points:
            fieldOfView[0] = 20
        else:
            fieldOfView[0] = 2
        if sorter.x + 20 > 580 or (sorter.x + 20, sorter.y) in game.points:
            fieldOfView[1] = 20
        else:
            fieldOfView[1] = 2
        if sorter.y + 20 > 580 or (sorter.x, sorter.y + 20) in game.points:
            fieldOfView[2] = 20
        else:
            fieldOfView[2] = 2
        if sorter.x - 20 < 0 or (sorter.x - 20, sorter.y) in game.points:
            fieldOfView[3] = 20
        else:
            fieldOfView[3] = 2
        
        if game.innerSorter == 'orange' and game.sortItem.topleft == (-100, -100):
            fieldOfView[4] = 20 if sorter.y > game.spawnButton.y else 2
            fieldOfView[5] = 20 if sorter.x < game.spawnButton.x else 2
            fieldOfView[6] = 20 if sorter.y < game.spawnButton.y else 2
            fieldOfView[7] = 20 if sorter.x > game.spawnButton.x else 2
        elif game.innerSorter == 'red':
            fieldOfView[4] = 20 if sorter.y > game.redRegion.y else 2
            fieldOfView[5] = 20 if sorter.x < game.redRegion.x else 2
            fieldOfView[6] = 20 if sorter.y < game.redRegion.y else 2
            fieldOfView[7] = 20 if sorter.x > game.redRegion.x else 2
        elif game.innerSorter == 'green':
            fieldOfView[4] = 20 if sorter.y > game.greenRegion.y else 2
            fieldOfView[5] = 20 if sorter.x < game.greenRegion.x else 2
            fieldOfView[6] = 20 if sorter.y < game.greenRegion.y else 2
            fieldOfView[7] = 20 if sorter.x > game.greenRegion.x else 2
        elif game.innerSorter == 'blue':
            fieldOfView[4] = 20 if sorter.y > game.blueRegion.y else 2
            fieldOfView[5] = 20 if sorter.x < game.blueRegion.x else 2
            fieldOfView[6] = 20 if sorter.y < game.blueRegion.y else 2
            fieldOfView[7] = 20 if sorter.x > game.blueRegion.x else 2
        elif game.innerSorter == 'yellow':
            fieldOfView[4] = 20 if sorter.y > game.yellowRegion.y else 2
            fieldOfView[5] = 20 if sorter.x < game.yellowRegion.x else 2
            fieldOfView[6] = 20 if sorter.y < game.yellowRegion.y else 2
            fieldOfView[7] = 20 if sorter.x > game.yellowRegion.x else 2
        elif game.innerSorter == 'orange':
            fieldOfView[4] = 20 if sorter.y > game.sortItem.y else 2
            fieldOfView[5] = 20 if sorter.x < game.sortItem.x else 2
            fieldOfView[6] = 20 if sorter.y < game.sortItem.y else 2
            fieldOfView[7] = 20 if sorter.x > game.sortItem.x else 2

        fieldOfView[8] = 6 if game.direction == [1,0,0,0] else 2
        fieldOfView[9] = 6 if game.direction == [0,1,0,0] else 2
        fieldOfView[10] = 6 if game.direction == [0,0,1,0] else 2
        fieldOfView[11] = 6 if game.direction == [0,0,0,1] else 2

        return fieldOfView

    def trainEpisode(self):
        if len(self.memory) > 1000:
            episode = random.sample(self.memory, 1000)
        else:
            episode = self.memory

        currentStates, nextStates, actions, rewards, gameOvers = zip(*episode)
        currentStates, nextStates, actions, rewards = np.array(currentStates), np.array(nextStates), np.array(actions), np.array(rewards)
        self.drlTrainer.train(currentStates, nextStates, actions, rewards, gameOvers)

    def getAction(self, state):
        self.epsilon = 20 - self.numberOfGames
        finalAction = [0,0,0,0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 200) % 4
            finalAction[move] = 1
        else:
            stateTensor = torch.tensor(state, dtype=torch.float)
            stateTensor = stateTensor.to(self.gpu)
            predictedAction = self.brain(stateTensor)
            finalAction[torch.argmax(predictedAction).item()] = 1

        return finalAction


def train():
    
    scores = []
    scoreMean = []
    scoreTotal = 0
    record = 0
    agent = Agent()
    carGame = Sorter()
    graph = Plot()
    random.seed(454)
    while True:

        # if agent.numberOfGames > 15:
        #     obs = True

        oldState = agent.getGameState(carGame)
        finalAction = agent.getAction(oldState)
        gameOver, score, reward = carGame.performAction(finalAction)
        newState = agent.getGameState(carGame)
        agent.trainAction(oldState, newState, finalAction, reward, gameOver)
        agent.storeGameEnvironment(oldState, newState, finalAction, reward, gameOver)

        if gameOver:
            carGame.resetGame()
            agent.numberOfGames += 1
            agent.trainEpisode()

            if score > record:
                record = score
                agent.brain.save()

            print('Game', agent.numberOfGames, 'Score', score, 'Record:', record)

            scores.append(score)
            scoreTotal += score
            scoreMean.append(scoreTotal / agent.numberOfGames)

            graph.plot('Training Colour Sort Game Agent', scores, scoreMean)




if __name__ == '__main__':
    train()